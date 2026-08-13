# CosyVoice3-0.5B generation architecture (Phase 1)

Date: 2026-08-11

Local source snapshot: `061ba5b81e32eaf408ef5dfc2f789f1f819f04b7`

H100 checkout: `feature/cosyvoice-api-service@732e30184af72f921fd3f8e73fe1e483e5b40011`
Model inspected: `Fun-CosyVoice3-0.5B` on H100, vLLM backend

This document is an observation-only architecture inventory. No inference or
model code was modified for this phase.

## 1. End-to-end inference path

The deployed API follows this path:

```text
HTTP text/voice request
  -> api_server.engine.CosyVoiceEngine
  -> CosyVoice3.inference_zero_shot / inference_instruct2
  -> CosyVoiceFrontEnd
       text normalization and Qwen tokenizer
       reference WAV -> speaker embedding, prompt speech tokens, prompt mel
  -> CosyVoice3Model.tts
  -> CosyVoice3LM.inference
       text embeddings + task markers + prompt speech embeddings
       autoregressive speech-token generation
  -> CausalMaskedDiffWithDiT.inference
       speech tokens -> mel spectrogram
  -> CausalHiFTGenerator.inference
       mel -> waveform
  -> WAV/PCM encoding and HTTP response
```

Relevant source locations:

| Stage | Source |
| --- | --- |
| API dispatch, seed and quality gate | `api_server/engine.py:52-304` |
| text/audio frontend | `cosyvoice/cli/frontend.py:29-218` |
| model selection | `cosyvoice/cli/cosyvoice.py:193-248` |
| token-generation thread and token-to-waveform handoff | `cosyvoice/cli/model.py:101-236`, `391-454` |
| CosyVoice3 autoregressive LM | `cosyvoice/llm/llm.py:480-768` |
| Flow/DiT decoder | `cosyvoice/flow/flow.py`, `cosyvoice/flow/DiT/dit.py` |
| HiFT/vocoder | `cosyvoice/hifigan/generator.py:572-714` |

The model YAML used on H100 is:

`/SharedData/yangyu/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml`

It defines a 24 kHz output, a 25 speech-token/second rate, a `token_mel_ratio`
of 2, a 22-layer Flow DiT, and a causal HiFT vocoder. The LLM section defines
`CosyVoice3LM` with `speech_token_size=6561`, `llm_input_size=896`, and
`mix_ratio=[5, 15]`.

## 2. Frontend and conditioning

`CosyVoiceFrontEnd` performs the following operations:

1. `text_normalize()` optionally applies ttsfrd/wetext normalization and splits
   long Chinese or English text into manageable chunks.
2. `CosyVoice3Tokenizer` is a Qwen2 tokenizer augmented with CosyVoice special
   tokens, including `<|endofprompt|>` and phoneme/event symbols.
3. For zero-shot voice, the reference WAV is processed by three independent
   components: CampPlus speaker embedding, the speech-tokenizer-v3 ONNX model,
   and the 24 kHz mel feature extractor.
4. `add_zero_shot_spk()` caches those reference tensors in `frontend.spk2info`,
   so the API request contains only a server-owned voice ID.

The Phase-2 H100 service log reported `no frontend is avaliable`: the wetext
snapshot lookup failed and ttsfrd was not available. Therefore this particular
runtime did not apply wetext/ttsfrd text normalization. It still applied the
remaining Chinese cleanup and paragraph-splitting code. The log and this
runtime fact must be carried with the dataset; results from a service that
successfully loads wetext are a different experimental condition.

The API's normal zero-shot sequence has the conceptual form:

```text
[SOS] + [prompt text, if present] + [target text] + [TASK] + [prompt speech tokens]
```

The API's cached `default` zero-shot voice has non-empty prompt text. The current
configuration is an assistant prefix, `<|endofprompt|>`, and the transcript of
the reference WAV. Zero-shot voice registration requires both prompt text and
prompt audio. The useful sequence boundaries are therefore:

```text
[SOS]
+ [assistant/instruction prefix + <|endofprompt|>]
+ [reference transcript]
+ [target text]
+ [TASK]
+ [prompt speech tokens]
```

For `inference_instruct2`, the instruction text (including the required
`<|endofprompt|>`) is part of the text span and the prompt speech tokens are
removed from the LLM input by `frontend_instruct2()`. Reference audio still
conditions Flow/HiFT through prompt features and speaker embedding.

There is no separate attention mask that isolates text, prompt speech, or
generated speech. They are one causal decoder sequence. This is important for
the proposed text-grounding experiment: the spans must be reconstructed from
their known lengths, not inferred from a token-type mask.

## 3. CosyVoice3 LM and decode loop

`CosyVoice3LM` is a subclass of `Qwen2LM` (`cosyvoice/llm/llm.py:258-768`).
The special-token IDs are derived from `speech_token_size`:

```text
speech token IDs: 0 .. 6560
SOS:              6561
EOS:              6562
TASK:             6563
fill token:       6564
reserved stop IDs: 6561 .. 6760
```

The actual vLLM export has vocabulary size 6761. During inference:

1. `CosyVoice3LM.inference()` concatenates prompt and target text, embeds it
   with the Qwen embedding table, and appends SOS/TASK/prompt-speech embeddings.
2. `min_len = 2 * target_text_token_count` and
   `max_len = 20 * target_text_token_count` (from the YAML-independent defaults
   in `Qwen2LM.inference`).
3. With vLLM, `inference_wrapper()` submits one `prompt_embeds` request to the
   vLLM `LLMEngine` and repeatedly calls `engine.step()`.
4. vLLM uses `top_k=25`, the reserved stop-token range, the computed min/max
   token limits, and the per-request seed. The current vLLM branch does not
   pass the PyTorch RAS sampler's `top_p`, `win_size`, or `tau_r` into
   `SamplingParams`.
5. Each returned token is appended as the next speech-token embedding. The loop
   stops on a reserved stop ID or at `max_len`.

With the non-vLLM PyTorch path, `inference_wrapper()` instead calls
`Qwen2Encoder.forward_one_step()` with a causal lower-triangular mask and a
Hugging Face `past_key_values` cache. It computes `llm_decoder` logits and uses
`ras_sampling(top_p=0.8, top_k=25, win_size=10, tau_r=0.1)` while suppressing
EOS before `min_len`.

Before Flow, `CosyVoiceModel.llm_job()` suppresses silent tokens after more than
five consecutive tokens from the CosyVoice3 silent-token set. The raw LLM token
trajectory can therefore differ from the token list consumed by Flow. Both
lists and the dropped-token positions must be saved for alignment. The filtered
speech-token list is consumed by `CosyVoice3Model.token2wav()`. Flow converts
tokens to mel features and HiFT converts the mel to waveform. In non-streaming
API mode the whole filtered token list is generated before Flow/HiFT
finalization.

## 4. Actual Qwen2 architecture

The H100 model's exported vLLM configuration is:

```text
model_type:             qwen2
hidden_size:            896
intermediate_size:      4864
num_hidden_layers:      24
num_attention_heads:    14
num_key_value_heads:     2
head_dim:                64  (896 / 14)
max_position_embeddings: 32768
rope_theta:              1000000.0
rope_scaling:            null
use_cache:               true
sliding_window:          null
hidden activation:       SiLU
dtype:                   bfloat16
vocab_size:              6761 (exported CosyVoice speech-token model)
```

The CosyVoice3 export uses vLLM's standard `Qwen2ForCausalLM`. Its speech-token
embedding and bias-free speech-token LM head are installed before
`save_pretrained()`. The export only rewrites the architecture to
`CosyVoice2ForCausalLM` when the decoder has a bias; that condition is false for
CosyVoice3. Therefore `cosyvoice/vllm/cosyvoice2.py` is not the active model
class for the inspected CosyVoice3 export.

The authoritative implementation for this experiment is vLLM `0.11.0`, commit
`gf71952c1c`, installed on H100 at:

```text
/SharedData/yangyu/cosyvoice_vllm_env/cosyvoice/lib/python3.10/site-packages/
vllm/model_executor/models/qwen2.py
```

In that installed file, `Qwen2Attention` starts at line 99,
`Qwen2DecoderLayer` at line 192, and `Qwen2Model` at line 278. The repository's
separate `3rdparty/vllm` checkout is not used as the line-number authority for
this H100 run.

Each layer is pre-norm RMSNorm -> causal self-attention -> RMSNorm -> SiLU
gated MLP, with residual connections. Grouped-query attention is used: 14 Q
heads share 2 K/V heads.

## 5. Attention, RoPE, position IDs and KV cache

### Attention backend

`Qwen2Attention.forward()` computes Q/K/V, applies RoPE, then calls vLLM's
generic `Attention` layer. The H100 startup log observed during the API tests
reported `Using Flash Attention backend on V1 engine`. Therefore the deployed
path is fused FlashAttention/vLLM attention, not an eager PyTorch attention map
that exposes probabilities.

The generic attention layer owns the paged KV-cache operations. The CosyVoice
wrapper submits prompt embeddings to vLLM and does not maintain a Python
`past_key_values` object in that mode.

### RoPE

The model definitely uses RoPE. The config has `rope_theta=1_000_000` and no
scaling. `Qwen2Attention` constructs `get_rope(...)` and applies
`self.rotary_emb(positions, q, k)` immediately before attention. The current
model has no evidence of a custom CosyVoice RoPE variant.

### Position IDs

In vLLM, `positions` is supplied by the vLLM scheduler/model runner. Prompt
embedding positions are expected to be contiguous from zero and decode
positions to advance with sequence length while prior K/V values remain in the
paged cache. The CosyVoice wrapper does not provide custom position IDs or
position interpolation. Exact vLLM 0.11 position construction and block-table
code locations still need to be traced and runtime-logged before position-based
claims are made; this paragraph is an implementation inference, not yet a
runtime observation.

In the PyTorch fallback, the code supplies a causal mask and lets the HF Qwen2
model derive cache positions from `past_key_values`; this path should be
instrumented separately if exact position IDs are required.

### Causal mask

The decoder is causal by model type. The PyTorch fallback explicitly constructs
`torch.tril(...)` masks in `Qwen2LM.inference_wrapper()` and
`inference_bistream()`. vLLM's decoder attention applies the equivalent causal
constraint inside its attention backend.

## 6. What can be instrumented

### Low-risk, first-stage signals

The first analysis pass should instrument `CosyVoice3LM.inference_wrapper()` and
`CosyVoiceModel.llm_job()`:

- generation step and effective position;
- returned token ID and stop reason;
- raw output length, filtered output length, dropped silent-token positions and
  token repetition statistics;
- vLLM seed and sampling parameters;
- vLLM `finished`, `finish_reason` and `stop_reason`, distinguishing EOS,
  another reserved stop token and maximum length;
- if supported by the installed vLLM version, chosen-token and top-k logprobs
  via `SamplingParams(logprobs=...)`.

This wrapper is the correct common boundary for the vLLM token stream and does
not require changing the Flow or HiFT models.

Top-k logprobs are not sufficient for exact vocabulary entropy. Any entropy
computed from a truncated distribution must be labeled as an approximation.
Exact entropy requires sampler logits/full log-sum-exp or an exact
teacher-forced replay.

### PyTorch debug mode

For hidden-state and probability instrumentation, add a separate debug-only
teacher-forced replay path around `Qwen2Encoder.forward_one_step()` and the HF
Qwen2 model. Replay the exact prompt embeddings and exact vLLM-generated token
trajectory rather than freely sampling again. Capture:

- final and per-layer hidden-state norm;
- cosine distance to the previous step;
- logits, entropy, top-1 probability and logit margin;
- cache length and derived position.

This must not be enabled in the normal vLLM service. The current vLLM loader
deletes `self.llm.llm.model.model.layers` after exporting the model, so a debug
run that needs Python module hooks must either use the PyTorch backend or keep a
separate unmodified HF model instance. A free-running PyTorch reproduction is
not equivalent to the vLLM sample: vLLM currently uses top-k sampling while the
PyTorch path uses RAS sampling. Replay validity must be checked through chosen
token logprob/rank and numerical-tolerance tests before interpreting features.

### Attention summaries and full maps

The deployed FlashAttention path does not expose attention probabilities at the
CosyVoice Python boundary. Do not switch the production backend globally.
Use a small debug/eager or SDPA run for selected samples, or implement an
explicit vLLM debug attention backend. Stage 1 should collect summaries only:

- attention entropy and maximum weight;
- mass assigned to text, prompt-speech, recent-speech and old-speech spans;
- per-layer/per-head concentration and sink ratio.

Raw attention mass is not comparable across spans of different length. Any
future attention report must also record the number of available keys and use
the following length-normalized quantities:

```text
text_attention_density = text_attention_mass / text_token_count
speech_attention_density = speech_attention_mass / speech_token_count
text_grounding_ratio = text_attention_mass /
                      (text_token_count / total_context_token_count)
normalized_attention_entropy = H / log(number_of_available_keys)
```

Both `H` and normalized entropy must be reported. This avoids manufacturing a
grounding or entropy trend from sequence length alone.

Only matched GOOD/BAD samples should receive full `layer x head x query x key`
maps. Q/K before and after RoPE can be captured at the Qwen2 attention module in
the debug path; it is not available from the fused production kernel without a
backend change.

## 7. Error-labeling implications

The model's generation unit is a speech token, not a character. A useful first
dataset row should therefore retain both text and speech-token timelines:

```json
{
  "text_token_count":  ...,
  "prompt_speech_token_count": ...,
  "speech_token_count": ...,
  "eos_or_stop_step": ...,
  "audio_duration_seconds": ...,
  "asr_text": ...,
  "cer": ...,
  "wer": ...,
  "repetition_ratio": ...
}
```

At 25 emitted speech tokens/second, the post-filter token position can be mapped
approximately to the Flow/audio timeline. Raw LLM token positions cannot use
this mapping until silent-token drops are applied. The alignment remains
approximate because Flow is a learned token-to-mel model and HiFT adds waveform
context. ASR character/word alignment should refine the error-onset estimate.

The existing API quality gate only rejects degenerate waveform statistics; it
does not measure text fidelity. A normal-sounding but semantically wrong audio
sample can therefore pass the API and must be found by ASR CER/WER evaluation.
The research generator must always submit an explicit seed or save every
pre-gate candidate; otherwise automatic quality retries hide BAD samples and
introduce selection bias.

## 8. First experiment recommendation

Before collecting attention maps, run a paired generation sweep:

1. Build a text set spanning short/long, numbers, punctuation, mixed Chinese and
   English, repeated phrases, and long lists.
2. Generate each text with many explicit seeds using the same voice, model,
   backend, and sampling parameters. Phase 2 treats this as screening because
   the public API does not expose the intermediate token trajectory. Phase 2.5
   later confirmed that the request seed changes vLLM sampling while the
   CosyVoice3 production Flow path reuses a fixed noise buffer.
3. Save raw input, normalized chunk text, chunk boundaries, WAV, exact request
   JSON, effective seed, raw/filtered token counts, dropped tokens, stop reason,
   duration, and API timing headers.
4. Transcribe with one fixed high-quality ASR model and compute CER/WER plus
   insertion/deletion/substitution and repetition metrics.
5. Select same-text GOOD/BAD pairs without cherry-picking and only then run
   exact-token teacher-forced PyTorch/eager replay for those pairs.

The first hypothesis to test is text-grounding collapse:

```text
before error onset:
text-span attention decreases
generated-speech attention increases
entropy/margin or hidden-state change becomes abnormal
```

This is a hypothesis, not an observed result. If the paired data does not show
it with a meaningful effect size or predictive AUC, the report must state
`NO STRONG PATTERN FOUND` and move to the next feature.

## 9. Required controls before causal interpretation

### Separate LLM and acoustic stages

The earlier architecture hypothesis incorrectly generalized from
`ConditionalCFM.forward()`, which samples with `torch.randn_like(mu)`. The
deployed CosyVoice3 YAML uses `CausalConditionalCFM` instead. Its constructor
calls `set_all_random_seed(0)`, creates one `rand_noise` tensor, and inference
only slices that stored tensor. Therefore the production API request seed does
not alter CFM noise in this model.

Phase 2.5 nevertheless separates the stages explicitly. Seed zero reproduces
the production CFM tensor; non-zero `flow_seed` values replace it only inside a
research context manager and restore it after decoding. This counterfactual
tests whether the same token trajectory is content-stable under acoustic noise
changes:

```text
fixed flow_seed + varied llm_seed
fixed speech-token trajectory + varied flow_seed
```

The completed H100 controls are reported in `02_phase2_5_controls.md`.

### Preserve all sequence views

For each chunk save:

```text
raw input text
normalized chunk text
prompt/instruction/reference/target span boundaries
raw vLLM token trajectory
post silent-filter token trajectory
Flow input token trajectory
stop token and finish reason
audio range in the concatenated request WAV
```

This prevents frontend splitting, silent-token suppression and waveform
concatenation from being mistaken for an LLM error.

### Verify instrumentation neutrality

For identical request, prompt, seeds and backend, compare instrumentation off
and on. Require identical token IDs and stop reason, and byte-identical WAV when
the backend is deterministic. Record latency and memory overhead separately.
For eager teacher-forced replay, require the chosen token rank and logprob to
agree within a declared tolerance before using hidden states or attention maps.

### Separate error sources

An ASR mismatch is an end-to-end error, not automatically an LLM error. Use
raw-token repetition/EOS evidence plus fixed-token acoustic re-decodes to
classify likely LLM-generation errors, acoustic-decoder errors and ASR-only
errors. Borderline cases should remain `BORDERLINE`, not be forced into GOOD or
BAD.

## 10. Phase-1 conclusion

- The generation model is a 24-layer Qwen2 decoder with GQA (14 Q heads, 2 KV
  heads), standard RoPE (`theta=1e6`), causal attention and vLLM paged KV cache.
- CosyVoice feeds text, task markers and optional prompt speech embeddings as a
  single causal sequence; there are no explicit text/speech attention masks.
- The production H100 path uses fused FlashAttention, so attention probabilities
  require a debug backend or an eager/SDPA reproduction.
- The safest first instrumentation point is the LM token-generation wrapper and
  silent-token filter for raw/filtered token, logprob and stop summaries,
  followed by exact-token teacher-forced replay for hidden states and attention.
- Phase 2 ASR screening cannot by itself attribute an error to the LLM because
  it does not retain the intermediate speech-token trajectory. Phase 2.5 now
  provides that boundary and verifies the production fixed-noise behavior.
- No RoPE intervention should be attempted before the paired dataset shows a
  position- or RoPE-related precursor.
