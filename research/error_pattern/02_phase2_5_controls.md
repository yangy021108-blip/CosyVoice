# Phase 2.5 H100 causal controls

Date: 2026-08-11

Hardware: one NVIDIA H100 80GB HBM3 (physical GPU 3)

Model: `Fun-CosyVoice3-0.5B`, vLLM 0.11.0, FlashAttention, BF16/FP16 runtime

Final run: `research/error_pattern/data/phase2_5_controls_v2_20260811`

## Decision

All three mechanism controls passed. The offline runner can now separate the
vLLM speech-token trajectory from the Flow/HiFT decode without modifying the
production serving path. This is sufficient to scale Phase 2 collection, but
it is not yet sufficient to start attention or RoPE experiments because the
three control texts produced no BAD content sample.

## Corrections made before the final run

The initial proposal assumed that the API request seed also changed Flow noise
because the generic `ConditionalCFM` samples `torch.randn_like(mu)`. The active
CosyVoice3 YAML uses `CausalConditionalCFM`: its constructor seeds to zero,
creates a fixed `rand_noise` tensor, and inference slices that tensor. The
production API therefore has fixed CFM noise for this model.

The research runner now makes this behavior explicit:

- `flow_seed=0` regenerates a byte-identical copy of the production CFM buffer;
- seeds 1–3 replace the buffer inside a research-only context manager;
- the original buffer and all Python, NumPy and Torch RNG state are restored
  after decoding;
- each row records the noise hash, mode, and whether it matches production.

An earlier entity control used homophonic Chinese names, which cannot be
distinguished from audio. The final control uses names with distinct
pronunciations. A remaining `赵琳`/`赵林` ASR spelling difference is treated as
an ASR orthography ambiguity, not as evidence of a TTS content error.

## Experimental design

Three predeclared texts cover distinct entities, repeated similar clauses, and
a long 15-item grounding task. The runtime text frontend reported `none`, so
wetext/ttsfrd normalization was not active.

| Control | Fixed | Varied | Observations |
| --- | --- | --- | ---: |
| A | CFM noise seed 0 (production) | four LLM seeds | 13 WAVs, including the repeat |
| B | one exact token trajectory | CFM noise seeds 0, 1, 2, 3 | 4 WAVs |
| C | text, LLM seed 20260821, CFM seed 0 | exact repeat | 2 observations inside A |

The token sweep saved normalized chunks, prompt/target/prompt-speech spans,
raw tokens, silent-filtered tokens, dropped indexes, stop/finish reason,
sampling bounds, chosen-token logprob/rank and returned top-5 logprobs. Entropy
derived from these truncated logprobs is labeled approximate.

## Results

### Token capture

```text
trajectory rows:                 13
unique trajectories:            12  (one deliberate exact repeat)
frontend chunks:                9 one-chunk rows, 4 two-chunk rows
raw speech-token steps:       6,671
post-filter token steps:      6,671
dropped silent tokens:            0
reserved-stop terminations:      17 / 17 chunks
max-token terminations:           0
chosen-token logprobs present: 6,671 / 6,671
```

All four seeds generated different trajectories for each independent text.
Token lengths varied even when ASR content stayed correct:

| Text | Raw/post-filter token counts by LLM seed |
| --- | --- |
| distinct entities | 261, 261, 252, 225 |
| repeated clauses | 367, 317, 342, 335 |
| long grounding | 962, 1030, 977, 1081 |

### WAV and ASR outcome

```text
decoded WAVs:                 17 / 17
quality-gate PASS:            17
quality-gate REJECTED:         0
total decoded audio:         308.60 s
content labels:               17 GOOD
```

The named-entity control has CER 0.02381 for every run because Qwen3-ASR writes
the homophonic name character `林` instead of reference `琳`. The spoken name
is unchanged, so this does not establish an error. Repetition-control CER spans
0–0.01961, and long-grounding CER spans 0–0.01036.

### Experiment A: fixed production CFM noise, varied LLM seed

Every text produced four distinct token trajectories. End-to-end CER remained
low for all trajectories. This confirms that the LLM seed boundary is active,
but these three texts do not provide a GOOD/BAD contrast.

### Experiment B: fixed token trajectory, varied CFM noise

The four explicit CFM noise tensors produced four different WAV hashes, while
the ASR transcript and CER stayed identical (`0.02381`). For this control,
acoustic variation did not alter recognized content. This is evidence of local
robustness for one token trajectory, not proof that Flow/HiFT can never cause a
content error.

### Experiment C: exact repeatability

The two repeated observations had identical:

- raw and filtered token trajectory hash;
- stop signature;
- PCM hash;
- WAV byte hash;
- ASR transcript and CER.

The instrumentation is neutral and deterministic under the tested fixed
conditions.

## Labeling and pairing rules now enforced

- `generation_status`: `SUCCESS` or `SERVICE_FAIL`;
- `content_label`: `GOOD`, `BORDERLINE`, `BAD`, or `UNKNOWN`;
- service failures have `content_label=UNKNOWN`, never automatic BAD;
- dates, numbers, abbreviations and mixed-language rows cannot be auto-labeled
  GOOD without pronunciation-aware/manual review;
- GOOD/BAD pairs are deterministically one-to-one within `sample_id`, never a
  Cartesian product;
- train/validation/test splitting must group by `sample_id`.

## Next gate

Proceed to the predeclared 24-text × 32-LLM-seed collection using production
CFM noise seed zero. Preserve every service failure and every pre-gate audio
candidate. Use the counterfactual CFM seeds only on selected fixed-token
trajectories to localize whether a content error is token-path or acoustic.

Do not start attention or RoPE modification yet. Phase 3 requires enough
pronunciation-reviewed, same-text GOOD/BAD WAV pairs. If attention is later
measured, report span density, relative-to-uniform grounding, raw entropy `H`,
and normalized entropy `H/log(number_of_available_keys)`; raw attention mass
alone is not comparable across sequence lengths.

## Reproduction notes

The successful H100 container required the CUDA 12.8 compiler tools used by
the mounted PyTorch/vLLM environment. With the NGC 26.04 image, explicitly set:

```text
CUDA_HOME=/usr/local/cuda-12.8
TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas
```

Without `TRITON_PTXAS_PATH`, Triton selected the image's CUDA 13.2 `ptxas` and
rejected it. The run manifests, JSONL records, WAVs and startup-failure logs are
kept in ignored timestamped data directories on H100. No API key is used by
the offline Phase 2.5 runner.
