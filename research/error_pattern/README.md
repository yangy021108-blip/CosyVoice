# CosyVoice generation-error research

This directory contains the reproducible research workflow for finding
speech-content generation errors in CosyVoice3-0.5B. The phases are deliberately
separated so that an end-to-end ASR mismatch is not prematurely attributed to
the language model.

## Current status

- Phase 1: architecture inspection is recorded in `00_architecture.md`.
- Phase 2: the baseline generator, challenge-text set and ASR evaluator are
  implemented. The first H100 pilot is written to a timestamped run directory.
- Phase 3 and later: not started. Token/logit/hidden-state instrumentation must
  not be enabled until Phase 2 has produced reproducible GOOD/BAD pairs.

## Phase 2 design

The pilot covers short through very-long text and deliberately includes dates,
numbers, English abbreviations, mixed Chinese/English, proper nouns, repeated
and similar phrases, a tongue twister, a long-distance dependency, a list,
dense punctuation, and target text overlapping the zero-shot prompt. Every
text is sent with the same four explicit seeds. Expanding every text uniformly,
rather than adding seeds only to a failed text, avoids adaptive cherry-picking.

An explicit seed is mandatory. It prevents the API quality gate from silently
retrying and replacing a rejected candidate. The current API seed affects both
vLLM token sampling and Flow noise, so this phase is an error-screening
experiment, not causal evidence about the LLM. `00_architecture.md` describes
the controls required before causal interpretation.

## Reproduce baseline generation

Start the existing CosyVoice vLLM API on one H100 with a fixed model, voice and
API key. Then run:

```bash
cd /SharedData/yangyu/CosyVoice
/opt/conda/envs/cosyvoice/bin/python research/error_pattern/generate_dataset.py \
  --config research/error_pattern/configs/phase2_pilot.json \
  --samples research/error_pattern/data/samples.jsonl \
  --base-url http://127.0.0.1:8021 \
  --api-key-file /SharedData/yangyu/cosyvoice_vllm_api.env \
  --output-dir research/error_pattern/data/phase2_pilot
```

The API-key file may contain either a raw key or a line of the form
`COSYVOICE_API_KEY=...`. The key itself is never written to the manifest.
Generation produces WAV files, `generation_results.jsonl`, a run manifest,
and ASR input/expected-text files. HTTP failures are retained as dataset rows;
they are not deleted or retried by the research script.

## Run fixed ASR and build labels

Run the repository's Qwen3-ASR script in the established offline ASR
environment. The H100 pilot used vLLM 0.17.1 because the CosyVoice service's
vLLM 0.11.0 environment does not recognize `qwen3_asr`. Eager mode is used for
evaluation reproducibility, not ASR performance benchmarking. The include path
is needed by Triton's small runtime extension on this host. The WAV sort order
matches `asr_expected_texts.txt`:

```bash
CUDA_VISIBLE_DEVICES=3 \
C_INCLUDE_PATH=/SharedData/yangyu/cosyvoice_vllm_env/cosyvoice/include/python3.10 \
CPATH=/SharedData/yangyu/cosyvoice_vllm_env/cosyvoice/include/python3.10 \
/SharedData/yangyu/qwen3_asr_518/bin/python \
  tools/cosyvoice_qwen3_asr_semantic_check.py \
  --model /SharedData/models/Qwen/Qwen3-ASR-0___6B \
  --audio research/error_pattern/data/phase2_pilot/audio \
  --expected-text-file research/error_pattern/data/phase2_pilot/asr_expected_texts.txt \
  --language zh \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-tokens 512 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --output-jsonl research/error_pattern/data/phase2_pilot/asr_results.jsonl \
  --summary-json research/error_pattern/data/phase2_pilot/asr_summary.json
```

Then merge generation and ASR records:

```bash
python research/error_pattern/evaluate_outputs.py \
  --config research/error_pattern/configs/phase2_pilot.json \
  --generation research/error_pattern/data/phase2_pilot/generation_results.jsonl \
  --asr research/error_pattern/data/phase2_pilot/asr_results.jsonl \
  --output research/error_pattern/data/phase2_pilot/error_dataset.jsonl \
  --summary research/error_pattern/data/phase2_pilot/evaluation_summary.json \
  --pairs research/error_pattern/data/phase2_pilot/good_bad_pairs.jsonl
```

`GOOD`, `BORDERLINE`, and `BAD` labels are derived from predeclared thresholds,
while CER/WER and edit counts remain continuous fields. Service failures are
kept and labeled `BAD`. A Phase-3 pair is emitted only when one text has at
least one GOOD WAV and one BAD WAV. A quality-gate failure without audio is
retained in the dataset but excluded from the pair file.

Raw CER remains the primary recorded metric. Numeric and Latin spellings are
orthographically ambiguous (`2026` versus `二零二六`, for example). If raw CER
is high but the remaining content agrees, the automatic label is deliberately
limited to `BORDERLINE` with `orthography_sensitive_terms_require_review`; it
is never promoted to `GOOD` without a pronunciation-aware or manual check.

## Reproducibility and limitations

Each run records the request JSON, response headers, latency, audio statistics,
repository commit/dirty state, GPU information, input hashes and prompt/voice
configuration hash. Fields that the current public API cannot expose—raw and
post-filter speech tokens, normalized frontend chunks, stop reason and exact
text-token count—are explicitly stored as `null`, never inferred.

Do not compare runs unless model directory, repository commit, backend,
sampling parameters, voice configuration, prompt audio and ASR model are all
fixed. ASR disagreement is an end-to-end signal and can come from the LLM,
Flow/HiFT, or ASR itself.
