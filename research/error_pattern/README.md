# CosyVoice generation-error research

This directory contains the reproducible research workflow for finding
speech-content generation errors in CosyVoice3-0.5B. The phases are deliberately
separated so that an end-to-end ASR mismatch is not prematurely attributed to
the language model.

## Current status

- Phase 1: architecture inspection is recorded in `00_architecture.md`.
- Phase 2: the baseline generator, challenge-text set and ASR evaluator are
  implemented. The first H100 pilot is written to a timestamped run directory.
- Phase 2.5: independent LLM/Flow seed controls and raw/post-filter token
  capture are implemented in research-only offline runners.
- Phase 2 scale: both H100 collections are complete. The combined 24 x 64 run
  contains 1,536 unique trajectories but still only one eligible GOOD/BAD
  pair, so Phase 3 remains blocked. See `03_phase2_scale_24x32.md` and
  `04_phase2_scale_24x64.md`.
- Candidate adjudication: Qwen3-ASR and Whisper disagree on `short_order` and
  agree on the `tongue_twister` opening collapse; both remain human-pending.
  Challenge bank v2 is frozen at 8 categories x 4 texts x 16 seeds. See
  `05_adjudication_and_challenge_v2.md`.
- Phase 3 and later: not started. Hidden-state/attention instrumentation must
  not be enabled until Phase 2.5 controls pass and reproducible GOOD/BAD pairs
  exist.

Human-confirmed Phase-3 readiness additionally requires at least three
independent challenge categories. Automatic ASR candidates are reported
separately and cannot satisfy the confirmed gate without an adjudication
record keyed by `decode_id`.

## Phase 2 design

The pilot covers short through very-long text and deliberately includes dates,
numbers, English abbreviations, mixed Chinese/English, proper nouns, repeated
and similar phrases, a tongue twister, a long-distance dependency, a list,
dense punctuation, and target text overlapping the zero-shot prompt. Every
text is sent with the same four explicit seeds. Expanding every text uniformly,
rather than adding seeds only to a failed text, avoids adaptive cherry-picking.

An explicit seed is mandatory. It prevents the API quality gate from silently
retrying and replacing a rejected candidate. Phase 2 is still an end-to-end
screen because the public API does not expose the token trajectory. Phase 2.5
showed that CosyVoice3's production `CausalConditionalCFM` reuses fixed
seed-zero noise; non-zero Flow seeds in the offline runner deliberately replace
that buffer as a research-only acoustic sensitivity test.

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

`GOOD`, `BORDERLINE`, and `BAD` content labels are derived from predeclared
thresholds, while CER/WER and edit counts remain continuous fields. Service
failures use `generation_status=SERVICE_FAIL` and `content_label=UNKNOWN`; they
are not content errors. A Phase-3 pair is emitted only when one text has at
least one GOOD WAV and one BAD WAV. Pairing is deterministic and one-to-one,
never the Cartesian product of all GOOD and BAD runs. Dataset splitting must
group by `sample_id` to prevent the same text appearing in train and test.

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

## Phase 2.5 controls

These commands are intentionally offline: the public API exposes only the LLM
sampling seed and does not expose the stored CFM noise buffer. The offline
runner must preserve the exact token trajectory and vary the two stages
separately.
Run each command from the repository root in the same vLLM environment used by
the service. Each output directory must be new or empty.

Capture four LLM trajectories per text, plus one exact repeat, without running
Flow/HiFT:

```bash
python research/error_pattern/run_token_sweep.py \
  --config research/error_pattern/configs/phase2_5_controls.json \
  --output-dir research/error_pattern/data/phase2_5_controls/token_sweep
```

Decode all trajectories with one fixed Flow seed, then decode one fixed token
trajectory with four Flow seeds. All WAVs are retained, including candidates
that the API quality gate would reject:

```bash
python research/error_pattern/decode_fixed_tokens.py \
  --config research/error_pattern/configs/phase2_5_controls.json \
  --trajectories research/error_pattern/data/phase2_5_controls/token_sweep/token_trajectories.jsonl \
  --output-dir research/error_pattern/data/phase2_5_controls/decode
```

Run fixed Qwen3-ASR over `decode/audio` using the command above with the Phase
2.5 paths, then evaluate the three controls:

```bash
python research/error_pattern/analyze_phase2_5_controls.py \
  --config research/error_pattern/configs/phase2_5_controls.json \
  --decode-results research/error_pattern/data/phase2_5_controls/decode/decode_results.jsonl \
  --asr research/error_pattern/data/phase2_5_controls/decode/asr_results.jsonl \
  --output research/error_pattern/data/phase2_5_controls/control_samples.jsonl \
  --summary research/error_pattern/data/phase2_5_controls/control_summary.json
```

The controls mean:

- A: fixed `flow_seed`, varied `llm_seed` isolates token-trajectory changes.
- B: fixed token trajectory, production noise at `flow_seed=0`, then
  research-only counterfactual CFM noise at seeds 1–3 measures acoustic
  sensitivity without changing production code.
- C: identical LLM and Flow seeds repeated twice tests instrumentation
  neutrality and determinism.

Top-k logprob entropy fields are explicitly approximate. Exact vocabulary
entropy and attention work remain deferred. Future attention comparisons must
use span density, relative-to-uniform grounding, and both raw `H` and
`H/log(number_of_available_keys)`.

The completed H100 control results and their limits are recorded in
`02_phase2_5_controls.md`.

## Phase 2 scale collection (24 texts x 32 seeds)

The scale configuration freezes 24 text groups and 32 LLM seeds before
generation, for exactly 768 production-noise trajectories. It does not add
extra seeds after observing failures. Both long-running stages support
`--resume`; resume is rejected if the config, text-set or voice hash changed.

Capture speech-token trajectories:

```bash
python research/error_pattern/run_token_sweep.py \
  --config research/error_pattern/configs/phase2_scale_24x32.json \
  --output-dir research/error_pattern/data/phase2_scale_24x32/token_sweep \
  --resume
```

Decode every trajectory with the production CFM noise buffer (`flow_seed=0`).
Quality-rejected candidates are still saved:

```bash
python research/error_pattern/decode_fixed_tokens.py \
  --config research/error_pattern/configs/phase2_scale_24x32.json \
  --trajectories research/error_pattern/data/phase2_scale_24x32/token_sweep/token_trajectories.jsonl \
  --output-dir research/error_pattern/data/phase2_scale_24x32/decode \
  --resume
```

Run the fixed Qwen3-ASR command over `decode/audio`, using the generated
`decode/asr_expected_texts.txt`. Before scoring, add Simplified-Chinese fields
with the `zhconv` already installed in the fixed ASR environment. Raw ASR text
is preserved:

```bash
/SharedData/yangyu/qwen3_asr_518/bin/python \
  research/error_pattern/normalize_asr_orthography.py \
  --input research/error_pattern/data/phase2_scale_24x32/decode/asr_results.jsonl \
  --output research/error_pattern/data/phase2_scale_24x32/decode/asr_results_zhcn.jsonl \
  --summary research/error_pattern/data/phase2_scale_24x32/decode/asr_orthography_summary.json
```

Then build labels and deterministic pairs:

```bash
python research/error_pattern/analyze_phase2_scale.py \
  --config research/error_pattern/configs/phase2_scale_24x32.json \
  --trajectories research/error_pattern/data/phase2_scale_24x32/token_sweep/token_trajectories.jsonl \
  --decode-results research/error_pattern/data/phase2_scale_24x32/decode/decode_results.jsonl \
  --asr research/error_pattern/data/phase2_scale_24x32/decode/asr_results_zhcn.jsonl \
  --output research/error_pattern/data/phase2_scale_24x32/error_dataset.jsonl \
  --summary research/error_pattern/data/phase2_scale_24x32/evaluation_summary.json \
  --pairs research/error_pattern/data/phase2_scale_24x32/good_bad_pairs.jsonl
```

Phase 3 remains gated on at least 16 one-to-one GOOD/BAD pairs across at least
four independent `sample_id` groups after pronunciation-aware/manual review.
The 24 x 32 run did not pass this gate, so every text was uniformly extended
with `phase2_scale_24x32_extension.json` (seeds `20260933..20260964`). For
combined 24 x 64 analysis, pass both run files after each plural input option
and use `phase2_scale_24x64.json`; `analyze_phase2_scale.py` accepts one or more
paths for `--trajectories`, `--decode-results`, and `--asr`. The completed
24 x 64 result also fails the gate. Do not add more seeds to these same texts;
use the challenge-bank and human-review next steps in
`04_phase2_scale_24x64.md`.
