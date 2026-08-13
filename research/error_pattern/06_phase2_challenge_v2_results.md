# Phase-2 challenge-bank v2: H100 results

## Decision

The frozen production-distribution challenge-bank run is complete. It did not
produce an automatic `BAD` row or a human-confirmed content error, so Phase 3
must not start. In accordance with the predeclared stopping rule, this study
will not add more seeds or selectively extend individual texts. Attention,
hidden-state and RoPE instrumentation remain out of scope.

This is a negative result for this particular bank and sampling distribution;
it is not evidence that CosyVoice cannot produce content errors.

## Frozen design

- Hardware: one NVIDIA H100.
- Model: `Fun-CosyVoice3-0.5B-2512`, vLLM language-model backend.
- Texts: 32 faithful-reading inputs, four in each of eight categories.
- Sampling: the production API distribution (`top_k=25`), one explicit LLM
  seed per trajectory and production Flow noise (`flow_seed=0`).
- Initial seeds: `20261001..20261016`.
- Uniform extension seeds: `20261017..20261032`.
- Scale: 32 texts x 32 seeds = 1,024 observations.
- Scoring: fixed Qwen3-ASR, Simplified-Chinese orthography normalization and
  the thresholds frozen in `phase2_challenge_v2_32x32.json`.

The extension was triggered because the initial 32 x 16 run contained fewer
than eight confirmed content errors. Every text received the same additional
16 seeds; no failure-targeted seed selection was used.

## Collection integrity

The combined collection contains:

- 1,024/1,024 completed observations;
- 1,024 unique `generation_id` values;
- 1,024 unique speech-token trajectory hashes;
- 32 independent `sample_id` groups and 32 distinct LLM seeds;
- 128 observations in each of the eight challenge categories;
- zero max-length terminations.

The raw trajectories contain 224,146 speech tokens. Three silent tokens were
removed by the existing post-filter, leaving 224,143 decoded tokens. This
difference is recorded explicitly and does not create a content-error pair.

## Results

| Run | Observations | GOOD | BORDERLINE | BAD | Audio quality PASS / REJECTED | Audio duration | Mean row CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initial seeds 01-16 | 512 | 499 | 13 | 0 | 499 / 13 | 4,504.12 s | 0.004415 |
| Extension seeds 17-32 | 512 | 503 | 9 | 0 | 509 / 3 | 4,461.60 s | 0.004014 |
| Combined | 1,024 | 1,002 | 22 | 0 | 1,008 / 16 | 8,965.72 s | 0.004215 |

All 512 extension WAVs received a non-empty ASR transcript. Its corpus-level
character error rate was 76/18,192 = 0.004178. Across both halves the
corpus-level rate was 174/36,384 = 0.004782; the table reports the analyzer's
mean of per-observation CER values, which is a different aggregation.

All 16 acoustic-quality rejections were `too_silent`; two also had a
`BORDERLINE` transcript. A quality rejection is not treated as a speech-content
error. It remains in the dataset as `error_source=ACOUSTIC_QUALITY_GATE`.

Per-category content labels were:

| Category | GOOD | BORDERLINE | BAD |
|---|---:|---:|---:|
| `action_order` | 127 | 1 | 0 |
| `controlled_self_reinforcement` | 127 | 1 | 0 |
| `entity_action_binding` | 122 | 6 | 0 |
| `local_interference` | 121 | 7 | 0 |
| `long_distance_anchor` | 127 | 1 | 0 |
| `negation_retention` | 125 | 3 | 0 |
| `parallel_clauses` | 125 | 3 | 0 |
| `repetition_with_anchor` | 128 | 0 | 0 |

The 22 `BORDERLINE` rows are retained as uncertain ASR mismatches. They do not
cross the frozen automatic `BAD` threshold and therefore are not promoted by
a second ASR or human review. Examples include proper-name homophones and
short local substitutions; these observations should not be described as
confirmed model errors.

## Phase-3 gate

| Requirement | Observed | Required | Result |
|---|---:|---:|---|
| Automatic one-to-one GOOD/BAD pairs | 0 | 16 | fail |
| Confirmed one-to-one GOOD/BAD pairs | 0 | 16 | fail |
| Confirmed failing text groups | 0 | 4 | fail |
| Confirmed failing categories | 0 | 3 | fail |

Both `phase3_candidate_ready` and `phase3_ready` are `false`. No causal claim
about the language model, Flow or vocoder is justified from this collection.

The two automatic `BAD` candidates from the earlier 24 x 64 bank remain a
separate human-adjudication task. Qwen3-ASR and Whisper disagree on
`short_order`; both ASRs flag the pronunciation-sensitive `tongue_twister`.
Neither enters the confirmed pair set until a blinded listener fills the
adjudication record described in `05_adjudication_and_challenge_v2.md`.

## Reproducible artifacts

Server-side run roots are:

```text
research/error_pattern/data/phase2_challenge_v2_32x16_20260811
research/error_pattern/data/phase2_challenge_v2_32x16_extension_20260811
research/error_pattern/data/phase2_challenge_v2_32x32_20260811
```

The combined run root contains `error_dataset.jsonl`,
`evaluation_summary.json` and `good_bad_pairs.jsonl`. The latter is empty by
design because no automatic BAD row exists. Raw WAVs, token trajectories and
ASR outputs remain in the two half-run directories. Large generated artifacts
are intentionally excluded from Git.

## Next experiment boundary

Do not append more production-distribution seeds to this bank. If mechanism
discovery is still required, create a separately named, frozen stress-sampling
experiment that varies one sampling factor at a time. Its error rate and pair
counts must be reported separately from the production distribution; stress
samples cannot be pooled into the Phase-2 production estimate. Freeze the
stress grid, seed list, sample bank and stopping rule before generation, then
apply the same independent-ASR and blinded-human confirmation policy.
