# Phase 2 scale result: 24 texts x 32 LLM seeds

## Decision

The H100 collection completed, but it does **not** satisfy the Phase-3 entry
gate. After correcting Simplified/Traditional Chinese ASR orthography, the run
contains only two automatic `BAD` observations and one deterministic
same-text GOOD/BAD pair. Phase 3 attention or RoPE work remains blocked. The
predeclared uniform extension to 64 seeds is therefore required.

## Fixed experiment

- Model: `Fun-CosyVoice3-0.5B-2512`
- Backend: vLLM speech-token generation plus fixed-token Flow/HiFT decode
- Hardware: one NVIDIA H100
- Text groups: 24
- LLM seeds per group: 32 (`20260901` through `20260932`)
- Flow seed: `0`, the production `CausalConditionalCFM` noise buffer
- Voice: server-owned `default` zero-shot voice
- Text frontend reported by the runtime: `none`
- Repeats: none
- Total planned and completed observations: 768

The token and acoustic stages were run separately. Every generated trajectory
was decoded exactly once. Quality-gate rejected WAVs were retained rather than
retried or deleted.

## Collection integrity

| Metric | Result |
|---|---:|
| Token trajectories | 768 / 768 |
| Unique trajectory SHA-256 values | 768 |
| Decoded WAV files | 768 / 768 |
| Fixed-ASR transcripts | 768 / 768 |
| Total audio duration | 11,624.36 s (3:13:44.36) |
| Raw speech tokens | 290,614 |
| Filtered speech tokens | 290,609 |
| Dropped silent tokens | 5 |
| Max-token terminations | 1 |
| Reserved-stop-token terminations | 1,055 frontend chunks |

The only max-token case was
`very_long_mixed__llm_20260922__repeat_0`: its single affected chunk reached
680 raw/filtered speech tokens and finished with `max_tokens`. No trajectory
hash collision or missing artifact was observed.

## Acoustic quality result

| Quality status | Count | Share |
|---|---:|---:|
| PASS | 720 | 93.75% |
| REJECTED | 48 | 6.25% |

All 48 rejections were `too_silent`. They were concentrated in short inputs:

| Text group | Rejected |
|---|---:|
| `short_basic` | 22 |
| `short_order` | 7 |
| `mixed_abbreviations` | 5 |
| `short_negation` | 4 |
| `tongue_twister` | 3 |
| `medium_natural` | 2 |
| `units_measurements` | 2 |
| `parallel_contrast` | 1 |
| `punctuation_dense` | 1 |
| `nested_punctuation` | 1 |

A quality rejection is not treated as a content error. The WAV, token
trajectory, Flow-noise hash and quality reason remain available for source
classification.

## ASR orthography correction

The first analysis incorrectly reported 18 `BAD` rows. Manual inspection found
that 16 were writing-system differences produced by Qwen3-ASR, for example:

```text
reference: 你好，这是短句测试。
ASR:       你好，這是短句測試。
```

The raw transcript is phonetically equivalent and cannot be evidence of a
CosyVoice generation error. A research-only postprocessor now uses
`zhconv:zh-cn` to add separate scoring fields while preserving raw ASR text.
It changed 17 of 768 transcripts and no references. The analyzer uses the
explicit normalized fields only when present.

| Label | Before orthography fold | After orthography fold |
|---|---:|---:|
| GOOD | 453 | 469 |
| BORDERLINE | 297 | 297 |
| BAD | 18 | 2 |

This correction is important: without it, the apparent dataset would contain
mostly ASR orthography false positives.

## Remaining BAD observations

1. `short_order`, LLM seed `20260910`: reference uses “读”, while ASR reports
   “涂” in all three clauses (`CER=0.2308`). This is the only text group with
   an automatic GOOD/BAD pair, but the audio still requires listening review
   to separate a CosyVoice pronunciation error from an ASR substitution.
2. `tongue_twister`, LLM seed `20260914`: the transcript collapses the opening
   “四是四，十是十” structure (`CER=0.0571`) and triggers
   `phrase_repetition`. This group is tagged `pronunciation_sensitive`, so it
   is not admitted to the high-confidence pair set automatically.

The corrected deterministic pair result is therefore:

```text
matched pairs:       1
independent groups:  1 (short_order)
required:           16 pairs from at least 4 groups
Phase 3 ready:      false
```

## Next action

The high-confidence BAD count is below the predeclared threshold of 20. The
next collection uniformly adds seeds `20260933` through `20260964` to **all 24
texts**, yielding a 24 x 64 combined dataset. No text is selected for extra
sampling based on its observed failure rate. The combined analysis must retain
the same voice, Flow seed, model/backend, label thresholds and ASR
orthography-normalization path.

If the 64-seed result still lacks at least 16 matched pairs across four text
groups, Phase 3 remains blocked. The next change should then improve challenge
text coverage or add human/pronunciation review; it should not weaken the gate.
