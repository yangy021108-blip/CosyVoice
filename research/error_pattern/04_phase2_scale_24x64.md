# Phase 2 scale result: 24 texts x 64 LLM seeds

## Decision

The uniform 64-seed extension completed successfully, but Phase 3 remains
blocked. The combined dataset has 1,536 unique token trajectories and only two
automatic `BAD` observations. Both were already present in seeds 1–32; seeds
33–64 added no new `BAD` row. Only one text group has an eligible one-to-one
GOOD/BAD pair, versus the predeclared requirement of 16 pairs from at least
four independent groups.

The next step is not more seeds for these same texts and not attention/RoPE
instrumentation. It is human review of the two candidates followed by a new,
predeclared challenge-text set with broader content-error coverage.

## Combined experiment

The two batches differ only in their LLM seed ranges:

| Batch | LLM seeds | Observations | Audio duration |
|---|---|---:|---:|
| Initial | `20260901..20260932` | 768 | 11,624.36 s |
| Uniform extension | `20260933..20260964` | 768 | 11,546.48 s |
| Combined | `20260901..20260964` | 1,536 | 23,170.84 s |

All other controlled variables remained fixed:

- 24 predeclared text groups;
- `Fun-CosyVoice3-0.5B-2512` and the same vLLM backend;
- server-owned `default` zero-shot voice;
- production CFM noise buffer at `flow_seed=0`;
- `top_k=25` and unchanged token-length bounds;
- runtime text frontend `none`;
- fixed Qwen3-ASR model and ASR options;
- `zhconv:zh-cn` scoring fields with raw transcript preservation.

## Integrity and aggregate result

| Metric | Result |
|---|---:|
| Planned/completed observations | 1,536 / 1,536 |
| Unique generation IDs | 1,536 |
| Unique trajectory hashes | 1,536 |
| WAV files | 1,536 |
| Non-empty ASR transcripts | 1,536 |
| Total audio duration | 23,170.84 s (6:26:10.84) |
| Raw speech tokens | 579,284 |
| Filtered speech tokens | 579,271 |
| Dropped silent tokens | 13 |
| Reserved-stop-token chunk endings | 2,111 |
| Max-token endings | 1 |

The sole max-token case remains `very_long_mixed` at LLM seed `20260922`.
The extension introduced no additional max-token termination.

## Acoustic quality

| Quality status | Count | Share |
|---|---:|---:|
| PASS | 1,451 | 94.47% |
| REJECTED | 85 | 5.53% |

Every rejection was `too_silent`; no clipping or missing-audio failure was
observed. All rejected WAVs and their trajectories remain in the research
artifacts. Short inputs account for most rejections: `short_basic` alone has
41/64 rejected candidates. This quality signal is kept separate from the
content label.

## Orthography-aware labels

The normalizer changed 17 transcripts in the first batch and 29 in the
extension, for 46/1,536 raw ASR transcripts. No reference text changed. The
raw text remains stored beside the Simplified-Chinese scoring form.

| Content label | Seeds 1–32 | Seeds 33–64 | Combined |
|---|---:|---:|---:|
| GOOD | 469 | 468 | 937 |
| BORDERLINE | 297 | 300 | 597 |
| BAD | 2 | 0 | 2 |

The combined mean per-observation CER after orthography folding is `0.05752`.
The large `BORDERLINE` count is intentional: numbers, dates, abbreviations,
mixed-language and pronunciation-sensitive rows are not auto-promoted to
high-confidence GOOD/BAD data.

## The two remaining BAD rows

### `short_order`, seed `20260910`

```text
reference: 先读蓝色，再读绿色，最后读红色。
ASR:       先涂蓝色，再涂绿色，最后涂红色。
CER:       0.2308
quality:   PASS
```

This is the sole eligible deterministic pair group. Because “读/涂” may still
be a close-pronunciation ASR substitution, listening review is required before
calling it a confirmed LLM speech-token error.

### `tongue_twister`, seed `20260914`

The ASR transcript collapses the opening “四是四，十是十” structure and the
automatic rule reports `phrase_repetition`. Its group is tagged
`pronunciation_sensitive`, so it is not paired automatically and also requires
listening review.

No additional candidate appeared in the second 32-seed batch. This makes a
seed-frequency explanation for either row weak and shows that the current
clean-reading challenge set is not yielding a diverse error dataset.

## Phase-3 gate

```text
matched one-to-one pairs:    1
paired sample_id groups:     1 (short_order)
required pairs:             16
required groups:             4
Phase 3 ready:              false
```

Full-attention dumping, hidden-state mining and RoPE intervention remain out of
scope. Any such analysis now would be based on one questionable text group and
would invite pseudo-replication and ASR confounding.

## Next experiment

Do not extend these 24 texts to more seeds. First listen to the two candidates
and a stratified sample of `too_silent` WAVs. Then freeze a new challenge bank
before generation, with multiple independent texts in each of these categories:

1. polarity/negation retention;
2. ordered actions with three or more distinct verbs;
3. entity-role binding with acoustically distinct names;
4. correction chains where the final value overrides an earlier value;
5. repeated syntax with unique semantic anchors per clause;
6. long-range closing anchors that repeat an early requirement;
7. insertion/deletion-sensitive parallel clauses;
8. self-reinforcement prompts with controlled, non-numeric markers.

Avoid dates, digits, abbreviations and tongue twisters in the high-confidence
subset. Use at least three text groups per category, 32 uniformly assigned LLM
seeds, the same fixed Flow seed, and the existing orthography-aware ASR path.
Predeclare the gate and do not add seeds only to texts that fail.

If that clean challenge set still produces no diverse confirmed BAD rows, the
research conclusion should be that normal production sampling is too robust
for this mining protocol. A later experiment may then vary one sampling axis
under a separate, explicitly out-of-distribution study; it must not be mixed
with the production-distribution dataset.
