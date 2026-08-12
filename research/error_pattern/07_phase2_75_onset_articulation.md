# Phase 2.75: local articulation and onset-degradation pilot

## Scope and current decision

Phase 2.75 is separate from the content-error study. It does not change any
existing `GOOD`/`BORDERLINE`/`BAD` content label, the production-distribution
error-rate report, or the Phase-3 content gate. Its independent labels are:

```text
CLEAR
MILD_DEGRADATION
SEVERE_DEGRADATION
UNKNOWN
```

The 5 + 5 trajectory, four-Flow-seed pilot has completed technically. No blind
human review has been recorded yet, so articulation counts, positional effect,
causal attribution and the Phase-2.75 mechanism gate remain unknown. ASR and
acoustic measurements below are screening features, not articulation labels.

## Implementation plan and status

| Step | Implementation | Status |
|---|---|---|
| Preserve content study | Independent articulation fields and outputs | complete |
| Freeze review subset | 22 BORDERLINE + 16 `too_silent` + nominations + 50 fixed random controls | complete |
| Blind review | Randomized, resumable CLI with full/0.5/1/2-second playback | implemented; labels pending |
| Local transcript metrics | Prefix CER and edit counts at 3/5/10 characters | complete |
| Local acoustic features | Four time windows with RMS, silence, voice/F0 and energy-continuity proxies | complete |
| Alignment audit | Probe existing Whisper timestamps and installed aligners | complete; character alignment unavailable |
| Pilot selection | Five suspected candidates and five clear-control candidates | frozen |
| Fixed-token re-decode | Four Flow seeds per trajectory, with no LLM rerun | complete, 40/40 |
| Pilot ASR/features | Fixed Qwen3-ASR plus local metrics | complete, 40/40 |
| Human attribution | Blindly label all 40 re-decodes | pending |
| Scale decision | Apply Gate A/B/C only after pilot review | pending |

## Phase-2.5 reuse audit

`phase2_5_core.decode_fixed_trajectory()` is reusable. It reconstructs the
same frontend inputs, checks that target token IDs did not change, consumes the
saved filtered speech-token trajectory and changes only the CFM noise buffer
inside `isolated_flow_seed()`.

The old `decode_fixed_tokens.py` CLI was not sufficient because it permits
multiple Flow seeds for only one selected trajectory and also schedules the
ordinary seed-zero decode for every input trajectory. The new
`decode_onset_pilot.py` is a narrow scheduling layer for exactly ten frozen
trajectories x Flow seeds 0, 1, 2 and 3. It never invokes speech-token
generation.

This intervention changes Flow noise and then executes Flow plus HiFT. It can
separate token-path persistence from acoustic-seed sensitivity, but it cannot
yet distinguish a Flow mel defect from a HiFT waveform defect. That later
split would require retaining the Flow mel and repeatedly decoding the same
mel through HiFT; no such attribution is claimed here.

## Frozen review subset

The generated manifest is:

```text
research/error_pattern/data/onset_review_manifest.jsonl
```

It contains 88 unique observations:

- all 22 content `BORDERLINE` rows from challenge-bank v2;
- all 16 acoustic `too_silent` rows from challenge-bank v2;
- two manually nominated earlier candidates, `short_order/20260910` and
  `tongue_twister/20260914`;
- 50 `GOOD` + `PASS` controls selected with seed `20260812`.

The BORDERLINE and `too_silent` sets overlap in two rows, so their union has 36
rows. Adding two nominations and 50 controls gives 88, not 90. The two user
nominations intentionally cover both earlier candidates because the user's
description did not disambiguate which one had the subtle onset merge. A
nomination is not a human articulation label.

Every row carries the requested identity, audio, text, content/quality state,
raw and filtered speech-token arrays, trajectory hash and duration. It also
records prompt speech-token count and the target's first 10, 25 and 50 token
IDs. The blind-review program does not display content label, seed, source
group or filename.

Use `record_onset_nomination.py` for an additional exact sample and seed. It
refuses duplicate records:

```bash
python research/error_pattern/record_onset_nomination.py \
  --output research/error_pattern/data/onset_manual_nominations.jsonl \
  --sample-id SAMPLE_ID \
  --llm-seed SEED \
  --notes "what was heard"
```

## Alignment capability audit

The fixed Qwen3-ASR pipeline returns only whole-utterance text. Whisper
large-v3-turbo was probed on the nominated tongue-twister WAV with both
segment and `word` timestamps. Both modes returned multi-character phrase
segments, for example a single 0.00-3.26 second span for the opening phrase;
neither returned character boundaries or confidence.

The fixed ASR environment contains `transformers`, `torchaudio`, `librosa` and
`scipy`, but contains neither WhisperX, a CTC forced aligner nor FunASR. No
compatible local aligner model was found. Therefore:

- prefix CER@3/5/10 is available on the transcript axis;
- RMS, silence, voiced/F0 and energy-continuity features are available in
  0-0.5, 0.5-1, 1-2 and 2-4 second windows;
- `first_0.5s_error`, `first_1.0s_error` and `first_2.0s_error` remain null;
- Whisper phrase timestamps can be stored as an explicitly approximate
  segment alignment, with `alignment_confidence=null`;
- no phoneme deletion or merge metric is emitted.

The audio features are descriptive proxies. They must not be interpreted as
pronunciation accuracy.

## Frozen pilot

The selection is stored in:

```text
research/error_pattern/data/onset_pilot_selection.jsonl
```

| Candidate group | Sample | LLM seed | Selection reason |
|---|---|---:|---|
| suspected | `tongue_twister` | 20260914 | user-review nomination |
| suspected | `short_order` | 20260910 | user-review nomination |
| suspected | `v2_negation_02` | 20261012 | prefix mismatch + `too_silent` |
| suspected | `v2_binding_04` | 20261030 | prefix mismatch |
| suspected | `v2_negation_01` | 20261025 | prefix mismatch |
| clear control | `v2_negation_01` | 20261023 | fixed random GOOD/PASS, prefix CER zero |
| clear control | `v2_self_04` | 20261025 | fixed random GOOD/PASS, prefix CER zero |
| clear control | `v2_binding_01` | 20261032 | fixed random GOOD/PASS, prefix CER zero |
| clear control | `v2_repeat_01` | 20261008 | fixed random GOOD/PASS, prefix CER zero |
| clear control | `v2_parallel_01` | 20261009 | fixed random GOOD/PASS, prefix CER zero |

The selected `v2_negation_01` trajectories form a useful same-text,
different-LLM-seed candidate/control pair. Selection is explicitly marked
`selection_is_human_confirmation=false`.

The run root is:

```text
research/error_pattern/data/phase2_75_onset_pilot_20260812
```

Technical integrity checks passed:

- 10 selected trajectories and 40/40 re-decodes;
- exactly four Flow seeds per trajectory: 0, 1, 2 and 3;
- 10 unique trajectory hashes and 40 unique WAV hashes;
- Flow seed zero matched the original production WAV byte-for-byte for all
  ten trajectories;
- Flow seeds 1-3 used counterfactual CFM noise without rerunning the LLM;
- 34 audio-quality `PASS` and six `REJECTED` re-decodes;
- fixed Qwen3-ASR produced 40/40 non-empty transcripts.

Manifest and selection SHA-256 values are frozen as:

```text
onset_review_manifest.jsonl  8d16e9288d2bf97b758f7906a299a1ac6b9905b229fe24c9a30869a311e84086
onset_pilot_selection.jsonl  57efc2307a04e2086fee78d7d4dd4c0cd0978cec60615a00d6992a4081ae7514
```

## Screening results before human review

Qwen3-ASR prefix results across Flow seeds 0/1/2/3 were:

| Trajectory | Prefix behavior | Quality behavior |
|---|---|---|
| `tongue_twister/20260914` | prefix mismatch at all four seeds | 4 PASS |
| `short_order/20260910` | mismatch at seeds 0/1/2, zero at seed 3 | 4 PASS |
| `v2_negation_02/20261012` | prefix mismatch at all four seeds | 4 REJECTED |
| `v2_binding_04/20261030` | mismatch at seeds 0/3, zero at 1/2 | 4 PASS |
| `v2_negation_01/20261025` | mismatch at seeds 0/2, zero at 1/3 | 4 PASS |
| five clear controls | prefix CER@3/5/10 zero at all 20 re-decodes | 18 PASS / 2 REJECTED |

This demonstrates that the pilot can observe both seed-stable and
seed-sensitive ASR-prefix patterns. It does not prove that either pattern is
audible articulation degradation. In particular, ASR agreement on a
pronunciation-sensitive tongue twister is not a substitute for human
listening, while the two rejected clear-control WAVs show that `too_silent`
and a prefix mismatch are not interchangeable.

## Blind review and resume

Download the pilot WAVs and `pilot_blind_manifest.jsonl` to a machine with
audio output, keeping the WAV filenames unchanged. The blind manifest contains
no seed, content label, source group or revealing filename field. From a
CosyVoice checkout run:

```bash
python research/error_pattern/review_local_articulation.py \
  --manifest pilot_blind_manifest.jsonl \
  --output pilot_human_reviews.jsonl \
  --reviewer REVIEWER_NAME \
  --audio-root /path/to/downloaded/audio
```

The command uses a stable randomized order, displays only a blind ID,
duration and reference text, and supports whole-WAV or first 0.5/1/2-second
playback. It appends each completed review and skips it on resume; it never
overwrites an existing blind ID. `u` is stored canonically as
`articulation_label=UNKNOWN` and `review_decision=UNCERTAIN`.

After uploading the review JSONL, rerun local metrics with `--reviews`, then
run `summarize_onset_pilot.py`. Only that output may compare:

```text
P(degradation | suspected trajectory)
P(degradation | clear-control trajectory)
```

or create the PASS/`too_silent` x CLEAR/DEGRADED contingency table.

## Required questions: current answers

1. Human-confirmed CLEAR/MILD/SEVERE: **0/0/0; review pending**.
2. Onset concentration: **unknown**; no human time spans yet.
3. Time/token-position distribution: **unknown**. The 25-token/s mapping is
   retained only as an explicitly approximate bucket conversion.
4. Association with `too_silent`: **unknown**; the human contingency table is
   empty. Quality-gate counts alone cannot answer it.
5. Persistence after fixed-token re-decode: **ASR screening is mixed**, but
   human articulation persistence is unknown.
6. Degradation of clear trajectories: **unknown by listening**; all clear
   prefix transcripts stayed correct, although two WAVs became `too_silent`.
7. Supported component: **unknown**. Current evidence cannot choose LLM token
   trajectory, Flow, HiFT or mixed.
8. LLM mechanism gate: **not satisfied**. This ten-trajectory pilot cannot by
   itself meet the 16 degraded + 16 clear-control threshold.
9. Next route: **finish blind pilot review first**. Seed-sensitive audible
   degradation would select acoustic-decoder analysis; persistent degradation
   with a lower clear-control rate would justify a predeclared scale-up before
   token/logprob work; no reproducible audible effect would select the
   separately defined stress-sampling study.

No hidden-state, attention or RoPE instrumentation is authorized by this
pilot.
