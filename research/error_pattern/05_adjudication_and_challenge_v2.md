# Candidate adjudication and challenge-bank v2 protocol

## Current adjudication state

The 24 x 64 production-distribution run has two automatic `BAD` rows. They are
not confirmed content errors. Each candidate now has results from two
independent ASR model families:

| Candidate | Qwen3-ASR | Whisper large-v3-turbo | State |
|---|---|---|---|
| `short_order`, seed `20260910` | “先涂蓝色……” | “先读蓝色……” | ASRs disagree; human pending |
| `tongue_twister`, seed `20260914` | opening collapses | opening collapses | ASRs agree; human pending |

The machine-readable record is
`data/phase2_24x64_adjudication.jsonl`. Both rows use
`adjudication_label=UNRESOLVED`; neither can enter the confirmed Phase-3 pair
set. If human listening rejects both candidates, the correct 24 x 64 result is
zero confirmed content errors. That is a valid negative result.

## Human listening protocol

For each candidate:

1. Listen from the original WAV with no ASR transcript visible.
2. Listen at normal speed at least twice and write an exact transcript.
3. Compare the human transcript with the reference only after transcription.
4. Record one decision:
   - `REFERENCE_CONFIRMED`: the human clearly hears the reference content;
   - `CONTENT_ERROR_CONFIRMED`: the human clearly hears an insertion,
     deletion, substitution, reordering or repetition not present in the
     reference;
   - `UNCLEAR`: pronunciation or audio quality prevents a reliable decision.
5. Map the decision to the dataset label:
   - `REFERENCE_CONFIRMED` -> `ASR_FALSE_POSITIVE`;
   - `CONTENT_ERROR_CONFIRMED` -> `CONFIRMED_CONTENT_ERROR`;
   - `UNCLEAR` -> `UNRESOLVED`.

The listener must also fill `human_reviewer`, `human_transcript`,
`human_decision` and `notes`. Qwen3-ASR and Whisper are supporting evidence;
their agreement is never a substitute for human listening.

Whisper can be reproduced without system `ffmpeg`; the helper reads WAV data
with `soundfile` and passes the array directly to Transformers:

```bash
CUDA_VISIBLE_DEVICES=3 \
/SharedData/yangyu/qwen3_asr_518/bin/python \
  research/error_pattern/transcribe_whisper_candidates.py \
  --model /SharedData/models/openai-mirror/whisper-large-v3-turbo \
  --audio /path/to/candidate_one.wav /path/to/candidate_two.wav \
  --output /path/to/whisper_results.jsonl
```

## Challenge-bank v2 design

The second bank tests faithful reading only. It does not expect CosyVoice to
interpret an instruction, resolve a correction, answer a question or suppress
earlier words. Every word in each input must be spoken in its written order.

The frozen bank contains 32 independent texts:

| Category | Texts | Intended observable error |
|---|---:|---|
| `negation_retention` | 4 | loss of “不/不能/不要” |
| `action_order` | 4 | clause deletion or action reordering |
| `entity_action_binding` | 4 | action assigned to the wrong entity |
| `parallel_clauses` | 4 | previous semantic anchor copied forward |
| `repetition_with_anchor` | 4 | repeated syntax overwrites changing anchors |
| `local_interference` | 4 | nearby conflicting phrase substituted |
| `long_distance_anchor` | 4 | explicit closing anchor differs from opening |
| `controlled_self_reinforcement` | 4 | later clause follows prior speech pattern |

Dates, Arabic digits, Latin abbreviations and tongue twisters are excluded.
Most inputs are medium or long to reduce the separate very-short-utterance
`too_silent` effect. The data file records `challenge_category`, and the token,
decode and analysis stages preserve it.

## Frozen scale and extension rule

Initial production-distribution collection:

```text
8 categories x 4 texts/category x 16 LLM seeds = 512 observations
```

All 32 texts use seeds `20261001..20261016`, fixed production Flow noise at
`flow_seed=0`, the same voice, model, backend and sampling as the earlier run.

Predeclared extension rule:

```text
if confirmed CONTENT_BAD < 8:
    extend every one of the 32 texts uniformly to seeds 20261017..20261032
else:
    do not run the extension
```

The maximum combined collection is 1,024 observations. Never add seeds only
to a text that happened to fail.

## Phase-3 gate

The analyzer now separates automatic candidate readiness from confirmed
readiness. `phase3_ready` requires human-confirmed BAD rows and all three
conditions:

```text
confirmed one-to-one GOOD/BAD pairs >= 16
independent sample_id groups >= 4
independent challenge categories >= 3
```

Automatic labels are reported as `phase3_candidate_ready` only. Human
adjudication is joined by `decode_id`; missing or unresolved decisions cannot
make `phase3_ready=true`.

If the full 32 x 32 challenge bank still produces only zero to two confirmed
errors, stop adding seeds and texts. A later mechanism-discovery experiment
may vary sampling under a separately named stress distribution. Production
and stress samples must never be combined when reporting error rate.
