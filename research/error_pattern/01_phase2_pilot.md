# Phase 2 H100 baseline pilot

Date: 2026-08-11

Hardware: one NVIDIA H100 80GB HBM3 (physical GPU 3)

CosyVoice checkout at generation time: `732e30184af72f921fd3f8e73fe1e483e5b40011`

## Purpose and fixed design

This pilot validates the complete Phase-2 loop before scaling to hundreds or
thousands of requests. It uses 12 challenge texts and four predeclared seeds
per text, for 48 requests. Every text receives the same seed set
`20260811..20260814`; seeds were expanded uniformly rather than only for texts
that failed in the first pass.

The model, voice, response format, speed, API quality gate and single-GPU
backend were fixed. Every request passed an explicit seed, so the API did not
replace a rejected candidate with an automatic retry. Phase 2 did not capture
the intermediate speech-token trajectory, so it remains an end-to-end
screening result. Phase 2.5 later established that this CosyVoice3
`CausalConditionalCFM` uses a fixed, seed-zero noise buffer in production; the
request seed changes vLLM sampling, not that stored Flow noise.

## Runtime configuration

Generation used:

```text
model: Fun-CosyVoice3-0.5B
API alias: cosyvoice3-0.5b
voice: default (server-owned zero-shot voice)
backend: vLLM 0.11.0, FlashAttention, bfloat16 model
response: mono 16-bit WAV at 24 kHz
concurrency: 1
GPU: physical GPU 3
```

Both generation service logs reported `no frontend is avaliable`. The wetext
snapshot lookup failed and ttsfrd was unavailable, so wetext/ttsfrd text
normalization was not active. This is part of the experimental condition and
may especially affect numbers, dates and mixed-language text.

ASR used Qwen3-ASR-0.6B, vLLM 0.17.1 and PyTorch 2.10.0+cu128 on the same H100
after the TTS service was stopped. It ran with `--enforce-eager`. The first ASR
attempt with the service's vLLM 0.11.0 environment failed because that version
does not recognize `qwen3_asr`. A second setup attempt exposed a missing
`Python.h`; the successful command supplied the existing Conda Python include
directory through `C_INCLUDE_PATH`/`CPATH`. Both failed setup logs were retained
in the ignored run directory.

## Generation result

```text
requests:                  48
successful WAV responses: 43
quality-gate failures:      5
total returned audio:       759.88 s
mean / median RTF:          0.08104 / 0.07505
mean request wall latency:  1.4200 s
```

All returned files were mono 16-bit 24 kHz WAVs. The maximum clipping ratio was
zero, the maximum locally measured silent-frame ratio was 0.4735, and every
successful response reported quality retry count zero as required by the
explicit-seed design.

The five failures were HTTP 503 `audio_quality_failed` candidates:

| Text | Seed(s) |
| --- | --- |
| `short_basic` | 20260811, 20260813, 20260814 |
| `mixed_abbreviations` | 20260811 |
| `repeated_structure` | 20260814 |

These failures were retained as JSON rows and were not retried. They are valid
evidence about service-level generation stability, but they have no returned
WAV and cannot be used for attention/hidden-state comparison against a GOOD
audio sample.

## ASR evaluation result

All 43 returned WAV files were transcribed. With the predeclared thresholds and
the orthography safeguard described below, the current labels are:

```text
GOOD:             28
BORDERLINE:       14
BAD content:       1  (returned WAV)
UNKNOWN content:   5  (service failures without WAV)
raw mean CER over returned WAVs: 0.08443
matched GOOD/BAD WAV pairs: 0
```

This table uses the corrected schema-v2 separation between
`generation_status` and `content_label`; the initial schema-v1 report had
incorrectly counted all five service failures as BAD content. The one returned
BAD WAV is `short_basic`, seed `20260812`, with raw CER 0.375.
The current four-seed pilot did not produce a same-text GOOD WAV for that text.
`similar_entities` and `tongue_twister` did show same-text GOOD/BORDERLINE seed
variation, which is useful for validating the pairing machinery but is not yet
a strong GOOD/BAD contrast.

Numbers and Latin abbreviations exposed an evaluation confound. Qwen3-ASR can
emit spoken Chinese forms such as `二零二六` for reference `2026`, causing high
orthographic CER even when the audio may be semantically correct. Raw CER is
still stored unchanged. When the mismatch disappears after removing only
number/Latin spelling variants, the sample is labeled `BORDERLINE` with
`orthography_sensitive_terms_require_review`, never automatically `GOOD`.
Pronunciation-aware normalization or human review is required before those
samples are used as errors.

## Phase-2 conclusion and next gate

The generation, failure preservation, fixed-ASR transcription, edit accounting
and labeling pipeline works end to end. The pilot also found three controls
that must remain in place:

1. Record whether wetext/ttsfrd actually initialized; never merge frontend-on
   and frontend-off runs.
2. Separate service failures from BAD audio and require both sides of a Phase-3
   pair to have WAV files.
3. Do not use raw orthographic CER alone for dates, digits, abbreviations or
   mixed-language names.

Phase 2 is started but not complete. The next collection should be a
predeclared, larger seed grid over the same fixed texts plus more short and
entity-heavy prompts. It must first repair or intentionally freeze the text
frontend condition. Phase 3 must not begin until the dataset contains enough
matched GOOD/BAD WAV pairs after pronunciation-aware/manual validation.
