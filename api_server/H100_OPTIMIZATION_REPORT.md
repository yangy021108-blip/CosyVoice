# H100 inference optimization report

## Scope

This report records real-model API measurements for `Fun-CosyVoice3-0.5B` on
an NVIDIA H100 80 GB GPU. The service used the vLLM backend, PyTorch 2.8.0
with CUDA 12.8, vLLM 0.11.0, Transformers 4.57.1, `COSYVOICE_MAX_CONCURRENCY=1`,
and a fixed default seed of `0`. The benchmark sent two warm-up requests and
then 20 non-streaming WAV requests across five fixed Chinese/English prompts.

The results are a single-GPU latency baseline, not a throughput or P95
production SLO. The model ran alone on the selected GPU, but the host itself
was shared.

## Retained optimization

The default `COSYVOICE_FLOW_STEPS` is now **6** rather than 8. The setting is
still configurable from 1 to 100, so a deployment can return to 8 steps or
the upstream 10 steps without changing code.

| Configuration | Mean latency | Median latency | P95 latency | Mean RTF |
| --- | ---: | ---: | ---: | ---: |
| 8 Flow steps (previous default) | 437.3 ms | 442.9 ms | 481.9 ms | 0.0741 |
| 6 Flow steps (new default) | 418.7 ms | 424.8 ms | 461.1 ms | 0.0710 |
| Change | -4.2% | -4.1% | -4.3% | -4.2% |

For these five prompts, the 6-step and 8-step responses had the same WAV
duration per prompt. Whisper large-v3-turbo independently transcribed both
sets with the same normalized character error rates: `0.0645, 0, 0.0435, 0,
0` (mean `0.0216`). This is a *relative regression check*: it shows no
detected speech-content change between the two settings, not a substitute for
human MOS evaluation.

The 6-step service also completed 100 consecutive requests successfully.
Its mean/median/P95 wall time was 497.7/494.4/507.5 ms for the longer
stability prompt. The first and last 20-request means were 501.1 ms and
500.2 ms, respectively, and GPU memory changed from 28,222 MiB to
28,244 MiB (+22 MiB). No request failed and no progressive memory growth was
observed.

To use a more conservative depth for a deployment, set it before starting the
service:

```bash
export COSYVOICE_FLOW_STEPS=8
python -m api_server.main
```

## Earlier retained improvements

The branch already contains two real-model optimizations that remain enabled:

- The Flow sampling depth was reduced from upstream's 10 steps to the previous
  8-step service default after a five-prompt ASR regression check.
- The request path no longer calls `torch.cuda.empty_cache()` and synchronizes
  the current CUDA stream after every synthesis. This avoids a forced global
  synchronization and allocator churn on each request.

On the same H100 test setup, removing the per-request cache clear reduced the
earlier 20-request mean from 425.3 ms to 382.4 ms; 8 Flow steps then measured
353.2 ms on that original prompt set. Results from separate prompt sets should
not be compared as a single absolute latency number.

## Candidate rejected in this run

An opt-in experiment skipped the all-true `B x L x L` DiT attention mask for
non-streaming requests without padding. It was mathematically plausible and
both variants passed the ASR comparison, but it had no stable end-to-end gain:
the 20-request mean was 450.7 ms with the candidate versus 437.3 ms without
it. Output durations also varied across service restarts, making RTF alone an
unsuitable decision metric. The code was reverted and no related environment
variable is exposed.

Other previously tested and rejected candidates include FP16 Flow inference,
F0 computation in FP32, removal of vLLM's short polling sleep, and an outer
Flow-mask allocation change. They did not show a reliable H100 service-level
improvement or were worse. vLLM already enables TorchInductor and CUDA Graphs
for its LLM stage, so the service does not duplicate that work.

## Reproducibility artifacts

The raw result files are intentionally outside Git because they include WAVs
and environment-specific logs. On the evaluation host they are stored under:

```text
/SharedData/yangyu/cosyvoice_h100_opt_results/
```

Relevant files from this run are `unmasked_off/summary.json`,
`unmasked_on/summary.json`, `unmasked_asr_comparison.json`,
`flow6/summary.json`, `flow6_asr.json`, and `flow6_stability_100.json`.

Before changing the default again, repeat the benchmark with a broader text,
voice, and language set, listen to samples, and run the same long-request
stability check. Keep `COSYVOICE_MAX_CONCURRENCY=1` until the real backend is
separately validated for concurrent inference.
