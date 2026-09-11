# CosyVoice3 PD production decision (vLLM 0.25.1)

Date: 2026-09-11
Branch: `feature/cosyvoice-pd-h100`
Model: `Fun-CosyVoice3-0.5B` (vLLM backend)

This document summarizes the independent PD PoC. It does not change the
production API/default path.

## Executive decision

PD is a real Decode-tail isolation mechanism, but it is not yet a safe default
for the complete CosyVoice online service. In the current measurements, Flow
and HiFT dominate the request cost and the PD two-GPU token-only path has lower
throughput than two Unified replicas. PD should be considered for a high-
concurrency, long-prompt, multi-tenant service with a hard Decode P99 SLA; it
should not be enabled for low-concurrency or mostly short requests until a
larger online E2E study confirms a P99 benefit.

## 1. Capabilities already verified

* **Correctness:** Unified and PD `FULL_DECODE_ONLY` argmax sequences matched in
the 400-step deterministic diagnostic. Random top-p token differences were
classified as BF16 numerical/sampling sensitivity, not a PD graph error.
* **KV transfer:** NIXL producer/consumer KV blocks were verified bitwise equal;
the remote block metadata and receiver cache state were consistent.
* **CUDA Graph:** PD graph runs are stable after warmup; the earlier 159/162
sampling discrepancy is not a graph computation bug.
* **Interference isolation:** With 30 repeats at each B prompt length, PD kept
A Decode phase P99 near 2.8--3.1 ms while Unified reached 10.6--24.4 ms.

## 2. Complete E2E benchmark (30 requests per mode)

Chain: `Frontend -> LLM -> Flow -> HiFT -> WAV`, fixed text/speaker/prompt and
`FULL_DECODE_ONLY`. Unified and PD each produced 30 valid WAV files. PD LLM
requests were persistent; its acoustic stage was run as each decode result
became available by a research harness. The PD E2E number is therefore a
per-request estimate (`frontend + LLM + acoustic`), not a single wall-clock
trace with Flow/HiFT overlap.

| metric | Unified (n=30) | PD (n=30) |
|---|---:|---:|
| E2E P50 / P95 / P99 / MAX | 0.650 / 0.682 / 1.185 / 1.390 s | 0.652 / 0.661 / 1.968 / 2.502 s |
| TTFT P50 / P95 / P99 | 11.24 / 15.12 / 47.99 ms | 12.11 / 14.10 / 804.28 ms |
| LLM total P50 / P95 / P99 | 0.381 / 0.397 / 0.437 s | 0.393 / 0.399 / 1.201 s |
| LLM TPOT P50 / P95 / P99 | 2.188 / 2.254 / 2.264 ms | 2.297 / 2.407 / 2.549 ms |
| Flow P50 / P95 / P99 | 222.5 / 223.7 / 506.4 ms | 222.0 / 223.9 / 538.8 ms |
| HiFT P50 / P95 / P99 | 26.23 / 27.03 / 206.8 ms | 25.52 / 26.47 / 211.7 ms |
| acoustic total P50 / P95 / P99 | 257.2 / 271.7 / 733.7 ms | 256.0 / 258.5 / 763.9 ms |
| RTF P50 / P95 / P99 | 0.099 / 0.104 / 0.181 | 0.102 / 0.103 / 0.308 |
| WAV duration | 6.56 s (all 30) | 6.40 s (all 30) |
| acceptable audio / clipping | 30/30; max 0 | 30/30; max 0 |
| CER / WER / speaker similarity | not wired | not wired |

The E2E result does **not** yet show a PD tail win: the posthoc PD P99 is higher,
mainly because several PD LLM/KV requests include first-request or transfer
outliers. The different 6.40 s vs 6.56 s durations come from the known random
token-count sensitivity (PD 166/160 raw/filtered tokens, Unified 170/164), not
from forced truncation. No CER/WER or speaker-similarity evaluator is present
in this PoC, so quality claims are limited to the existing anomaly/silence and
clipping checks.

Raw result directories:

* Unified: `pd_exp/results/service_unified_full30/summary.json`
* PD LLM: `pd_exp/results/service_pd_llm30/pd_metrics.json`
* PD full harness: `pd_exp/results/service_pd_full30/summary.json`

## 3. Same-cost online-shape workload

The following scan uses two H100s for each side: two Unified replicas (one per
GPU) versus one Prefill + one Decode worker. Each batch contains synthetic
prompt embeddings at the indicated length; acoustic Flow/HiFT are skipped, so
this is a scheduler/capacity experiment, not an audio-quality benchmark. Each
cell has one pass per length and is directional rather than a production P99
estimate.

For long prompts, aggregate warm throughput (requests/s) was:

| prompt | concurrency 4 | concurrency 8 | concurrency 16 | concurrency 32 |
|---:|---:|---:|---:|---:|
| 4096 | 2U 27.66 vs PD 13.39 | 46.79 vs 17.04 | 74.34 vs 19.83 | 112.14 vs 21.90 |
| 8192 | 2U 21.46 vs PD 7.84 | 34.32 vs 9.38 | 49.79 vs 10.69 | 62.63 vs 11.47 |

Values are `2 x Unified replicas vs 1P+1D PD`, measured on the same two-GPU
budget. PD TPOT medians were close to Unified, but PD throughput was lower due
to Prefill scheduling, KV transfer and the current worker admission policy.

Observed direction matrix (provisional; `PD` means a materially lower measured
TPOT P99, `Unified` means no demonstrated tail advantage or a cost/throughput
loss):

| prompt_len \\ concurrency | 1 | 4 | 8 | 16 | 32 |
|---:|:---:|:---:|:---:|:---:|:---:|
| 256  | Unified | Unified | Unified | Unified | Unified |
| 512  | Unified | Unified | Unified | Unified | Unified |
| 1024 | Unified | Unified | PD | Unified | PD |
| 2048 | Unified | Unified | PD | PD | PD |
| 4096 | Unified | PD | PD | PD | PD |
| 8192 | Unified | PD | PD | PD | PD |

Equivalent YAML view:

```yaml
concurrency: [1, 4, 8, 16, 32]
recommendation:
  256:  [Unified, Unified, Unified, Unified, Unified]
  512:  [Unified, Unified, Unified, Unified, Unified]
  1024: [Unified, Unified, PD,     Unified, PD]
  2048: [Unified, Unified, PD,     PD,     PD]
  4096: [Unified, PD,     PD,     PD,     PD]
  8192: [Unified, PD,     PD,     PD,     PD]
```

The matrix is a boundary-finding experiment, not a deployment gate: each cell
has too few repetitions for a statistically reliable P99, and the embeddings
are synthetic repetitions of the canonical prompt. Real text and real speaker
prompts can move the boundary.

## 4. GPU/resource efficiency

The two-Unified and PD scans used the same two-GPU budget. Thus
`req/s/H100 = aggregate req/s / 2`; the 4096/8192 figures above imply that the
current PD PoC has lower effective service capacity per H100 in this workload.
`latency/H100` can be derived from the reported latency divided by two, but
that metric is not a useful SLA quantity.

Per-GPU utilization was not recorded for the matrix runs. The existing NVML
sampler polls every 20 ms, while many individual token-only batches finish in
less than one polling interval; the resulting samples are empty or misleading.
The interference worker records this limitation explicitly. A production
capacity decision still requires a longer-duration load generator with a
continuous DCGM/NVML sampler for Prefill, Decode and Unified GPUs.

## 5. Cost and workload interpretation

* **Benefit:** PD reduces Decode P99/MAX during a long concurrent Prefill by
roughly 68--88% / 68--84% in the 30-repeat interference study.
* **Overhead:** isolated PD TPOT is about 9--12% above Unified control; it also
requires a dedicated Prefill H100 and pays NIXL transfer/admission overhead.
* **Full pipeline:** Flow/HiFT account for roughly 0.25--0.28 s at the median,
so an LLM-only tail win is not automatically an E2E win. The 30-request pilot
currently shows no E2E P99 improvement.
* **Useful-service capacity:** with the current PoC, two Unified replicas have
higher aggregate throughput than 1P+1D for the synthetic scan. PD is therefore
not merely a free throughput optimization; its value is primarily an SLA/tail
isolation tradeoff.

## 6. Deployment recommendation

### Recommend evaluating PD

* Multi-tenant or mixed short/long request traffic.
* Prompt/speaker context commonly at least 2K--4K tokens.
* Concurrency at least 8.
* Decode P99/MAX is a contractual SLA and an extra Prefill H100 is acceptable.
* A scheduler can keep both workers warm and avoid first-shape cold starts.

### Do not deploy PD by default

* Single-request or low-concurrency traffic.
* Mostly short (256--512 token) prompts.
* Throughput/$ is more important than Decode tail latency.
* A single GPU is available, or Flow/HiFT dominates the SLA and no E2E tail
benefit has been demonstrated.

Before integration, run a real online load test with 30 or more requests per
(workload, concurrency) cell, real speaker audio, Flow/HiFT in the critical
path, continuous GPU telemetry, CER/WER and speaker-similarity scoring. Keep
the production default on Unified until that gate passes.

## Reproduction entry points

* `pd_exp/full_unified_service_benchmark.py` -- persistent 30-request Unified
  E2E harness.
* `pd_exp/full_pd_service_benchmark.py` -- PD acoustic completion harness.
* `pd_exp/benchmark_pd.py` -- Unified token-only matrix driver (supports a
  payload sequence list).
* `pd_exp/coordinator.py` -- PD worker coordinator (supports a payload sequence
  list and LLM-only canonical runs).
* `pd_exp/pd_interference_benchmark_persistent.py` -- 30-repeat event-boundary
  interference benchmark.
