# CosyVoice3 vLLM 0.25.1 PD service-scenario benchmark

Date: 2026-09-11
Branch: `feature/cosyvoice-pd-h100`
Model: `Fun-CosyVoice3-0.5B/vllm`
GPU: H100; Unified tests used GPU 7, PD tests used Prefill GPU 2 + Decode GPU 4.
Decode mode: `FULL_DECODE_ONLY`; NIXL/UCX loopback transport.

This is an independent research PoC. It does not change the production API
path or production defaults.

## 1. Interference benchmark (30 repeats per length)

The persistent workers load the model, weights, CUDA graph and NIXL handshake
before measurement. Request A is decoded to token 40. The Decode worker then
signals the Prefill worker through a Unix-domain socket (no 50 ms file polling).
The Prefill worker records B start immediately before `engine.add_request` and B
end when its output is returned. A phase boundaries therefore use B's actual
prefill timestamps, not the barrier timestamp.

Values below are milliseconds. `run p99` is the mean plus standard deviation of
the 30 per-run p99 values; the last value is the largest per-run p99. `MAX` is
the maximum interval observed in the phase.

| B prompt | Unified P50 | Unified P95 | Unified P99 | Unified MAX | Unified run p99 mean +/- sd / max | PD P50 | PD P95 | PD P99 | PD MAX | PD run p99 mean +/- sd / max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512  | 2.553 | 11.229 | 15.408 | 24.306 | 10.155 +/- 2.707 / 23.656 | 2.382 | 2.705 | 3.142 | 4.804 | 2.994 +/- 0.397 / 4.488 |
| 1024 | 2.557 | 9.460  | 13.202 | 17.301 | 9.047 +/- 1.808 / 16.884  | 2.423 | 2.723 | 2.850 | 3.137 | 2.802 +/- 0.089 / 3.007 |
| 2048 | 2.410 | 9.366  | 10.557 | 16.226 | 9.060 +/- 1.302 / 15.838  | 2.407 | 2.702 | 3.061 | 5.211 | 2.866 +/- 0.386 / 4.662 |
| 4096 | 2.614 | 12.445 | 12.716 | 22.431 | 12.274 +/- 1.764 / 21.697 | 2.396 | 2.604 | 2.757 | 5.372 | 2.730 +/- 0.313 / 4.348 |
| 8192 | 2.456 | 23.465 | 24.354 | 29.154 | 22.738 +/- 0.955 / 27.737 | 2.415 | 2.696 | 2.944 | 5.161 | 2.848 +/- 0.339 / 4.296 |

The no-interference control p99 was 2.27--2.30 ms for Unified and
2.51--2.56 ms for PD. Thus the isolated PD decode path adds about 9--12% to
baseline TPOT, but removes the long-prefill tail: the observed phase p99 falls
by about 68--88% and the phase maximum by about 68--84% in these runs.

### B prefill timing decomposition

The vLLM offline `LLMEngine` path used by this PoC does not populate
`first_scheduled_time`, `model_forward_time` or `model_execute_time`; those
fields are recorded as null. The driver instrumentation records request-add
CPU time, every `engine.step()` wall time, and empty/non-empty steps. NVTX
ranges are emitted for the prefill and NIXL operations. A 20 ms NVML sampler is
also enabled, but B requests are shorter than that sampling period, so GPU
utilization is often unavailable.

The table is the steady portion (repeat 1--29), in milliseconds. `poll/residual`
is the wall time not accounted for by `add_request` and the measured step call;
it is approximately the 1 ms cooperative polling sleep per empty step.

| B prompt | wall | add_request | non-empty step | poll/residual | empty steps (mean) |
|---:|---:|---:|---:|---:|---:|
| 512  | 62.811 | 0.565 | 0.305 | 61.474 | 58.2 |
| 1024 | 68.040 | 0.754 | 0.198 | 66.503 | 62.9 |
| 2048 | 57.091 | 0.968 | 0.226 | 55.412 | 52.5 |
| 4096 | 79.202 | 1.389 | 0.177 | 77.138 | 73.1 |
| 8192 | 70.901 | 2.368 | 1.213 | 66.756 | 63.2 |

The first B request in a persistent PD Prefill worker took 1.21--1.26 s,
while later requests took 57--79 ms. Its driver trace contains about 1,140--
1,190 empty steps before the first output; this is a first-shape/graph/runtime
initialization artifact amplified by the 1 ms polling loop, not 1.2 s of GPU
matmul. It must be excluded from steady-state service numbers, while real
services should warm every expected prompt shape before admitting traffic.

Existing PyTorch profiler traces are available under
`pd_exp/results/pd_profile_trace/` and NVTX labels are present. A dedicated
profiler capture was not enabled inside the 30-repeat loop because it changes
runtime and produces very large traces; an exact GPU-only split for B therefore
remains a follow-up item. The current evidence supports the conclusion that the
steady wall time is dominated by driver polling/engine scheduling, not by a
1.2-second GPU prefill.

## 2. Complete TTS pipeline pilot

Pipeline measured: `Frontend -> LLM -> Flow -> HiFT -> WAV`, with local wetext
normalization and the research-only soundfile loader fallback. Unified has three
fresh runs; PD has one fresh run with two iterations, and the second iteration
is the steady sample. PD startup (`67.8 s`) is excluded from steady-state E2E.

| mode | frontend (s) | LLM (s) | TTFT (s) | Flow (s) | HiFT (s) | acoustic total (s) | steady E2E (s) | audio (s) | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unified, run 0 | 0.0066 | 0.4889 | 0.0581 | 0.5709 | 0.2497 | 0.8425 | 1.3790 | 7.24 | 0.1905 |
| Unified, run 1 | 0.0056 | 0.4814 | 0.0557 | 0.5342 | 0.2684 | 0.8228 | 1.3537 | 7.24 | 0.1870 |
| Unified, run 2 | 0.0057 | 0.4797 | 0.0501 | 0.4789 | 0.2251 | 0.7209 | 1.2432 | 7.24 | 0.1717 |
| PD, steady iteration | 0.0050 | see decode below | 0.0126 first decode wait | 0.4643 | 0.2147 | 0.6976 | 1.0884 | 6.40 | 0.1701 |

Unified LLM TPOT p99 was 2.261--2.307 ms. PD steady decode TPOT p99 was
2.338 ms. Both WAV quality checks passed (`quality_acceptable=true`, clipping
ratio 0). The PD sample generated 166 raw/160 filtered tokens versus the
Unified sample's 188/181; this is the already classified BF16 + top-p sampling
sensitivity, so the different audio durations are not a PD graph correctness
failure. CER/WER and speaker-similarity scorers are not wired into this PoC;
only the existing silent-frame/clipping/audio-anomaly checks were available.
Consequently these are pipeline pilots, not a statistically valid E2E P99
comparison.

## 3. Same-cost two-GPU mixed workload pilot

For a cost-shape experiment, each batch alternated synthetic 512-token and
4096-token prompt embeddings. This isolates scheduler/prefill pressure and is
not a quality test. Each concurrency used two iterations (the second is the
warm sample), not 30 repetitions. Acoustic Flow/HiFT were skipped; values are
LLM-only. `rps` is the warm batch lower-bound throughput.

| concurrency | Unified wall (s) | Unified rps | Unified TPOT P50/P95/P99/MAX (ms) | PD wall (s) | PD rps | PD TPOT P50/P95/P99/MAX (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 4  | 0.174 | 23.00 | 2.448 / 2.517 / 17.364 / 20.393 | 0.226 | 17.67 | 2.473 / 2.645 / 11.641 / 44.004 |
| 8  | 0.194 | 41.24 | 2.493 / 2.596 / 4.096 / 37.893  | 0.325 | 24.64 | 2.563 / 2.986 / 3.477 / 133.025 |
| 16 | 0.298 | 53.70 | 2.769 / 3.045 / 47.702 / 48.696 | 0.525 | 30.47 | 2.865 / 3.565 / 4.539 / 225.644 |
| 32 | 0.395 | 81.00 | 3.417 / 3.965 / 42.457 / 50.526 | 0.907 | 35.28 | 3.406 / 3.896 / 4.667 / 575.078 |

The PD throughput is lower in this two-iteration token-only pilot because every
request pays Prefill scheduling/KV-transfer overhead and Decode is intentionally
kept on a separate GPU. The very large pilot MAX values are single observations
from only two batches and are not P99 estimates.

## 4. Production decision

* At the token-level interference point, PD clearly isolates Decode from a long
  Prefill: p99 and max TPOT remain near 3--5 ms while Unified shows 10--29 ms
  tails. This is a real benefit for a Decode P99 SLA under mixed long/short
  traffic.
* At the complete TTS level, Flow/HiFT currently consume about 0.7--0.84 s,
  much more than a 2--3 ms LLM token step. The pilot does not yet prove that the
  token-level tail reduction survives as an E2E P99 improvement. PD also has a
  measured 9--12% no-interference TPOT overhead and consumes an additional H100.
* Therefore PD is not yet justified as the default production path for low
  concurrency or short prompts. It is a candidate for production when long
  prompts are common, concurrency is high (the intended decision region is
  roughly >=8), and Decode tail latency is a hard SLA; the extra Prefill GPU
  must be acceptable.
* Before integration, run at least 30 real-text requests per concurrency for
  both modes with Flow/HiFT enabled, collect E2E P50/P95/P99, TTFT/TPOT, GPU
  utilization, audio duration, CER/WER, speaker similarity and anomaly rate.
  Keep `COSYVOICE_MAX_CONCURRENCY=1` for the current real-model service until
  that study is complete.

## Reproduction commands

Interference (30 repeats and event-level boundary):

```bash
export PYTHONPATH=/SharedData/yangyu/CosyVoice_pd:/SharedData/yangyu/CosyVoice_pd/third_party/Matcha-TTS
export VLLM_USE_FLASHINFER_SAMPLER=0
export COSY_PD_CONNECTOR_MODULE=pd_exp.prompt_nixl_connector_0251
export UCX_TLS=tcp,cuda_ipc,cuda_copy,sm,self
export UCX_NET_DEVICES=lo
/SharedData/yangyu/cosyvoice_vllm0251_env/bin/python pd_exp/pd_interference_benchmark_persistent.py --python /SharedData/yangyu/cosyvoice_vllm0251_env/bin/python --p-gpu 2 --d-gpu 7 --model-dir /SharedData/yangyu/CosyVoice_pd/pretrained_models/Fun-CosyVoice3-0.5B/vllm --prompt-payload /SharedData/yangyu/CosyVoice_pd/pd_exp/results/service_unified_full_pilot/prompt_payload.pt --baseline-tokens /SharedData/yangyu/CosyVoice_pd/pd_exp/results/service_unified_full_pilot/baseline_tokens.json --results-root /SharedData/yangyu/CosyVoice_pd/pd_exp/results/interference_pd_final30 --lengths 512 1024 2048 4096 8192 --repeats 30 --instrument-nixl --timeout 1800
```

Unified mixed pilot:

```bash
/SharedData/yangyu/cosyvoice_vllm0251_env/bin/python pd_exp/benchmark_pd.py --gpu 7 --prompt-payload-list /SharedData/yangyu/CosyVoice_pd/pd_exp/results/service_cost_mixed/payloads_c16.json --baseline-tokens /SharedData/yangyu/CosyVoice_pd/pd_exp/results/service_unified_full_pilot/baseline_tokens.json --concurrency 16 --iterations 2 --engine-mode standard_optimized --output /tmp/unified_mixed.json
```

PD mixed pilot:

```bash
export COSY_PD_CONNECTOR_MODULE=pd_exp.prompt_nixl_connector_0251
/SharedData/yangyu/cosyvoice_vllm0251_env/bin/python pd_exp/coordinator.py --prefill-gpu 2 --decode-gpu 4 --acoustic-gpu 7 --prompt-payload-list /SharedData/yangyu/CosyVoice_pd/pd_exp/results/service_cost_mixed/payloads_c16.json --baseline-tokens /SharedData/yangyu/CosyVoice_pd/pd_exp/results/service_unified_full_pilot/baseline_tokens.json --results-dir /tmp/pd_mixed --skip-acoustic --standard-optimized --iterations 2 --batch-size 16 --timeout 1200
```
