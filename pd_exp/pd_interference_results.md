# CosyVoice3 vLLM 0.25.1 PD interference / P99 benchmark

Status: research-only benchmark. Production API/defaults and CUDA Graph dispatch were not changed.

## Scope and protocol

- Official comparison: Unified FULL_DECODE_ONLY versus 1P1D PD FULL_DECODE_ONLY.
- vLLM 0.25.1, BF16, max_num_seqs=2, canonical 131x896 prompt embeddings, normal top-p sampling.
- Three repetitions for each B prompt length: 512, 1024, 2048, 4096, 8192 (15 cases per mode).
- Both PD workers build the engine, load weights, compile/warm CUDA Graphs, and complete the NIXL handshake in warm/control requests before measured A.
- P waits on a barrier; D writes it when A reaches output token 40. P then starts B on its already-ready engine. Unified injects B into the same engine/GPU at A token 40.
- TPOT phase boundaries use monotonic timestamps. during is from barrier/injection until B completion; after is N/A when A has already completed (this occurs for every PD case).
- B has one auxiliary output token (max_tokens=1); only prefill wall time is measured. No Flow/HiFT/frontend is included.

## A Decode TPOT (ms)

Values are the mean of three per-run summaries. Each cell is P50 / P95 / P99 / MAX.

| B prompt | Unified control | Unified before | Unified during | Unified after | PD control | PD before | PD during | PD after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512  | 2.26 | 2.19 / 2.21 / 2.23 / 2.25 | 5.68 / 12.94 / 13.85 / 14.07 | 2.24 / 2.28 / 2.34 / 2.45 | 2.42 | 2.37 / 2.43 / 2.50 / 2.54 | 2.46 / 2.64 / 3.31 / 4.33 | N/A |
| 1024 | 2.20 | 2.15 / 2.19 / 2.25 / 2.28 | 1.83 / 12.62 / 20.21 / 22.18 | 2.22 / 2.25 / 2.29 / 2.45 | 2.45 | 2.40 / 2.46 / 2.79 / 2.99 | 2.55 / 2.72 / 3.33 / 5.62 | N/A |
| 2048 | 2.22 | 2.18 / 2.21 / 2.26 / 2.27 | 2.72 / 11.36 / 12.13 / 12.33 | 2.24 / 2.27 / 2.32 / 2.66 | 2.44 | 2.40 / 2.44 / 2.48 / 2.50 | 2.49 / 2.68 / 3.46 / 4.18 | N/A |
| 4096 | 2.19 | 2.15 / 2.20 / 2.22 / 2.23 | 2.57 / 13.19 / 14.86 / 15.28 | 2.20 / 2.24 / 2.32 / 2.44 | 2.43 | 2.39 / 2.43 / 2.45 / 2.45 | 2.46 / 2.58 / 2.68 / 2.72 | N/A |
| 8192 | 2.19 | 2.15 / 2.20 / 2.23 / 2.24 | 2.26 / 18.92 / 24.54 / 25.94 | 2.21 / 2.25 / 2.30 / 2.35 | 2.43 | 2.39 / 2.45 / 2.49 / 2.50 | 2.49 / 2.70 / 2.97 / 4.52 | N/A |

## Cost and benefit

| B prompt | PD control overhead vs Unified | Unified during P99 -> PD during P99 | P99 reduction | Unified observed MAX -> PD observed MAX | MAX reduction | PD B Prefill wall | PD KV transfer |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512  | 7.1%  | 13.85 -> 3.31 | 76.1% | 19.79 -> 5.53 | 72.0% | 1183.23 ms | 5.08 ms |
| 1024 | 11.8% | 20.21 -> 3.33 | 83.5% | 44.92 -> 7.97 | 82.3% | 1166.70 ms | 6.27 ms |
| 2048 | 9.6%  | 12.13 -> 3.46 | 71.5% | 18.03 -> 5.79 | 67.9% | 1166.72 ms | 5.49 ms |
| 4096 | 11.1% | 14.86 -> 2.68 | 82.0% | 21.41 -> 2.90 | 86.4% | 1269.56 ms | 3.06 ms |
| 8192 | 11.2% | 24.54 -> 2.97 | 87.9% | 29.65 -> 6.20 | 78.9% | 1158.53 ms | 6.77 ms |

PD control TPOT is about 2.42-2.45 ms versus Unified about 2.19-2.26 ms, an extra about 7.1-11.8%. The extra GPU cost is one resident Prefill worker plus about 3-7 ms measured NIXL READ completion per A request.

## B Prefill and GPU observations

PD B Prefill wall time is approximately 1.13-1.21 s across these lengths. This wall time includes vLLM scheduler/CPU waiting, not only GPU kernels. NVML polling at 20 ms on P GPU0 recorded low busy percentages for short bursts; for 8192 the three runs were approximately 3.7-3.9% mean and 21-23% peak. This is a useful observation, not a substitute for a GPU-kernel trace.

## Output checks

- All 30 measured A requests completed with non-empty integer token sequences and no worker exception.
- Unified raw token counts across the matrix were 157, 159, 161, 162, and 168; PD produced 159 tokens in all 15 measured cases. This is the previously known graph/sampling behavior, not a new interference failure.
- This stripped case deliberately stops before Flow/HiFT, so audio duration, CER/WER and waveform quality are not available here; they remain required for normal product/API validation.
- Token equality is not used as the Graph correctness gate; prior deterministic argmax and KV bitwise checks remain the correctness evidence.

## Conclusion

PD clearly isolates A Decode from a long B Prefill on this H100: during-window P99 fell by about 71.5-87.9% and the largest observed during-window MAX fell by about 67.9-86.4%, while no-interference TPOT overhead was about 7-12%. The benefit is largely independent of B length because the Unified scheduler suffers tail stalls whereas the separate Decode GPU does not.

**Recommendation:** PD is worthwhile for tail-sensitive multi-tenant serving when a dedicated Prefill GPU is available. Do not integrate it as the production default yet: first repeat with the full Flow/HiFT path, normal audio-quality checks, a longer A workload so an after phase is observable, and a larger sample (at least 30-100 cases per length) for stable P99/P99.9 estimates.

## Reproduction artifacts

- Unified JSON: `pd_exp/results/interference_unified_full.json`
- PD JSON: `pd_exp/results/interference_pd_final/pd_interference_summary.json`
- PD per-case logs/traces: `pd_exp/results/interference_pd_final/pd/len_<L>_rep_<N>/`
- Experimental drivers: `pd_exp/unified_interference_benchmark.py`, `pd_exp/pd_interference_benchmark_ready.py`, `pd_exp/pd_interference_prefill_ready.py`, `pd_exp/pd_interference_decode_ready.py`
- NIXL transfer timing is opt-in via `--instrument-nixl` and `COSY_PD_TRACE_FILE`.
