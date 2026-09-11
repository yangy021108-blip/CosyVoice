# CosyVoice3 vLLM 0.25.1 Prefill/Decode follow-up

## Scope

This report records the minimal prompt_embeds -> Prefill -> NIXL GPU KV READ -> Decode -> speech tokens experiment after upgrading the isolated environment to vLLM 0.25.1. Flow, HiFT, frontend, and HTTP are not part of this case.

Environment: H100, PyTorch 2.11.0+cu130, vLLM 0.25.1, NIXL 0.10.1, Transformers 5.17.0, one P worker on GPU 7 and one D worker on GPU 1. The canonical input is the serialized prompt_payload.pt (131 x 896 BF16) and the historical baseline trajectory (162 raw / 154 filtered tokens).

## Four-way result

| Case | Raw tokens | Filtered tokens | First mismatch | Mean TPOT |
|---|---:|---:|---|---:|
| Unified eager | 164 | n/a | differs from historical baseline | 7.45 ms |
| Unified FULL_DECODE_ONLY | 162 | n/a | none vs historical baseline | 2.34 ms |
| PD eager, canonical prefill | 162 | 154 | none | 8.60 ms |
| PD eager, prefill exports prompt[:-1] | 162 | 154 | none | 7.84 ms |
| PD FULL_DECODE_ONLY + graph safety | 159 | n/a | output index 121, position 251 | 2.67 ms |

The graph result above is the standard optimized configuration with the first local prompt-continuation step forced eager. cudagraph_copy_inputs=1 and an explicit CUDA synchronize after NIXL completion were also tested; both still produced 159 raw tokens (3.51 ms and 3.43 ms respectively).

Raw counts exclude the reserved stop token 6562. The graph trajectory has a different length, so the canonical-only coordinator correctly rejects it instead of filtering or truncating it.

## Step-by-step diagnosis

The first local computation after remote KV is:

    num_computed_tokens = 130
    num_prompt_tokens   = 131
    position            = 130
    query_len           = 1
    seq_len             = 131
    block_table         = [1, 2, ..., 9]
    slot_mapping        = 146

This is a prompt continuation, not a normal autoregressive decode. The vLLM 0.25.1 graph dispatcher originally classified it as FULL decode. The experiment now detects this condition from scheduled_cached_reqs and bypasses CUDA Graph for this one step. The next step (position 131) enters FULL decode normally.

After this semantic fix, all request state remains aligned. At the first sampled-token mismatch (raw output index 121):

    position       = 251
    seq_len        = 252
    query_len      = 1
    block_table    = [1, 2, ..., 16]
    slot_mapping   = 267

Unified graph and PD graph have the same top-k token IDs, but BF16 logits differ by one quantization step:

    Unified: [12.125, 12.125, 11.875, 11.8125, 11.75, ...]
    PD:      [12.125, 12.0625, 11.875, 11.8125, 11.75, ...]

The sampled token changes from 1433 to 703. This is not a stale position, sequence length, block table, or slot-mapping value. The trace reads slot mapping from the vLLM 0.25.1 GPU tensor (slot_mapping.gpu); the previous CPU mirror was always zero and was only a trace bug.

## Code changes in the experiment worktree

- pd_exp/prompt_nixl_connector_0251.py: vLLM 0.25.1 connector adapter. The scheduler uses request.num_prompt_tokens because CosyVoice prompt embeddings have no prompt_token_ids.
- pd_exp/graph_runtime.py: vLLM 0.25.1 dispatcher/prepare-input signatures, detection of the first local prompt continuation, graph-safety bypass, and GPU slot-mapping tracing.
- pd_exp/prefill_worker.py and pd_exp/coordinator.py: opt-in COSY_PD_TRUNCATE_PREFILL_LAST=1, exporting KV for prompt[:-1] so Decode computes the final prompt position locally.
- pd_exp/common.py: configurable connector module and opt-in COSY_PD_CUDAGRAPH_COPY_INPUTS=1.
- pd_exp/prompt_nixl_connector_0251.py: opt-in COSY_PD_NIXL_SYNC=1 diagnostic synchronization after NIXL completion.
- cosyvoice/vllm/cosyvoice2.py: import Union required by the vLLM 0.25.1 model adapter.

All modified Python files pass compileall.

## Root cause and decision

The first graph error was a real protocol classification bug: the first local prompt token after remote prefill was sent to the full-decode graph. That bug is fixed in the experiment path.

The remaining strict-token mismatch is numerical. NIXL supplies BF16 KV produced by a separate worker, while Unified Graph consumes KV produced in the same engine. The resulting sub-ULP/one-BF16-step logit changes are enough to change stochastic top-p sampling on near ties. Eager PD remains token-identical, and the graph remains around 2--3 ms TPOT, but exact graph equality is not demonstrated.

No sampling rule, seed, output truncation, or correctness check was changed to hide the mismatch. Therefore PD FULL_DECODE_ONLY is not promoted to a correctness-approved path yet. Keep PD eager as the valid PoC; investigate an upstream newer NixlConnector/vLLM branch or an explicitly numerically reproducible KV path before running an interference benchmark.


## Correctness gate update: Unified Graph versus PD Graph (2026-09-11)

The requested correctness baseline is now **Unified FULL_DECODE_ONLY versus PD
FULL_DECODE_ONLY**.  Unified eager is retained only as context (its sampled
trajectory is 164 tokens), and is no longer a Graph correctness gate.

### KV transfer

Debug hashing was enabled for the actual NIXL READ.  The producer and consumer
covered the same physical blocks [1..9] and all 24 registered attention KV
cache tensors.  The producer hash was taken after the prefill request had been
released (the transfer had already completed), and the consumer hash was taken
after NIXL completion.  The two source records were identical to each other and
matched the receiver record:

    source SHA-256 = b05db585c8bf590101a89ff3798767a11aebd0de9520f6862cf5d612c3cc6ea8
    receiver SHA-256 = b05db585c8bf590101a89ff3798767a11aebd0de9520f6862cf5d612c3cc6ea8
    per-cache hashes = 24/24 equal

Therefore the remote KV payload is bitwise identical for this case.

### Deterministic argmax

With the same prompt and FULL_DECODE_ONLY graph configuration, the diagnostic
sampler selected argmax(logits) at every step.  Unified Graph and PD Graph
matched for the complete 400-step diagnostic trajectory (400/400 tokens, no
first difference).  This mode is diagnostic-only; production top-p sampling
was not changed.  The 400-step cap was reached because raw argmax does not
necessarily emit the reserved stop token.

### First sampled divergence

The first production-sampler divergence remains output index 121 at position
251 (seq_len=252, query_len=1, slot_mapping=267, block table [1..16]).  Both
complete logits dumps contain 6,761 vocabulary values:

    pd_exp/results/v0251_unified_norm2_logits.jsonl
    pd_exp/results/v0251_pd_norm2_logits.jsonl

The full-logit comparison at that step is:

| stage | max_abs_diff | mean_abs_diff |
|---|---:|---:|
| model logits | 0.0625 | 0.0142457129 |
| logits after repetition penalty, before top-p | 0.0625 | 0.0142431819 |

The raw argmax is token 1432 on both sides.  Near the top-p boundary, one BF16
step changes token 3620 from 12.125 (Unified) to 12.0625 (PD); after the
repetition penalty it is 11.022727 versus 10.965909.  top_k=25 and top_p=0.8
leave 15 candidates in each case, but the candidate sets have 14/15 tokens in
common:

    Unified: [1432,1433,197,196,3619,463,3620,706,1434,439,1192,170,223,466,1435]
    PD:      [1432,1433,197,196,3619,463,706,1434,3620,439,1192,170,466,1435,703]

The cutoff is rank 10.  Its cumulative probability is 0.8046365 (Unified)
versus 0.8049110 (PD); rank 9 is 0.7635735 versus 0.7634092.  Thus a normal
BF16 quantization step moves a near-tied token across the top-p ordering/candidate
boundary.  With the same seeded random draw, the sampled token becomes 1433
(Unified) versus 703 (PD).  This is sampling sensitivity, not an argmax or
KV-transfer discrepancy.

### Repetition and batch-invariance checks

Normal sampling was repeated ten times per mode with one request at a time:

| mode | runs | token count | unique sequences | mean TPOT |
|---|---:|---:|---:|---:|
| Unified FULL_DECODE_ONLY | 10 | 162 every run | 1 | 2.220 ms |
| PD FULL_DECODE_ONLY + graph-safe first local prompt step | 10 | 159 every run | 1 | 2.565 ms |

VLLM_BATCH_INVARIANT=1 was also tried as a diagnostic-only configuration.  It
did not align the modes: Unified produced 162 tokens at 4.102 ms TPOT while PD
produced 164 tokens at 4.389 ms TPOT.  It is not enabled for production.

### Decision

KV is bitwise identical, argmax is fully identical, and the first sampled
difference is only a normal BF16-scale logit perturbation amplified by top-p.
No remaining **PD Graph calculation bug** is demonstrated by this minimal case.
Do not alter the production sampler, truncate output, or disable correctness
checks to hide the difference.  The strict sampled-token equality gate is
reclassified as numerical/sampling sensitivity; production validation should
use audio/quality statistics and seed-repeat distributions.

It is now reasonable to proceed to the PD interference/P99 benchmark, using
Unified FULL_DECODE_ONLY versus PD FULL_DECODE_ONLY as the performance baseline
and retaining token/audio quality monitoring.  The previously identified
first-local-prompt graph classification fix remains in place.
