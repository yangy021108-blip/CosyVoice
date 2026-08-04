# CosyVoice SDAA vLLM 优化第五轮：RoPE 复用与 HiFT F0 上卡

> 2026-07-30 审核说明：本文的候选交错 A/B 有原始数据支持，但旧在线结果
> 只有一次服务运行且包含异常慢请求。新增的两次独立启动、20 请求/配置、
> 对称 HTTP A/B 见
> [端到端延迟审核](cosyvoice_sdaa_vllm_e2e_latency_audit_20260730.md)。

## 结论

本轮从第四轮最终 Perfetto trace 的剩余热点继续分析，在不减少 Flow
步数、不缩短音频、不改变采样参数的前提下，采用两项新优化：

1. Flow RoPE 的 sine/cosine 从“每个 DiT block 重算”改成“每个请求只算
   一次并复用”。
2. CausalHiFT 的 F0 predictor 在 SDAA 推理时使用 FP32，避免原始
   float64 路径产生 CPU fallback 和设备间拷贝。

最终结果：

- RoPE 预计算单独交错 A/B 提升 0.65%，20/20 对音频长度和 SHA256
  完全一致。
- F0 SDAA FP32 在 RoPE 版本之上再提升 2.41%，20/20 对全部更快。
- F0 A/B 的基线和候选 ASR 均为 16/568 字符错误，20/20 条原始转写和
  规范化转写完全一致。
- 最终 Perfetto 中完整请求均值从 2596.07 ms 降到 2459.74 ms，
  降低 5.25%；HiFT 阶段从 187.53 ms 降到 148.42 ms，降低 20.86%。
- 同 seed 在线 EvalScope 10/10 成功、零质量重试；即使包含一次
  3.70 s 异常慢请求，服务端均值仍降低 1.74%，吞吐提高 1.64%。
- 同 seed 最终在线音频 ASR 为 4/284，CER 1.4085%，10/10 非空。

服务启动配置新增：

```bash
export COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE=1
export COSYVOICE_SDAA_HIFT_F0_FP32=1
```

## 实现位置

| 文件 | 作用 |
| --- | --- |
| `cosyvoice/flow/DiT/modules.py` | 定义预计算 RoPE 数据结构和数值等价的应用路径 |
| `cosyvoice/flow/DiT/dit.py` | 每个 Flow 请求只生成一次 sine/cosine |
| `cosyvoice/hifigan/generator.py` | SDAA 推理模式下用 FP32 执行 CausalHiFT F0 predictor |
| `tools/benchmark_sdaa_flow_rope.py` | RoPE 原始路径与预计算路径的数值和微基准验证 |
| `tools/benchmark_cosyvoice_sdaa_candidates.py` | 同进程交错顺序的端到端候选 A/B |

所有新路径都由环境开关控制，并限定为 SDAA 推理。关闭开关、设备不是
SDAA 或处于梯度模式时，代码自动回到原始实现。

## Flow RoPE 预计算

第四轮 trace 中，两个请求共出现约 1440 次 `aten::sin` 和 1444 次
`aten::cos`。位置频率在一次请求的 22 个 DiT block 和 8 个 Flow step
之间不变，因此无需在每个 attention 中重复计算。

固定形状 batch 2、序列长度 256、hidden size 512、rotary dim 64，
100 次微基准：

| 指标 | 原始路径 | 预计算路径 | 变化 |
| --- | ---: | ---: | ---: |
| 平均耗时 | 0.72728 ms | 0.51643 ms | -28.99% |
| query 最大绝对误差 | - | 0 | 完全一致 |
| key 最大绝对误差 | - | 0 | 完全一致 |

手动复现：

```bash
/opt/kube/bin/docker exec \
  -e SDAA_VISIBLE_DEVICES=19 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  bash -lc 'source /opt/tecoai/setvars.sh && source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/miniconda3/envs/vllm_env_py310 && python tools/benchmark_sdaa_flow_rope.py --warmup 20 --iterations 100'
```

端到端 4 轮交错 A/B 结果：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/rope_precompute_interleaved/results.json`

| 指标 | fused norm 基线 | RoPE 预计算 | 变化 |
| --- | ---: | ---: | ---: |
| 平均推理时延 | 2.55476 s | 2.53807 s | -0.65% |
| 平均 RTF | 0.40331 | 0.40066 | -0.66% |
| 候选更快 | - | 16/20 | - |
| WAV SHA256 一致 | - | 20/20 | - |
| 音频长度一致 | - | 20/20 | - |

## HiFT F0 SDAA FP32

原始 `CausalHiFTGenerator.inference` 强制 F0 predictor 使用 float64。
在当前 Torch-SDAA 运行时中，该路径的部分卷积和激活会发生 CPU
fallback。新路径禁用 autocast 后用 SDAA FP32 执行 F0 predictor，其余
HiFT 仍保持原来的 dtype 和执行路径。

4 轮交错 A/B 结果：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/f0_fp32_interleaved_v2/results.json`

| 指标 | RoPE 基线 | F0 SDAA FP32 | 变化 |
| --- | ---: | ---: | ---: |
| 平均推理时延 | 2.55318 s | 2.49154 s | -2.41% |
| 平均 RTF | 0.40313 | 0.39326 | -2.45% |
| 候选更快 | - | 20/20 | - |
| 音频长度一致 | - | 20/20 | - |
| 质量重试 | 0 | 0 | 持平 |

四轮候选都更快：

| 轮次 | RoPE 基线 | F0 SDAA FP32 | 变化 |
| --- | ---: | ---: | ---: |
| 0 | 2.53714 s | 2.47691 s | -2.37% |
| 1 | 2.55118 s | 2.49895 s | -2.05% |
| 2 | 2.57589 s | 2.49265 s | -3.23% |
| 3 | 2.54852 s | 2.49766 s | -2.00% |

手动复现：

```bash
/opt/kube/bin/docker exec \
  -e SDAA_VISIBLE_DEVICES=19 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  bash -lc 'source /opt/tecoai/setvars.sh && source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/miniconda3/envs/vllm_env_py310 && python tools/benchmark_cosyvoice_sdaa_candidates.py --model-dir /workspace/model --prompt-file tools/cosyvoice_perf_prompts.txt --output cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/f0_fp32_interleaved_v2/results.json --audio-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/f0_fp32_interleaved_v2/audio --variants unmasked_hift_steps8_contiguous_fused_norm_precomputed_rope unmasked_hift_steps8_contiguous_fused_norm_precomputed_rope_f0_fp32 --warmup 1 --requests 5 --rounds 4 --seed 20260729'
```

## 音频质量回归

交错 A/B 的 40 个音频使用 Qwen3-ASR-0.6B 一次性转写，前 20 个为
基线，后 20 个为候选：

| 指标 | RoPE 基线 | F0 SDAA FP32 |
| --- | ---: | ---: |
| 非空转写 | 20/20 | 20/20 |
| 字符错误数 | 16/568 | 16/568 |
| CER | 2.8169% | 2.8169% |
| 原始转写完全相同 | - | 20/20 |
| 规范化转写完全相同 | - | 20/20 |

结果文件：

- `cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/f0_fp32_interleaved_v2/asr_all.jsonl`
- `cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/f0_fp32_interleaved_v2/asr_all_summary.json`
- `cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/f0_fp32_interleaved_v2/waveform_similarity.json`

候选波形会因 F0 精度变化而改变，因此不能再用 SHA256 作为相等判据。
验证同时使用音频长度、静音比例、服务质量门禁、ASR 和人工试听目录。

## 在线 EvalScope

严格可比结果使用与第四轮相同的 seed 20260726：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/optimized_rope_f0_fp32/evalscope_c1_seed20260726`

| 指标 | 第四轮 fused norm | RoPE + F0 FP32 | 变化 |
| --- | ---: | ---: | ---: |
| 成功请求 | 10/10 | 10/10 | 持平 |
| 客户端平均时延 | 2.68096 s | 2.63570 s | -1.69% |
| 服务端平均时延 | 2676.25 ms | 2629.60 ms | -1.74% |
| 请求吞吐 | 0.37243 req/s | 0.37852 req/s | +1.64% |
| 平均 RTF | 0.41809 | 0.41081 | -1.74% |
| 生成音频总时长 | 63.84 s | 63.84 s | 持平 |
| 质量重试 | 0 | 0 | 持平 |

该组包含一次 3.70 s 异常慢请求。没有该离群请求的交错 A/B 和 Perfetto
仍显示稳定增益，因此保留完整数据，不手工删除离群值。

最终在线音频 ASR 为 4/284，CER 1.4085%，10/10 非空；错误只涉及
“太初、SDAA、CosyVoice”等专名和英文缩写，没有漏读正文。

## Perfetto

最终成功 trace：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/optimized_rope_f0_fp32/perfetto/rope_f0_fp32_v6.perfetto.json.gz`

| 指标，2 个请求 | 第四轮 fused norm | RoPE + F0 FP32 | 变化 |
| --- | ---: | ---: | ---: |
| 完整请求平均时延 | 2596.07 ms | 2459.74 ms | -5.25% |
| Flow 平均阶段时延 | 1145.79 ms | 1058.22 ms | -7.64% |
| HiFT 平均阶段时延 | 187.53 ms | 148.42 ms | -20.86% |
| HiFT D2H 次数 | 78 | 18 | -76.92% |
| HiFT D2H 总耗时 | 83.30 ms | 24.89 ms | -70.11% |
| HiFT H2D 次数 | 204 | 180 | -11.76% |
| HiFT kernel 总耗时 | 198.24 ms | 199.70 ms | +0.74% |

HiFT kernel 总耗时基本持平，但 CPU op、同步和设备拷贝明显下降，说明
收益来自消除 fallback，而不是减少模型计算量。

直接生成压缩 trace：

```bash
/opt/kube/bin/docker exec \
  -e SDAA_VISIBLE_DEVICES=19 \
  -e COSYVOICE_VLLM_SDAA_GRAPH=1 \
  -e COSYVOICE_SDAA_FLOW_FLASH_ATTN=1 \
  -e COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH=1 \
  -e COSYVOICE_SDAA_HIFT_REFLECTION_PAD=1 \
  -e COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE=1 \
  -e COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE=1 \
  -e COSYVOICE_SDAA_FLOW_STEPS=8 \
  -e COSYVOICE_SDAA_FLOW_FUSED_NORM=1 \
  -e COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE=1 \
  -e COSYVOICE_SDAA_HIFT_F0_FP32=1 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  bash -lc 'source /opt/tecoai/setvars.sh && source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/miniconda3/envs/vllm_env_py310 && python tools/cosyvoice_perfetto_profile.py --model-dir /workspace/model --prompt-file tools/cosyvoice_perf_prompts.txt --outputs-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/optimized_rope_f0_fp32/perfetto --warmup 1 --active 2 --seed 20260729 --tag rope_f0_fp32_v6'
```

## 未采用尝试

`tecocustomRotaryEmbeddingExt` 的微基准比 PyTorch 组合路径快，但算子使用
的旋转布局与 `x_transformers.apply_rotary_pos_emb` 不一致，所有
half-mode 和 sine/cosine 顺序组合仍有约 5 到 8 的最大绝对误差。
该方案已拒绝，相关 C++ 注册和 Python 调用全部删除，扩展已从干净源码
重新编译；没有保留失败 trace。

## 回归测试

```text
49 passed, 2 warnings, 9 subtests passed
```

同时通过以下文件的 `py_compile`：

- `cosyvoice/flow/DiT/modules.py`
- `cosyvoice/flow/DiT/dit.py`
- `cosyvoice/hifigan/generator.py`
- `tools/benchmark_cosyvoice_sdaa_candidates.py`
- `tools/benchmark_sdaa_flow_rope.py`
