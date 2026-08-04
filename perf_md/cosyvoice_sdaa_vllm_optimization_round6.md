# CosyVoice SDAA vLLM 推理优化第六轮

## 结论

本轮从 vLLM、PCP/DCP、Flow 模型和 HiFT Fourier 路径继续探索。最终保留
两项单卡优化：

1. 跳过 API 非流式单样本场景中已知全有效 mask 的 SDAA scalar 同步；
2. 用固定 Hann Fourier basis 的 SDAA 卷积实现 HiFT 的
   `n_fft=16, hop=4` STFT/ISTFT。

真实 vLLM HTTP 对称 A/B 共 20 对请求，新配置相对第五轮最终配置：

| 指标 | 第五轮最终配置 | 第六轮最终配置 | 变化 |
| --- | ---: | ---: | ---: |
| 客户端平均时延 | 2.63883 s | 2.62138 s | -0.66% |
| 客户端 p50 | 2.50855 s | 2.47878 s | -1.19% |
| 客户端 p95 | 3.70199 s | 3.69544 s | -0.18% |
| 服务端平均推理时延 | 2630.21 ms | 2611.82 ms | -0.70% |
| 请求吞吐 | 0.37795 req/s | 0.38062 req/s | +0.71% |
| 平均 RTF | 0.41092 | 0.40804 | -0.70% |
| 生成音频总时长 | 127.68 s | 127.68 s | 持平 |
| 质量重试 | 0 | 0 | 持平 |

配对客户端变化均值为 -0.67%，95% 置信区间为
`[-0.92%, -0.41%]`，18/20 对请求更快。两轮反向运行中候选平均时延分别
为 2.61808 s 和 2.62468 s，均低于两轮旧配置的 2.63080 s 和
2.64685 s。

结果文件：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260802/e2e_final_ab/results.json`

## 已有 trace 审查

第五轮最终 trace 的主要结论：

- Flow 两请求有 19,418 次 kernel launch；
- Flow 的 `cat/copy_stride` 开销仍高，但主要来自 partial RoPE 和布局转换；
- HiFT 仍有 18 次 D2H、180 次 H2D，并执行固定小尺寸 STFT/ISTFT；
- vLLM EngineCore 位于独立进程，父进程 trace 只能看到 LLM 墙钟段。

第五轮 trace：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/optimized_rope_f0_fp32/perfetto/rope_f0_fp32_v6.perfetto.json.gz`

## 保留优化一：全有效 mask 快路径

API 的非流式单请求在 Flow 中复制为 batch 2 的 classifier-free guidance
输入，mask 由唯一有效长度构造，全部为真。旧快路径仍用
`attn_mask.all().item()` 在每个请求验证，最终 trace 中两个请求产生 4 次
`sdaa::_local_scalar_dense`，累计 CPU 等待 38.47 ms。

新开关：

```text
COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK=1
```

保护条件仍要求 SDAA、batch 2、非流式且 inference mode；其他场景不使用
该捷径。四轮进程内交错 A/B：

| 指标 | 原最终配置 | trusted mask | 变化 |
| --- | ---: | ---: | ---: |
| 平均推理时延 | 2.73423 s | 2.72619 s | -0.29% |
| 候选更快 | - | 14/20 | - |
| WAV SHA256 一致 | - | 20/20 | - |
| 质量重试 | 0 | 0 | 持平 |

结果：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260802/trusted_mask_interleaved/results.json`

## 保留优化二：HiFT 卷积 STFT/ISTFT

HiFT 的 Fourier 参数固定为 `n_fft=16`、`hop=4`、periodic Hann window。
新路径预计算实部/虚部 Fourier basis：

- STFT 使用一次 18-channel `conv1d`；
- ISTFT 使用 `conv_transpose1d` overlap-add，并除以 Hann window-square
  envelope；
- 只在 SDAA inference mode 且显式启用开关时生效；
- 原始 `torch.stft/istft` 保留为默认回退。

150,000 samples 微基准：

| 指标 | 原生 Torch | 固定卷积 | 变化 |
| --- | ---: | ---: | ---: |
| STFT | 3.72803 ms | 0.18733 ms | -94.98% |
| ISTFT | 6.37982 ms | 1.24319 ms | -80.51% |
| STFT 最大绝对误差 | - | 0.0007613 | - |
| ISTFT 最大绝对误差 | - | 0.0007507 | - |

四轮进程内交错 A/B 在 trusted-mask 基础上继续比较：

| 指标 | trusted mask | 加卷积 Fourier | 变化 |
| --- | ---: | ---: | ---: |
| 平均推理时延 | 2.73406 s | 2.72590 s | -0.30% |
| 配对变化均值 | - | -0.32% | 95% CI `[-0.63%, -0.01%]` |
| 候选更快 | - | 14/20 | - |
| 音频长度一致 | - | 20/20 | - |
| 质量重试 | 0 | 0 | 持平 |

结果：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260802/conv_stft_interleaved/results.json`

## 音频正确性

20 对 A/B 音频的波形统计：

| 指标 | 结果 |
| --- | ---: |
| 平均波形相关系数 | 0.999545 |
| 平均 SNR | 31.03 dB |
| 平均频谱余弦相似度 | 0.996966 |
| 平均绝对误差 | 0.000664 |
| 音频长度差 | 20/20 为 0 |

Qwen3-ASR-0.6B 对基线 20 条和候选 20 条统一转写：

| 指标 | 基线 | 候选 |
| --- | ---: | ---: |
| 非空转写 | 20/20 | 20/20 |
| 字符错误 | 4/568 | 4/568 |
| CER | 0.7042% | 0.7042% |
| 原始转写逐对一致 | - | 20/20 |
| 规范化转写逐对一致 | - | 20/20 |

文件：

- `cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260802/conv_stft_interleaved/waveform_similarity.json`
- `cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260802/conv_stft_interleaved/asr_all.jsonl`
- `cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260802/conv_stft_interleaved/asr_all_summary.json`

## Perfetto

成功 trace：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260802/optimized_trusted_mask_conv_stft/perfetto/trusted_mask_conv_stft_v1.perfetto.json.gz`

| 两请求合计或阶段均值 | 第五轮 trace | 第六轮 trace | 变化 |
| --- | ---: | ---: | ---: |
| HiFT 平均阶段 | 148.42 ms | 124.47 ms | -16.13% |
| HiFT D2H 次数 | 18 | 4 | -77.78% |
| HiFT D2H 总耗时 | 24.89 ms | 1.28 ms | -94.86% |
| HiFT H2D 次数 | 180 | 166 | -7.78% |
| HiFT H2D 总耗时 | 6.94 ms | 3.96 ms | -42.99% |
| `aten::stft/istft` | 4/2 次 | 0/0 次 | 消除 |
| `_fft_r2c/_fft_c2r` | 2/2 次 | 0/0 次 | 消除 |
| Flow SDAA scalar | 4 次，38.47 ms | 0 | 消除 |

第六轮 trace 的完整请求均值为 2515.12 ms，高于第五轮的 2459.74 ms，
原因是该次采样中的 Flow 阶段由 1058.22 ms 波动到 1131.94 ms。它不是
同一服务进程内的对称 A/B，不能用来判断端到端回退；端到端结论采用上面的
20 对交错 HTTP 结果。Perfetto 用于确认目标算子和设备拷贝确实被消除。

## PCP/DCP 审查

导出的 vLLM 模型为 Qwen2ForCausalLM，24 层、hidden size 896、14 个 Q
heads、2 个 KV heads。

PCP2 使用逻辑卡 18、19 成功启动两个 TCCL worker，日志确认 PCP rank 0/1。
但两个真实请求的平均时延由 2.54157 s 增至 5.68916 s（+123.84%），
0/2 更快，吞吐下降 55.20%；固定 seed 的音频时长也不一致。短 prompt
prefill 不足以抵消多进程和通信开销，因此拒绝且不扩大测试。

DCP 在当前最多 8 卡范围内没有合法组合：

- `TP=2,DCP=2`：TP 必须大于 2 个 KV heads；
- `TP=7,DCP=2/3`：TP 必须能被 DCP 整除；
- 14 个 Q heads 又要求 TP 整除 14。

PCP 结果：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260802/pcp2_screen/results.json`

## 未采用尝试

这些候选没有生成 Perfetto trace：

| 尝试 | 结果 | 决策 |
| --- | ---: | --- |
| vLLM 每 token 只在 idle 时 sleep | +0.05%，5/10 更快 | 波动范围，删除代码 |
| vLLM `RequestOutputKind.DELTA` | +0.05%，9/20 更快 | 无收益，删除代码 |
| RoPE 前 64 维原位写回 | 局部 0.515→1.009 ms | stride copy 使其变慢 |
| Flow QKV 融合 | 端到端 +0.69%，2/20 更快 | 非连续 V 布局，删除代码 |
| Flow QK 融合 | 端到端 +0.90%，0/20 更快 | 非连续 Q/K 中间布局，删除代码 |
| PCP2 | 端到端 +123.84%，0/2 更快 | 通信开销和确定性均不合格 |
| DCP | 8 卡内无合法配置 | 不启动无效实验 |

失败/无收益尝试没有保留 trace。对应的小型 JSON 结果保留，用于防止后续重复
探索。

## 手动复现

进程内交错 A/B：

```bash
/opt/kube/bin/docker exec \
  -e SDAA_VISIBLE_DEVICES=19 \
  -e COSYVOICE_VLLM_SDAA_GRAPH=1 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  bash -lc 'source /opt/tecoai/setvars.sh && source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/miniconda3/envs/vllm_env_py310 && python tools/benchmark_cosyvoice_sdaa_candidates.py --model-dir /workspace/model --prompt-file tools/cosyvoice_perf_prompts.txt --output cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260802/conv_stft_recheck/results.json --audio-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260802/conv_stft_recheck/audio --variants optimized_trusted_mask optimized_trusted_mask_conv_stft --warmup 1 --requests 5 --rounds 4 --seed 20260726'
```

真实 HTTP A/B 在已经运行 vLLM 服务的容器内执行：

```bash
cd /workspace/CosyVoice
python tools/benchmark_cosyvoice_vllm_e2e_ab.py \
  --output-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260802/e2e_final_recheck \
  --variants optimized optimized_trusted_mask_conv_stft \
  --cycles 2 \
  --number 10 \
  --warmup 2 \
  --seed 20260726 \
  --startup-timeout 240
```

直接生成压缩 Perfetto trace：

```bash
/opt/kube/bin/docker exec \
  -e SDAA_VISIBLE_DEVICES=19 \
  -e COSYVOICE_VLLM_SDAA_GRAPH=1 \
  -e COSYVOICE_SDAA_FLOW_FLASH_ATTN=1 \
  -e COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH=1 \
  -e COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK=1 \
  -e COSYVOICE_SDAA_HIFT_REFLECTION_PAD=1 \
  -e COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE=1 \
  -e COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE=1 \
  -e COSYVOICE_SDAA_FLOW_STEPS=8 \
  -e COSYVOICE_SDAA_FLOW_FUSED_NORM=1 \
  -e COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE=1 \
  -e COSYVOICE_SDAA_HIFT_F0_FP32=1 \
  -e COSYVOICE_SDAA_HIFT_CONV_STFT=1 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  bash -lc 'source /opt/tecoai/setvars.sh && source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/miniconda3/envs/vllm_env_py310 && python tools/cosyvoice_perfetto_profile.py --model-dir /workspace/model --prompt-file tools/cosyvoice_perf_prompts.txt --outputs-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260802/optimized_trusted_mask_conv_stft/perfetto_recheck --warmup 1 --active 2 --seed 20260726 --tag trusted_mask_conv_stft_recheck'
```

无明显收益的候选不要生成 trace；若临时生成，应删除对应 `.json.gz`。
