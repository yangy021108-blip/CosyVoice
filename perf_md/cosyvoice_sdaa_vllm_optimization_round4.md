# CosyVoice SDAA vLLM 优化第四轮：Flow 融合归一化

> 2026-07-30 审核说明：本文的候选交错 A/B 和在线数字均有原始数据支持；
> 在线表格仍是当时的单次运行。第四轮与最终配置的同环境对称 HTTP A/B
> 见 [端到端延迟审核](cosyvoice_sdaa_vllm_e2e_latency_audit_20260730.md)。

## 结论

本轮在不减少 Flow 步数、不缩短音频、不修改采样参数的前提下，新增
TECO LMK `tecolmkFusedNorm` 自定义 SDAA 算子，并把 Flow DiT 中的门控
残差、LayerNorm 和 AdaLN scale/shift 融合执行。

最终结果：

- 4 轮交错 A/B、20 组同文本同 seed 配对全部加速，平均推理时延降低
  3.22%，平均 RTF 从 0.41559 降到 0.40217。
- 在线 EvalScope 10/10 成功；相对第三轮最终版本，服务端平均时延从
  2754.84 ms 降到 2676.25 ms，吞吐从 0.36153 提升到 0.37243 req/s。
- 同 seed A/B 的规范化 ASR 转写 10/10 完全相同，CER 均为 8/284。
- 最终在线输出的 ASR CER 为 3/284，与第三轮最终版本一致。
- Perfetto 中 Flow kernel launch 从 29,066 次降到 22,202 次，减少
  23.62%；Flow 平均阶段时延从 1231.45 ms 降到 1145.79 ms，降低
  6.96%。

因此该优化已经加入 SDAA 服务启动配置：

```bash
export COSYVOICE_SDAA_FLOW_FUSED_NORM=1
```

## 实现位置

| 文件 | 作用 |
| --- | --- |
| `cosyvoice/sdaa_ops/mm_encoder_fa_poc.cc` | 注册 `flow_fused_norm`，调用 TECO LMK fused norm |
| `cosyvoice/flow/DiT/modules.py` | 融合单个 block 内的 attention 残差与 FFN pre-norm，并准备下一 block 的调制参数 |
| `cosyvoice/flow/DiT/dit.py` | 在相邻 block 之间融合 FFN 残差与下一 block attention pre-norm |
| `tools/benchmark_sdaa_fused_norm.py` | 自定义算子与原始 PyTorch 算子序列的微基准及数值比较 |
| `tools/benchmark_cosyvoice_sdaa_candidates.py` | 同进程、交错顺序、同文本同 seed 的候选 A/B |

开关关闭、设备不是 SDAA、dtype 不是 FP16、batch 不是 API 推理固定的
CFG batch 2，或处于梯度模式时，代码自动走原始 PyTorch 路径。

DiT 深度为 22、Flow 为 8 步。每个请求共融合：

- 每个 block 内 22 次，覆盖 attention 门控残差到 FFN pre-norm；
- 相邻 block 之间 21 次，覆盖 FFN 门控残差到下一 block attention
  pre-norm；
- 总计 43 × 8 = 344 次自定义 op 调用。

## 数值与微基准

固定形状为 batch 2、序列长度 256、hidden size 512，100 次统计：

| 指标 | 结果 |
| --- | ---: |
| 原始 PyTorch 序列平均耗时 | 0.19392 ms |
| 自定义 fused norm 平均耗时 | 0.07481 ms |
| 微基准加速比 | 2.59× |
| residual 最大绝对误差 | 0 |
| normalized output 最大绝对误差 | 0.015625 |

FP16 LayerNorm 的归约顺序改变会产生上述正常舍入差异，因此又使用同 seed
音频、ASR 和人工输出文件进行端到端质量验证。

手动运行微基准：

```bash
/opt/kube/bin/docker exec \
  -e SDAA_VISIBLE_DEVICES=19 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  bash -lc 'source /opt/tecoai/setvars.sh && source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/miniconda3/envs/vllm_env_py310 && python tools/benchmark_sdaa_fused_norm.py --warmup 20 --iterations 100'
```

## 交错 A/B

结果文件：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/fused_norm_interleaved_v2/results.json`

测试包含 4 轮；偶数轮先跑基线，奇数轮先跑候选。每轮每个版本先预热
1 次，再正式执行 5 个固定文本。20 组配对结果如下：

| 指标 | 基线 | fused norm | 变化 |
| --- | ---: | ---: | ---: |
| 平均推理时延 | 2.63278 s | 2.54796 s | -3.22% |
| 平均 RTF | 0.41559 | 0.40217 | -3.23% |
| 配对中候选更快 | - | 20/20 | - |
| 音频长度一致 | - | 20/20 | - |

四轮的基线与候选平均时延分别为：

| 轮次 | 基线 | fused norm |
| --- | ---: | ---: |
| 0 | 2.64418 s | 2.54195 s |
| 1 | 2.62892 s | 2.56547 s |
| 2 | 2.63431 s | 2.54711 s |
| 3 | 2.62372 s | 2.53732 s |

手动复现：

```bash
/opt/kube/bin/docker exec \
  -e SDAA_VISIBLE_DEVICES=19 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  bash -lc 'source /opt/tecoai/setvars.sh && source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/miniconda3/envs/vllm_env_py310 && python tools/benchmark_cosyvoice_sdaa_candidates.py --model-dir /workspace/model --prompt-file tools/cosyvoice_perf_prompts.txt --output cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/fused_norm_interleaved_v2/results.json --audio-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/fused_norm_interleaved_v2/audio --variants unmasked_hift_steps8_contiguous unmasked_hift_steps8_contiguous_fused_norm --warmup 1 --requests 5 --rounds 4 --seed 20260729'
```

## ASR 质量回归

同 seed 基线和候选各取 10 个音频：

| 指标 | 基线 | fused norm |
| --- | ---: | ---: |
| 非空转写 | 10/10 | 10/10 |
| 字符错误数 | 8/284 | 8/284 |
| CER | 2.8169% | 2.8169% |
| 规范化转写完全相同 | - | 10/10 |

最终在线 EvalScope 音频再次运行 ASR，得到 3/284、CER 1.0563%，与第三轮
最终结果一致。

## 在线 EvalScope

结果目录：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/optimized_fused_norm/evalscope_c1`

| 指标 | 第三轮最终版本 | fused norm | 变化 |
| --- | ---: | ---: | ---: |
| 成功请求 | 10/10 | 10/10 | 持平 |
| 客户端平均时延 | 2.76117 s | 2.68096 s | -2.90% |
| 服务端平均时延 | 2754.84 ms | 2676.25 ms | -2.85% |
| 请求吞吐 | 0.36153 req/s | 0.37243 req/s | +3.01% |
| 平均 RTF | 0.43059 | 0.41809 | -2.90% |
| 生成音频总时长 | 63.84 s | 63.84 s | 持平 |

手动复现：

```bash
/opt/kube/bin/docker exec \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  /root/miniconda3/envs/vllm_env_py310/bin/python \
  tools/cosyvoice_evalscope_perf.py \
    --prompt-file tools/cosyvoice_perf_prompts.txt \
    --outputs-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/optimized_fused_norm/evalscope_c1 \
    --api-key-env-file cosyvoice_sdaa_vllm_api.env \
    --number 10 \
    --parallel 1 \
    --warmup 2 \
    --seed 20260726 \
    --run-name cosyvoice-sdaa-fused-flow-norm
```

## Perfetto

成功候选 trace：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/optimized_fused_norm/perfetto/fused_norm_v5.perfetto.json.gz`

| Flow 指标，2 请求 | 第三轮最终版本 | fused norm | 变化 |
| --- | ---: | ---: | ---: |
| Flow 平均阶段时延 | 1231.45 ms | 1145.79 ms | -6.96% |
| 完整请求平均时延 | 2679.06 ms | 2596.07 ms | -3.10% |
| `sdaaLaunchKernel` 次数 | 29,066 | 22,202 | -23.62% |
| FlashAttention kernel | 551.03 ms | 547.37 ms | -0.66% |
| copy-stride kernel | 115.81 ms | 107.55 ms | -7.13% |
| `_C_sdaa_poc::flow_fused_norm` | 0 | 688 次 | 新增 |

`tecolmkFusedNorm` 以 CFG batch 的两个分支分别发起内核，因此 trace 中记录
1372 次 fused norm kernel、总计 48.34 ms。虽然单个融合 kernel 比单个
elementwise kernel 更长，但它替代了多组 add、mul、LayerNorm、copy 和
launch，最终 Flow 阶段净减少约 85.65 ms。

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
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  bash tools/run_cosyvoice_perfetto_sdaa.sh \
    --prompt-file tools/cosyvoice_perf_prompts.txt \
    --outputs-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260729/optimized_fused_norm/perfetto \
    --warmup 1 \
    --active 2 \
    --seed 20260726 \
    --tag fused_norm_v5
```

## 未采用尝试

未形成收益的尝试没有保留 trace，候选音频也已删除，只保留小型 JSON 结果
作为结论依据。

| 尝试 | 结果 | 结论 |
| --- | ---: | --- |
| checkpoint 后移除 HiFT weight norm | -0.17% | 波动范围内，拒绝 |
| F0 predictor 改为 SDAA FP32 | 约 -1.69% | 改变关键数值路径，未纳入本轮 |
| HiFT 静态 buffer | +0.07% | 无收益，拒绝 |
| Flow QKV 合并为单个 `F.linear` | +0.11% | SDAA 上略慢，拒绝 |

失败尝试对应的环境开关和代码已经移除，避免后续误启用。
