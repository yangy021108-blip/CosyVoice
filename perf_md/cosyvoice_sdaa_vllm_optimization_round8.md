# CosyVoice SDAA vLLM 推理优化第八轮

## 结论

本轮从上一轮 Perfetto 中占时最大的 Flow `linear/addmm` 路径继续下钻，最终保留一项模型层优化：Flow Attention 的 Q/K/V/out 投影在长矩阵上使用 TECO SDAA `blas_gemm_fusion`，短矩阵仍走原始 `nn.Linear`。

真实 vLLM HTTP 端到端测试采用 `old,new,new,old` 对称顺序，两个 cycle、每次重启服务后预热 2 条并测量 10 条，共得到 20 对同文本、同 seed 请求：

| 指标 | 第七轮配置 | 融合 GEMM | 变化 |
| --- | ---: | ---: | ---: |
| 客户端平均延迟 | 2.43853 s | 2.29344 s | -5.95% |
| 客户端 p50 | 2.35211 s | 2.24856 s | -4.40% |
| 客户端 p95 | 3.12205 s | 2.60598 s | -16.53% |
| 服务端平均推理延迟 | 2430.61 ms | 2284.78 ms | -6.00% |
| 请求吞吐 | 0.40913 req/s | 0.43488 req/s | +6.29% |
| 平均 RTF | 0.38046 | 0.35847 | -5.78% |
| 音频时长一致 | - | 20/20 | 持平 |
| 质量重试 | 0 | 0 | 持平 |

配对客户端延迟变化均值为 -5.59%，95% 置信区间为 `[-7.38%, -3.80%]`，20/20 对请求均更快。配对服务端延迟变化均值为 -5.64%，95% 置信区间为 `[-7.44%, -3.84%]`，同样为 20/20 更快。两次旧配置 run 的均值分别为 2.44002 s、2.43703 s；两次新配置分别为 2.29157 s、2.29530 s，没有观察到顺序漂移造成的假收益。

正式结果：
`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260803/round8_fused_gemm_e2e_final_ab/results.json`

## 优化实现

实现位于 `cosyvoice/flow/DiT/modules.py`。每个 Attention 模块为 Q/K/V/out Linear 注册非持久 buffer。首次命中优化路径时：

1. 将原始权重转成 FP16，并按算子要求转置；
2. 用 `blas_gemm_fusion_weight` 在 CPU 上生成预排布权重，再搬到 SDAA；
3. 缓存预排布权重和 FP16 bias，后续 Flow step 直接复用；
4. 用 `blas_gemm_fusion` 完成矩阵乘，再原位加 bias。

只有同时满足以下条件才使用融合路径：

- 显式启用优化开关；
- SDAA、FP16、eval 且无梯度；
- 展平后的矩阵行数不少于 512；
- 输入和输出维度均为 32 的倍数。

训练、非 SDAA、短矩阵或不受支持的 shape 自动回退到原始 `nn.Linear`。预排布 buffer 不进入 checkpoint，不改变模型参数格式。生产开关为：

```text
COSYVOICE_SDAA_FLOW_FUSED_GEMM=1
COSYVOICE_SDAA_FLOW_FUSED_GEMM_MIN_ROWS=512
```

这会额外缓存 Attention 的 FP16 预排布权重，并带来一次性的首次请求准备开销，因此端到端测试在每次服务重启后都先执行预热请求。

## 算子微基准与进程内筛选

`tools/benchmark_sdaa_flow_gemm.py` 对真实 Flow 相关 shape 进行同步微基准：

| shape | 原始 Linear | 融合 GEMM | 变化 |
| --- | ---: | ---: | ---: |
| 600 x 1024 -> 1024 | 0.26653 ms | 0.14027 ms | -47.37% |
| 800 x 1024 -> 1024 | 0.31734 ms | 0.16370 ms | -48.42% |
| 1000 x 1024 -> 1024 | 0.34931 ms | 0.17117 ms | -51.00% |
| 600 x 1024 -> 6144 | - | - | -65.01% |
| 800 x 1024 -> 6144 | - | - | -69.25% |
| 1000 x 1024 -> 6144 | - | - | -69.98% |

1024 输出 shape 的最大绝对误差为 0.00390625，符合 FP16 算子重排后的数值差异范围。进程内两轮交错、每轮 5 条请求的筛选结果为 2.49385 s 降至 2.29697 s，下降 7.89%，10/10 条更快，质量重试均为 0。

筛选结果：
`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260803/round8_fused_gemm_interleaved/results.json`

## 音频正确性

正式 HTTP A/B 的 20 对 WAV 均为相同采样率和相同样本数：

| 指标 | 结果 |
| --- | ---: |
| 音频长度一致 | 20/20 |
| 平均波形相关系数 | 0.728803 |
| 最低波形相关系数 | 0.409723 |
| 平均 SNR | 3.368 dB |
| 平均频谱余弦相似度 | 0.980654 |
| 最低频谱余弦相似度 | 0.968000 |
| 平均绝对误差 | 0.022287 |
| 质量重试 | 0/40 |

由于投影权重从 FP32 转成 FP16 融合计算，WAV 不要求逐字节一致。额外使用 Qwen3-ASR 对旧配置 20 条和新配置 20 条统一转写。18/20 对原始和规范化转写完全一致；另外两对是两个 cycle 中同一条请求，旧配置为“SDA卡”，新配置为“SDA A卡”，原文为“SDAA 卡”，语义一致且新配置更接近原文。两侧 20/20 转写均非空。

相关文件：

- `round8_fused_gemm_e2e_final_ab/waveform_similarity_cycle0.json`
- `round8_fused_gemm_e2e_final_ab/waveform_similarity_cycle1.json`
- `round8_fused_gemm_e2e_final_ab/asr_all.jsonl`
- `round8_fused_gemm_e2e_final_ab/asr_pair_comparison.json`

人工试听 WAV 位于正式结果目录各 run 的 `audio` 子目录。

## Perfetto

保留的新配置 trace：
`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260803/round8_fused_gemm_perfetto/fused_gemm_v1.perfetto.json.gz`

该文件由 profiler 直接导出为压缩 JSON，并通过 `gzip -t`。两条请求的 Flow 阶段均值从上一轮 trace 的 1020.63 ms 降至 914.86 ms，下降 10.36%。

| Flow 阶段算子 | 第七轮 trace | 融合 GEMM trace | 变化 |
| --- | ---: | ---: | ---: |
| `aten::linear` | 3684 次 / 866.66 ms | 964 次 / 194.96 ms | 调用减少 73.83% |
| `aten::addmm` | 3684 次 / 710.41 ms | 964 次 / 148.79 ms | 调用减少 73.83% |
| SDAA addmm | 1842 次 / 333.23 ms | 482 次 / 60.47 ms | 调用减少 73.83% |
| `_C_sdaa::blas_gemm_fusion` | 0 | 1360 次 / 47.95 ms | 启用 |

两份 trace 不是同一进程内 A/B，因此这里只用来确认算子路径和阶段热点变化，端到端收益以前述 20 对 HTTP 对称 A/B 为准。分析文件为 `fused_gemm_v1.analysis.json` 和 `fused_gemm_v1.phases.json`。

优化后 Flow 仍有大量 `cat`、RoPE 相关重排和 bias add 调度，是下一轮可继续研究的模型层方向；vLLM EngineCore 的每 token 采样同步仍是独立热点，但本轮固定 top-k 元数据实验表明它不是当前端到端瓶颈。

## 未采用尝试

本轮还测试了 vLLM SDAA 采样器固定 top-k 的 CPU 元数据快路径。微基准由 0.11713 ms 降至 0.10522 ms，下降 10.17%，但每次只节省约 0.012 ms。真实 HTTP 5 对请求由 2.49816 s 增至 2.51356 s，聚合慢 0.62%，配对慢 0.69%，95% 置信区间为 `[+0.09%, +1.29%]`，仅 1/5 更快，因此删除补丁并恢复安装包源码。结果保留在：
`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260803/round8_fixed_topk_screen/results.json`

该失败尝试没有保留 Perfetto trace。

## 手动复现

先按 `api_server/README_SDAA.md` 的 vLLM 终端 A 命令进入容器并停止占用 8021 的服务。端到端对称 A/B：

```bash
cd /workspace/CosyVoice
python tools/benchmark_cosyvoice_vllm_e2e_ab.py \
  --output-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260803/round8_fused_gemm_e2e_recheck \
  --variants optimized_trusted_mask_conv_stft_fused_ffn optimized_trusted_mask_conv_stft_fused_ffn_gemm \
  --cycles 2 \
  --number 10 \
  --warmup 2 \
  --seed 20260726 \
  --port 8021 \
  --startup-timeout 240
```

融合 GEMM 微基准：

```bash
source /opt/tecoai/setvars.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_env_py310
cd /workspace/CosyVoice
python tools/benchmark_sdaa_flow_gemm.py --iterations 50
```

所有代码、报告、trace 和音频均位于 `/workspace/CosyVoice` 仓库内；失败尝试未保留压缩 trace。
