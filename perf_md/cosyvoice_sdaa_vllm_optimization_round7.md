# CosyVoice SDAA vLLM 推理优化第七轮

## 结论

本轮从 vLLM EngineCore、采样、并行策略和 Flow 模型继续审查。最终保留一项
模型层优化：在 Flow DiT 的长序列 FeedForward 中，用 SDAA `ffnv2` 融合
`linear + GELU + linear`，短序列仍走原生 PyTorch 路径。

真实 vLLM HTTP 对称 A/B 共 20 对请求，新配置相对第六轮最终配置：

| 指标 | 第六轮最终配置 | 第七轮最终配置 | 变化 |
| --- | ---: | ---: | ---: |
| 客户端平均时延 | 2.61885 s | 2.43005 s | -7.21% |
| 客户端 p50 | 2.49600 s | 2.33871 s | -6.30% |
| 客户端 p95 | 3.68569 s | 3.11540 s | -15.47% |
| 服务端平均推理时延 | 2612.73 ms | 2421.26 ms | -7.33% |
| 请求吞吐 | 0.38121 req/s | 0.41046 req/s | +7.67% |
| 平均 RTF | 0.40815 | 0.37903 | -7.13% |
| 音频时长一致 | - | 20/20 | 持平 |
| 质量重试 | 0 | 0 | 持平 |

配对客户端变化均值为 -6.80%，95% 置信区间为
`[-8.23%, -5.36%]`，20/20 对请求更快。测试采用
`old,new,new,old` 的对称顺序；两轮旧配置平均时延分别为 2.61936 s 和
2.61833 s，两轮新配置分别为 2.42743 s 和 2.43267 s，未观察到顺序漂移
导致的假收益。

结果文件：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260803/flow_fused_ffn_e2e_final_ab/results.json`

## 为什么优化 Flow FFN

第六轮 Perfetto 显示 Flow 是端到端最大的非 LLM 阶段。旧路径在每个 DiT
block 执行两个独立线性层和 GELU，中间激活需要写回显存，再被第二个 GEMM
读取。SDAA LMK 已提供 `torch.ops._C_sdaa.ffnv2`，可以使用为该算子预排布的
FP16 权重，把两个 GEMM 和激活的调度合并。

实现位于 `cosyvoice/flow/DiT/modules.py`：

- 只在 SDAA、FP16、eval/inference 模式且显式开启环境变量时生效；
- 为融合算子注册非持久化的预排布 FP16 权重 buffer；
- 只对矩阵行数不少于 800 的 shape 使用融合算子；
- 不满足 shape、dtype、设备或 inference 保护条件时，完整回退到原始 PyTorch 实现；
- 权重 buffer 不写入 checkpoint，不改变原模型参数格式。

生产开关：

```text
COSYVOICE_SDAA_FLOW_FUSED_FFN=1
COSYVOICE_SDAA_FLOW_FUSED_FFN_MIN_ROWS=800
```

阈值来自真实 shape 微基准。batch 2 下 sequence 300、350 的融合路径分别快
5.9% 和 14.3%；sequence 400、450、500 分别快 62.3%、57.6%、60.2%。更短
shape 收益不稳定，因此不强制覆盖全部 FFN。

微基准脚本：`tools/benchmark_sdaa_flow_ffn.py`。

## 进程内交错验证

四轮交错、每轮 5 个请求的结果：

| 指标 | 第六轮配置 | 融合 FFN | 变化 |
| --- | ---: | ---: | ---: |
| 平均推理时延 | 2.83540 s | 2.60716 s | -8.05% |
| 请求数 | 20 | 20 | - |
| 质量重试 | 0 | 0 | 持平 |

结果：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260803/flow_fused_ffn_interleaved/results.json`

## 音频正确性

融合算子采用 FP16 内部计算，生成 token 和波形不要求逐字节一致，因此同时
检查波形、质量门禁和 ASR 语义。

真实 HTTP 20 对音频：

| 指标 | 结果 |
| --- | ---: |
| 音频长度一致 | 20/20 |
| 平均波形相关系数 | 0.799915 |
| 最低波形相关系数 | 0.616599 |
| 平均 SNR | 4.998 dB |
| 平均频谱余弦相似度 | 0.982763 |
| 最低频谱余弦相似度 | 0.962928 |
| 平均绝对误差 | 0.01893 |
| 质量重试 | 0/40 |

Qwen3-ASR 对旧配置 20 条和新配置 20 条统一转写，旧/新原始转写逐对一致
20/20，规范化转写逐对一致 20/20，两侧非空均为 20/20。

文件：

- `cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260803/flow_fused_ffn_e2e_final_ab/waveform_similarity_cycle0.json`
- `cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260803/flow_fused_ffn_e2e_final_ab/waveform_similarity_cycle1.json`
- `cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260803/flow_fused_ffn_e2e_final_ab/asr_all.jsonl`
- `cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260803/flow_fused_ffn_e2e_final_ab/asr_pair_comparison.json`

两种配置的人工试听 WAV 位于同一结果目录下各轮的 `audio` 子目录。

## Perfetto

保留的第七轮 trace：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260803/flow_fused_ffn_perfetto/flow_fused_ffn_v1.perfetto.json.gz`

该文件由 profiler 直接导出为压缩 JSON，并已通过 `gzip -t`。分析结果：

| 两请求合计或阶段均值 | 第六轮 trace | 融合 FFN trace | 变化 |
| --- | ---: | ---: | ---: |
| Flow 平均阶段 | 1131.94 ms | 1020.63 ms | -9.83% |
| Flow `linear/addmm` 调用 | 5092 | 3684 | -27.65% |
| 独立 GELU 调用 | 704 | 0 | 消除 |
| `_C_sdaa::ffnv2` | 0 | 352 | 启用 |
| 主 N512 GEMM kernel | 2518 | 1820 | -27.72% |

两份 trace 不是同一进程内 A/B，阶段数字只用于确认算子变化；端到端收益以
前述 20 对 HTTP 对称 A/B 为准。候选 trace 的分析文件：

- `flow_fused_ffn_v1.analysis.json`
- `flow_fused_ffn_v1.phases.json`

## vLLM EngineCore 审查

父 API 进程的 trace 看不到独立 EngineCore 进程。本轮新增
`tools/profile_cosyvoice_vllm_engine.py`，使用 vLLM ProfilerConfig 直接采集
EngineCore。保留的当前配置 trace：

`cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260803/vllm_engine_profile_current/vllm_engine_current_v1_rank0.1785738104669604933.pt.trace.json.gz`

两个请求共 336 个 decode step，观察到 336 次 event sync，累计约 994.9 ms；
采样路径的 softmax、multinomial 和同步仍是下一轮可研究方向。但本轮尝试的
异步调度、repetition penalty 改写和 n-gram speculative 均未同时满足速度与
音频确定性要求，因此没有修改已安装 vLLM 包。

## PCP、DCP 与其他未采用尝试

失败或无明显收益的尝试均没有保留 Perfetto trace：

| 尝试 | 结果 | 决策 |
| --- | ---: | --- |
| HiFT 跳过重复 F0 与提前移除 weight norm | -0.29%，95% CI `[-0.61%, +0.01%]` | 置信区间跨 0，删除代码 |
| vLLM full prefill graph | 实际回退为 `FULL_AND_PIECEWISE`，+0.30% | BlockAttention 不支持，删除配置 |
| vLLM graph block size 1024 | `block_attention.cc` 启动失败 | 删除配置 |
| vLLM async scheduling | 屏测 -10.15%，但仅 1/5 音频时长/hash 一致 | 正确性不合格 |
| repetition penalty 广义 scatter 路径 | +13.96%，0/5 更快 | 删除代码 |
| repetition penalty expand-only | +0.12%，1/5 更快 | 删除代码 |
| n-gram speculative decode | 聚合 -0.68%，配对 -0.18%，仅 1/3 更快且 0/3 音频一致 | 删除配置 |
| PCP2 | 第六轮已测 +123.84% | 单请求通信开销过大 |
| DCP | 当前 8 卡内无合法 TP/DCP 组合 | 不启动无效实验 |

未采用尝试只保留小型 JSON 结果用于审计，相关实验代码、环境开关和 trace
均已删除。

## 手动复现

先按 `api_server/README_SDAA.md` 的 vLLM 终端 A 命令进入容器，并停止正在
占用 8021 端口的服务。端到端对称 A/B：

```bash
cd /workspace/CosyVoice
python tools/benchmark_cosyvoice_vllm_e2e_ab.py \
  --output-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260803/flow_fused_ffn_e2e_recheck \
  --variants optimized_trusted_mask_conv_stft optimized_trusted_mask_conv_stft_fused_ffn \
  --cycles 2 \
  --number 10 \
  --warmup 2 \
  --seed 20260726 \
  --port 8021 \
  --startup-timeout 240
```

FFN 微基准：

```bash
source /opt/tecoai/setvars.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_env_py310
cd /workspace/CosyVoice
python tools/benchmark_sdaa_flow_ffn.py \
  --sequence-lengths 300,350,400,450,500 \
  --warmup 10 \
  --iterations 50
```

结果目录和 trace 都位于仓库内的 `cosyvoice_api_outputs`，未在仓库外创建
代码、音频或性能文件。
