# CosyVoice SDAA Round 8 累计审计与 Round 9 选择（2026-08-05）

## 1. 执行摘要

本轮没有修改任何 CosyVoice 推理数值路径、模型代码或未跟踪 SDAA `.so`。在单一 vLLM HTTP API、并发 1、相同模型/输入/seed 下，完成了 A/B/C 三个已有配置的交错配对审计：每个配置 4 次独立服务启动、每次 5 次 warmup 加 25 次测量，共 **100 个有效样本/版本**和 **100 对/比较**。

Round 8 的累计收益真实存在：相对 A（原始 SDAA baseline），B（Round 5）端到端配对平均变化为 **-42.34%**，95% bootstrap CI **[-43.03%, -41.62%]**；C（Round 8）相对 B 为 **-11.36%**，CI **[-12.18%, -10.68%]**；C 相对 A 为 **-48.97%**，CI **[-49.46%, -48.44%]**。三组比较均为 100/100 对 C 或 B 更快，P50/P95 均没有回退。

推荐的唯一 Round 9 候选是：**将剩余的 Flow `gate_mlp * ff_output + residual` 融入现有 SDAA gated-residual/norm 路径**。本轮只给出方案和严格验证门槛，不实现。

## 2. 三个可复现定义

Git 历史中不存在可直接 checkout 的、彼此不同的 A/Round5/Round8 性能提交：可访问历史为 `732e301 -> 8b24261 -> 70e6df1`。性能代码已在 `8b242615e22faed849f23ba17ffd09c7ec2aa9c9` 集中，A/B/C 是该提交中已有的环境开关组合。因此本审计**不虚构三个 SHA**；三个定义共享同一性能代码 SHA，仅改变已有、完整记录的 flags。

| 定义 | 性能代码 SHA | 包含 | 不包含 |
|---|---|---|---|
| A `baseline_sdaa` | `8b24261` | 原始 SDAA 路径、Flow 10 steps | Flash/unmasked、HiFT 快路径、graph、fused norm、RoPE/F0、Round6-8 flags |
| B `round5_final` | `8b24261` | A 之外：Flash/unmasked、HiFT reflection/nearest/contiguous、Flow 8 steps、vLLM graph、fused norm、预计算 RoPE、F0 FP32 | trusted mask、conv-STFT、fused FFN/GEMM |
| C `round8_final` | `8b24261` | B 之外：trusted mask、conv-STFT、fused FFN(min rows=800)、fused GEMM(min rows=512) | 无额外 Round 9 改动 |

审计运行时脚本版本为 `5e894c2`；后续审计脚本修复为独立提交 `4b32518`、`6214002`、`f51816d`、`0c54a3a`、`266ff34`，均不修改推理实现。审计初始 API 的模型路径是 `/workspace/model`、模型别名 `cosyvoice3-0.5b`、容器可见设备为 `SDAA_VISIBLE_DEVICES=18`。

服务启动语义为 `python -m api_server.main`，其中设定 `COSYVOICE_LOAD_VLLM=true`、`COSYVOICE_FP16=true`、`COSYVOICE_MAX_CONCURRENCY=1`、`COSYVOICE_QUALITY_CHECK_ENABLED=true`、`COSYVOICE_QUALITY_MAX_RETRIES=2` 和对应版本的 flags。认证密钥仅从已有忽略的环境文件读取，未写入结果或提交。

## 3. 环境与设计

主机 `teco-smi` 显示 TECO-SMI 1.15.0、SDAA Driver 3.2.0、Runtime 3.2.0，卡型为 `TECO_AICARD_01`（64GB）。TTS 使用同一 Python/SDAA/vLLM 容器环境、同一 `/workspace/model` 模型、五条固定中文 prompt、同一默认 voice、同一 `seed=20260726` 和质量重试策略。

顺序为 `A-B-C-C-B-A-C-A-B-B-A-C`；不同时运行多版本 API。每一运行先等待 `/ready`，再执行 5 warmup 和 25 个串行 HTTP 请求，随后停止 API/EngineCore，才进入下一运行。请求端使用 `tools/cosyvoice_evalscope_perf.py`，服务端使用 `python -m api_server.main`；原始命令、完整 flag map 和 run order 在 `perf_results/round8_cumulative_audit/metadata.json`。

配对键是 `run_ordinal:request_index`，因而每个版本都有相同文本和 seed 的 100 对。统计同时保存端到端 client latency、服务 inference latency、RTF、queue wait、P50/P95/mean/stddev、吞吐以及 10,000 次 bootstrap CI。

## 4. HTTP 原始统计

| 版本 | n / 启动次数 | 端到端 mean ± std (s) | P50 / P95 (s) | 服务 mean (ms) | 请求吞吐 (req/s) | RTF mean |
|---|---:|---:|---:|---:|---:|---:|
| A baseline | 100 / 4 | 4.528 ± 0.442 | 4.463 / 5.311 | 4521.6 | 0.221 | 0.6962 |
| B Round 5 | 100 / 4 | 2.611 ± 0.315 | 2.520 / 3.118 | 2602.2 | 0.382 | 0.4004 |
| C Round 8 | 100 / 4 | 2.307 ± 0.221 | 2.250 / 2.645 | 2298.7 | 0.433 | 0.3540 |

累计音频秒数均为 649.6s；三个版本的质量重试总数均为 0。完整逐样本数据在 `raw_samples.jsonl`，每条记录含 input、seed、完整 speech token、token 数、WAV SHA、PCM/时长/采样率、RTF、重试次数和运行标识。

## 5. 配对结果与收益判定

| 配对（第二者相对第一者） | 配对平均变化 | 95% bootstrap CI | P50 / P95 变化 | 更快对数 | 服务端平均变化 |
|---|---:|---:|---:|---:|---:|
| A -> B | -42.34% | [-43.03%, -41.62%] | -43.54% / -41.29% | 100/100 | -42.45% |
| B -> C | -11.36% | [-12.18%, -10.68%] | -10.70% / -15.18% | 100/100 | -11.66% |
| A -> C | -48.97% | [-49.46%, -48.44%] | -49.58% / -50.21% | 100/100 | -49.16% |

三项均满足收益门槛：CI 不跨 0、P50/P95 无回退、多数方向一致、四次独立服务启动可复现。CSV 中保留每对的 client/server latency、相对差和胜负：`pairwise_results.csv`。

## 6. token、音频和语义质量

| 检查 | A -> B | B -> C | A -> C |
|---|---:|---:|---:|
| speech token 完全相同 | 100/100 | 100/100 | 100/100 |
| token 数相同、PCM 帧数/采样率/时长相同 | 100/100 | 100/100 | 100/100 |
| WAV SHA 相同 | 0/100 | 0/100 | 0/100 |
| 波形相关系数 mean / min | 0.529 / 0.257 | 0.738 / 0.411 | 0.518 / 0.234 |
| SNR mean (dB) | 0.471 | 3.827 | 0.523 |
| 频谱 cosine mean / min | 0.879 / 0.836 | 0.924 / 0.852 | 0.880 / 0.829 |
| ASR 原始文本相同 / 归一化相同 | 100/100 / 100/100 | 100/100 / 100/100 | 100/100 / 100/100 |

每版本 100 个文件均为有限 PCM；无异常静音、截断、潜在爆音/削波，平均 clipping ratio 为 0，质量重试为 0。三个版本的 ASR 相对固定文本平均 CER 都是 0.0193，且同一配对两侧转写完全相同。WAV SHA 不同是预期结果：A/B/C 的 Flow 步数和已有音频近似/融合路径不同；因此 SHA 不能替代声学和语义检查。

结论需分层理解：**token 完全一致**；音频是**声学近似一致**而非字节相同；ASR 表明三者的**语义一致**。ASR 工具输出了 Transformers tokenizer 上游兼容性警告，因此其 CER 仅作辅助质量信号，不能替代波形/频谱检查。

## 7. 当前 Round 8 阶段重定基线

在完成统一 HTTP 审计后，使用相同 C flags 直接加载引擎，5 warmup 后测量 20 个样本：

| 阶段 | mean (s) | 占端到端 2.292s |
|---|---:|---:|
| LLM（正确包围 generator 消费） | 1.285 | 56.0% |
| Flow | 0.874 | 38.1% |
| HiFT | 0.127 | 5.5% |
| 其他 API | 0.006 | 0.3% |

因此 Round 9 不应把 HiFT 当作首要端到端优化点；但 Flow 仍足够大，可接受一个局部、可回退的候选。

## 8. Perfetto / PyTorch Profiler 热点

生成了一个代表性 profile（5 warmup、2 active request、record shapes）。压缩 trace 位于忽略目录 `cosyvoice_api_outputs/perf/round8_cumulative_audit_20260805/round8_profiler/round8_current.perfetto.json.gz`；机器可读的分析和完整 Flow/HiFT 前 20 kernel 表在 `round8_perfetto_analysis.json`、`round8_trace_hotspots.json`、`kernel_summary.csv`。

Flow stage 两个 trace request 合计 1777.5ms；首要 SDAA kernels 为 flash-attention 340.3ms/175 calls、dilated Conv1D 303.7ms/36、第二个 flash-attention 208.2ms/176、GEMM 162.8ms/1359、GEMM-NT 138.2ms/463、copy-stride 102.1ms/3577。HiFT 两 request 合计 285.9ms；最大 kernel 是 31.99ms 的 GEMM（2 calls），其后为 transpose 18.72ms/212、Conv1D 18.14ms/118、transpose 18.12ms/109、broadcast-mul 15.99ms/287。

Flow 目标簇（跨嵌套 profiler 事件的累计，**不能相加为墙钟**）包括：copy/stride 1210.7ms/83961、GEMM 1075.1ms/7145、cat 929.9ms/8747、gate multiply 567.2ms/13366、attention 548.4ms/351、layer norm 340.0ms/2293、Conv1D 339.0ms/252。精确 RoPE 归因只有 6.23ms/198，不是候选；synchronization 0.74ms，也不是候选。

存在大量短 kernel，而不是一个可被单独删除的大 kernel：例如 Flow copy-stride 3577 次、broadcast-mul 2172 次、fused norm 1374 次、concat 736 次。trace 可见的 ATen CPU-op gap 为 2643.5ms/2 request（最大 1382.0ms），其中只有 129.9ms 与已记录 `kernel` 时间重叠。这个指标不证明剩余时间是真正 CPU idle（Python/设备事件可能未覆盖），但 trace 没有显示“CPU gap 期间大部分 SDAA 持续计算”的直接证据；不据此启动泛化 Python 微优化。

## 9. 已排除方向

不恢复已正式拒绝的 vLLM 空 sleep 和 Flow Euler/noise-buffer 移动；它们在先前 100 对等价审计中没有真实收益。也不重新尝试 Euler 不变量移动、`sol` 删除、单独 `rand_noise` buffer、泛化 Python/队列小优化，以及已失败 PCP/DCP 路线。Round 8 trace 也不支持把 RoPE、同步或 HiFT ISTFT 作为唯一下一步。

## 10. 唯一 Round 9 候选：剩余 gate residual 融合

**范围与链路。** `cosyvoice/flow/DiT/modules.py:920-929` 已将 attention gate/residual/norm 接入 `_gated_residual_layer_norm`，但随后仍执行独立的 `gate_mlp.unsqueeze(1) * ff_output + x`。`forward_prepared` 的末块路径在 954-955 也保留它；MMDiT 路径在 1037-1048 还保留多个同类 gate multiply。目标不是重写 attention，而是在已有 SDAA fused gated-residual/norm 机制中扩展一个**仅限现有 shape/dtype 的 gate-MLP residual**入口。

**trace 依据。** Flow 的 `teco_slave_broadcast_mul` 为 2172 calls / 50.35ms（两个 active request），gate-multiply 簇为 13366 profiler events / 567.2ms（嵌套累计）。它不是单独最大 kernel，但高频、直接映射到仍未融合的代码，且已有 gated-residual 实现可作回退基线。

**预期和风险。** 保守预期 Flow 墙钟改善 1--4%、端到端改善 **0.5--1.5%**；不把嵌套累计 567ms 误作可获得收益。风险是 FP16/F32 广播与舍入、mask/末块 shape、多分支 MMDiT、SDAA 扩展编译、临时张量生命周期和峰值显存。设计开关应默认为关闭，例如 `COSYVOICE_SDAA_FLOW_FUSED_GATE_RESIDUAL_NORM=0`，出现任何质量或形状问题可立即回退到现有表达式。

**下一轮严格 A/B。** 只实现该候选后，以当前 C 为 A、开关为 B；保持同一模型/API/seed/5 warmup/25 requests/run/4 starts/100 pairs/并发 1。收益要求为 paired CI 不跨 0、P50/P95 不回退、多数配对更快。额外要求：100/100 speech token 相同；PCM/时长/采样率无漂移；无 NaN/Inf/静音/截断/爆音/重试；波形/频谱指标不低于本轮 C 的可接受基线；300 条 ASR 的版本间文本一致。微等价失败时不留 trace。

## 11. 产物、提交与清理

必需机器数据均已保存：

- `perf_results/round8_cumulative_audit/raw_samples.jsonl`
- `perf_results/round8_cumulative_audit/summary.json`
- `perf_results/round8_cumulative_audit/pairwise_results.csv`
- `perf_results/round8_cumulative_audit/quality_results.json`
- `perf_results/round8_cumulative_audit/kernel_summary.csv`

另有 metadata、阶段计时、ASR 和 trace 分析 JSON，均为审计可追溯数据。独立审计脚本提交已产生，尚未推送任何远端。提交结果报告和数据前会再次确认：没有 API/EngineCore/compile worker 留存、8021 已关闭，并删除临时 Docker socket/本地 askpass 文件；用户已有未跟踪 `.so` 保持原样。
