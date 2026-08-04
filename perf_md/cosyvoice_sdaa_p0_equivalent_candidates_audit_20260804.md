# CosyVoice SDAA P0 等价候选端到端审计（2026-08-04）

## 结论

本轮单独审计了两个低风险、数值等价候选：

1. vLLM token 循环只在无进展时 sleep；
2. Flow Euler 不变量移出循环、删除 `sol`，并将 `rand_noise` 注册为非持久 buffer。

两项候选都通过了严格的输出等价检查，但都没有通过真实 HTTP 性能门槛，因此候选推理代码均已回退，不进入最终配置，也没有采集或保留 Perfetto trace。

| 候选 | 100 对平均 latency 变化 | 配对 95% CI | 更快请求 | P50/P95 | 决策 |
| --- | ---: | ---: | ---: | --- | --- |
| 仅无进展时 sleep | +0.11%（变慢） | [-0.05%, +0.31%] | 42/100 | P50 +0.08%，P95 -0.07% | 回退 |
| Flow Euler 不变量与 noise buffer | -0.16% | [-0.36%, +0.03%] | 55/100 | P50 +0.26%，P95 +0.13% | 回退 |

这里没有把“减少了调用次数”或“均值有极小下降”当成优化：两项结果的置信区间都跨 0，并且没有同时改善平均值、配对结果和 P95。

## 统一测试方法

- 分支：`feature/cosyvoice-api-service`
- 审计基点：`8b242615e22faed849f23ba17ffd09c7ec2aa9c9`
- 设备：SDAA，容器可见设备 `SDAA_VISIBLE_DEVICES=18`
- API：真实 `/v1/audio/speech`，vLLM backend，并行度 1
- 文本：`tools/cosyvoice_perf_prompts.txt` 中固定 5 条文本循环
- seed：`20260726`，同一 pair 使用相同文本和 seed
- 顺序：`old -> new -> new -> old`
- 独立启动：每个候选 4 次服务启动
- 每次启动：2 条 warmup 后测量 50 条请求
- 总样本：每个 variant 100 条，共 100 对
- 指标：客户端 latency、服务端 inference latency、P50、P95、吞吐、RTF、配对相对变化、95% CI、更快请求数
- 正确性：speech token、WAV SHA256、音频长度和质量重试次数逐对检查

正式证据：

- `cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260804/sleep_formal_100_pairs/results.json`
- `cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260804/flow_formal_100_pairs/results.json`
- `cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260804/flow_equivalence/round8_current_to_round8_flow_euler_invariants.json`

## 候选一：vLLM 仅无进展时 sleep

### 临时改动

位置：`cosyvoice/llm/llm.py::Qwen2LM.inference_wrapper()`。

临时实现包括：

- 只有 `self.vllm.step()` 没有新增输出、并且请求输出队列也没有 token 可取时，才 sleep 100 微秒；
- 根据 cumulative `RequestOutput.outputs[0].token_ids` 的已消费长度，将完整新增 suffix 放入队列；
- 每请求记录 `decode_steps`、`enqueued_tokens`、`sleep_calls` 和 `requested_sleep_ms`；
- baseline 保留原始每 token 固定 sleep 1 ms 的路径。

### 控制路径效果

| 指标（每请求平均） | 当前实现 | 候选 |
| --- | ---: | ---: |
| decode steps | 170.76 | 170.76 |
| sleep calls | 169.76 | 1.00 |
| requested sleep | 169.76 ms | 0.10 ms |

候选确实消除了 99.4% 的 sleep 调用和几乎全部名义 sleep 时间，统计不是未生效的开关。

### HTTP 结果

| 指标 | 当前实现 | 候选 | 候选变化 |
| --- | ---: | ---: | ---: |
| 请求数 | 100 | 100 | - |
| 平均 latency | 2.257113 s | 2.259679 s | +0.11% |
| P50 | 2.201555 s | 2.203294 s | +0.08% |
| P95 | 2.636459 s | 2.634501 s | -0.07% |
| 吞吐 | 0.442309 req/s | 0.441738 req/s | -0.13% |
| 平均 RTF | 0.351282 | 0.351748 | +0.13% |

配对 latency 均值变化为 +0.13%，95% CI 为 [-0.05%, +0.31%]，候选只有 42/100 对更快。服务端 inference latency 也变慢 +0.13%。

这说明原来的 `time.sleep(0.001)` 并不是可直接累加到 wall time 的 169.76 ms；它与 EngineCore 工作、同步或主机调度重叠。减少控制路径等待是真实的，但没有转化为端到端收益。

### 正确性与决策

- speech token：100/100 完全一致；
- WAV SHA256：100/100 完全一致；
- 音频长度：100/100 完全一致；
- 两边质量重试总数：0。

决策：回退。该候选没有显著 HTTP 收益，不生成 Perfetto。

## 候选二：Flow Euler 不变量与 rand_noise buffer

### 临时改动

位置：`cosyvoice/flow/flow_matching.py`。

临时实现包括：

- `mask_in`、`mu_in`、`spks_in` 和 `cond_in` 在 Euler 循环外只初始化一次；
- 循环内只更新随 step 变化的 `x_in` 和 `t_in`；
- 删除只用于返回最后元素的 `sol` 列表，直接返回最终 `x.float()`；
- 将 `CausalConditionalCFM.rand_noise` 注册为 `persistent=False` buffer，保留请求时 dtype 转换。

没有同时引入原位 Euler、长度 buffer 池或新的近似融合。

### 真实模型等价性

使用独立进程分别加载 old/new，在固定文本与 seed 下截取送入 HiFT 的真实 Flow mel：

- mel shape：两边均为 `[1, 80, 198]`；
- mel dtype：两边均为 `torch.float32`；
- mel SHA256：两边均为 `12eb2c37f365d401aecb1586381db5a2ff999293ffaf8548b3c10b7e083f1ddf`；
- `torch.equal`：true；
- `max_abs_error`：0.0；
- WAV SHA256：两边均为 `88ab42cda2a1f2bb480208e138879b66f71858093a0cb1c034c5a21fc03912fd`。

buffer 迁移也实际生效：当前实现的 `rand_noise` 位于 CPU 且不是 registered buffer；候选位于 `sdaa:0` 且是 non-persistent registered buffer。

### HTTP 结果

| 指标 | 当前实现 | 候选 | 候选变化 |
| --- | ---: | ---: | ---: |
| 请求数 | 100 | 100 | - |
| 平均 latency | 2.258309 s | 2.254623 s | -0.16% |
| P50 | 2.187793 s | 2.193391 s | +0.26% |
| P95 | 2.652739 s | 2.656300 s | +0.13% |
| 吞吐 | 0.441999 req/s | 0.442705 req/s | +0.16% |
| 平均 RTF | 0.351496 | 0.350937 | -0.16% |

配对 latency 均值变化为 -0.16%，95% CI 为 [-0.36%, +0.03%]，候选 55/100 对更快；服务端 inference latency 为 -0.16%，其 CI 同样跨 0。P50 和 P95 都略微变慢。

被移出的 copy 和单次 noise H2D 都是真实存在的，但相对约 2.25 秒的完整 LLM + Flow + HiFT 路径过小，无法在端到端层面确认收益。

### 正确性与决策

- speech token：100/100 完全一致；
- WAV SHA256：100/100 完全一致；
- 音频长度：100/100 完全一致；
- 两边质量重试总数：0；
- 独立 Flow mel：bitwise 一致，`max_abs_error=0.0`。

决策：回退。平均值的 -0.16% 不显著，且 P50/P95 变慢，不生成 Perfetto。

## 测试框架修正

正式审计中发现 `tools/benchmark_cosyvoice_vllm_e2e_ab.py` 原先只匹配 cmdline 中的 `python -m api_server.main`。当解释器实际显示为 `python3 -m api_server.main` 时，会在 `/ready` 正常的情况下误报没有服务。

保留的唯一代码修正是按相邻参数 `-m`、`api_server.main` 查找进程，不再依赖解释器 basename。该修正只影响 benchmark 服务发现，不修改模型、API 或推理计时路径。

## 后续方向

本轮结果否定了两个“看起来有理论收益、但端到端占比过小”的方向。下一轮不应把它们与其他改动打包重测。更值得继续的方向是：

1. 单独治理 packed weight 和固定 buffer 的冷启动准备，并分别报告 `/ready`、首请求、第二请求、稳态及显存/CPU 内存；
2. 先以 trace 确认跨 DiT block 重复 SiLU/时间调制的实际占比，再做严格等价复用；
3. 只有在新 trace 仍显示 gate 广播乘法或 QKV/RoPE/layout 是显著热点时，才进入中高风险自定义融合。

由于本轮没有显著 HTTP 优化，失败候选的 Perfetto trace 数量为 0，无需删除 trace 文件。
