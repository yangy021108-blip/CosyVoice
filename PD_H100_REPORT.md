> Legacy vLLM 0.11.0 report. See PD_VLLM0251_REPORT.md for the vLLM 0.25.1 follow-up and CUDA Graph diagnosis.

# CosyVoice3 vLLM Prefill-Decode Disaggregation H100 报告

## 结论先行

1P1D PoC 已证明正确：Prefill/Decode 确实运行在不同进程和不同 H100，
1.69 MiB prompt KV 通过 NIXL/UCX GPU READ 传输，decoder 复用 130 个
external cached tokens；固定 canonical prompt 下 raw speech tokens 为
162/162、filtered tokens 为 154/154，WAV 与 Unified baseline 字节完全
一致。

但当前版本不建议进入生产集成。原因不是 KV 传输慢，而是 vLLM 0.11 的
远端 KV 与现有 `FULL_DECODE_ONLY` CUDA Graph 组合会把输出从 162 token
稳定改变为 164 token。正确的 PD 只能退回 eager，C=16 吞吐 14.42 req/s，
明显低于正确 Unified graph 的 36.16 req/s。默认生产代码和 API 路径因此
保持不变，也没有增加 feature flag。

## 阶段状态

| 阶段 | 状态 | 结论 |
|---|---|---|
| 源码/环境调查 | 完成 | 以服务器实际 vLLM 0.11.0/V1 源码为准 |
| Unified baseline | 完成 | 可复现，162 raw tokens，6.16 s WAV |
| NIXL 1P1D eager PoC | 完成 | token 和 audio correctness 通过 |
| 独立 KV 计时 | 完成 | 热态 1.25–1.42 ms（C=1 多轮） |
| C=1/2/4/8/16 | 完成 | 126 个矩阵请求全部通过；另有 1008 请求压力测试 |
| CUDA Graph | 失败并停止 | 164 vs 162 token，不保留为有效优化 |
| 干扰实验 | 部分完成 | Unified spike 已测；PD 同 P 注入未完成 |
| 生产 feature flag | 未集成 | 数据不支持生产化 |

## Environment

| 项目 | 实际值 |
|---|---|
| GPU | 8 × NVIDIA H100 80GB HBM3 |
| Driver | 580.65.06 |
| CUDA runtime | 12.8（PyTorch 2.8.0+cu128）；系统无 `nvcc` |
| Python | 3.10.20 |
| vLLM | 0.11.0，V1 engine |
| Transformers | 4.57.1 |
| NIXL | 0.6.1，独立 system-site-packages venv |
| UCX | 1.18.0 |
| GPU topology | 任意 GPU pair 均为 NV18，CUDA P2P 可用 |

GPU1 被现有 CosyVoice API 占用，GPU3 被其他用户占用；实验使用 GPU0
Prefill、GPU2 Decode、GPU4 Acoustic。所有 GPU pair 都是同一 NVSwitch
级别，因此本机没有“较差 PCIe-only pair”可用于拓扑对照。

## Architecture

Unified：

```text
prompt_embeds -> H100: Prefill + autoregressive Decode -> speech tokens
                                                      -> Flow -> HiFT -> WAV
```

1P1D PoC：

```text
Coordinator
  | canonical prompt_embeds
  +-> process P / GPU0: prompt prefill, retain KV blocks
  |                         |
  |                         +-- NIXL UCX GPU READ (no CPU KV file)
  |                                           |
  +-> process D / GPU2: external KV -> autoregressive decode -> speech tokens
  |
  +-> GPU4: unchanged Flow -> unchanged HiFT -> WAV
```

Producer 的 `max_tokens=1` 只用于让 vLLM materialize 并保留 prompt KV；
该辅助 token 被丢弃，最终 speech tokens 全部由 decoder 重新生成。这与
vLLM 0.11 官方 disaggregated proxy 的处理一致。

vLLM 0.11 `NixlConnectorScheduler` 原实现对 prompt embeddings 调用
`len(request.prompt_token_ids)`，会触发 `len(None)`。实验 connector 只将
长度来源改为 vLLM 已计算的 `request.num_prompt_tokens`，没有修改
site-packages 或其他 scheduler/KV 逻辑。

## Correctness

固定输入不是“相同文本重新跑 frontend”，而是同一个序列化
`prompt_embeds` tensor、spans 和 sampling metadata。原因是实测 wetext
网络状态可把 `CosyVoice3` 归一化成不同形式，参考音频重新编码也出现过
1 个 prompt speech token 漂移。

| 检查 | Unified | PD | 结果 |
|---|---:|---:|---|
| Prompt embeds shape | `[131, 896]` BF16 | `[131, 896]` BF16 | 相同 tensor |
| Decoder external cached tokens | 0 | 130 | 证明未重做完整 prefill |
| Raw speech tokens | 162 | 162 | 逐 token 100% 相同 |
| Filtered speech tokens | 154 | 154 | 逐 token 100% 相同 |
| Stop token | 6562 | 6562 | 相同，进入 Flow 前删除 |
| WAV SHA256 | `b392da60…50be6` | `b392da60…50be6` | 完全相同 |

### Audio Gate

| 指标 | Unified | PD |
|---|---:|---:|
| Sample rate / channels / width | 24 kHz / mono / PCM16 | 相同 |
| Duration | 6.16 s | 6.16 s |
| RMS | 0.09473254 | 0.09473254 |
| Peak | 0.85202026 | 0.85202026 |
| Clipping ratio | 0 | 0 |
| Silent-frame ratio | 0.211039 | 0.211039 |
| Pairwise speaker similarity | 1.0 | 字节相同，因此无回归 |
| Pairwise CER/WER regression | 0 | 字节相同，因此 ASR 输入无变化 |

没有重新运行绝对 ASR CER/WER；这里回答的是 PD 相对 Unified 是否回归。
相同 WAV 对任何确定性 ASR/说话人模型都会产生相同结果。

## Latency Breakdown

### C=1 稳态

| 阶段 | Unified standard | 正确 PD eager | 说明 |
|---|---:|---:|---|
| Frontend | 40.35 ms | 固定 payload，profiling 时为 0 | correctness 路径未修改 frontend |
| Prefill compute | 包含在 TTFT | 16.05 ms | P worker 热态 |
| KV prepare | — | 约 0.05–0.18 ms | descriptor preparation |
| KV transfer | — | 1.42 ms | 1,769,472 bytes，submit 到首次 DONE |
| KV estimated bandwidth | — | 1.25 GB/s | 小 payload 下包含 connector/轮询固定开销 |
| TTFT speech token | 34.26 ms（原 baseline） | 15.53 ms | PD 值从 D admission 起计 |
| LLM total | 373.46 ms | 1004.13 ms | 162-token trajectory |
| TPOT P50 / P95 / P99 | 2.04 / 2.30 / 2.33 ms | 6.08 / 6.24 / 6.40 ms | PD eager 明显更慢 |
| Flow | 477.22 ms | 319.59 ms | 单次波动；算法/seed 未改 |
| HiFT | 214.48 ms | 243.72 ms | 单次波动；算法未改 |

首次 NIXL agent handshake 约 11–12 秒，不能计入稳态请求。冷态第一轮中
前端会在 EngineCore 已累计输出后快速取回 token，因此冷态 TPOT 时间戳被
标记为无效；报告只使用第 2 轮稳态数据。

## Concurrency

所有 PD 档均为两轮：第 0 轮建立 connector，第 1 轮统计。C=1/2/4/8/16
合计 62 个并发矩阵请求，连同前序 correctness/profile 请求共 126 个；全部
保持 162-token canonical trajectory。

| C | PD req/s | PD audio-s/s | PD E2E P95 | PD TTFT P95 | PD TPOT P95 | KV P95 | Unified req/s | Unified audio-s/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 6.13 | 1.004 s | 15.53 ms | 6.24 ms | 1.42 ms | 2.84 | 17.50 |
| 2 | 1.98 | 12.20 | 1.010 s | 18.59 ms | 6.20 ms | 7.07 ms | 5.42 | 33.38 |
| 4 | 3.92 | 24.13 | 1.021 s | 17.90 ms | 6.25 ms | 3.25 ms | 10.49 | 64.64 |
| 8 | 7.81 | 48.09 | 1.024 s | 25.19 ms | 6.27 ms | 11.29 ms | 19.73 | 121.51 |
| 16 | 14.42 | 88.83 | 1.109 s | 31.28 ms | 7.64 ms | 12.74 ms | 36.16 | 222.77 |

Unified 对照为现有 `FULL_DECODE_ONLY`，各档都保持 162 token。PD graph
虽然 TPOT 可回到约 2.05 ms，但稳定产生 164 token，按 Gate 1 判定无效。

### Tail latency 明细

以下均使用 warm iteration；C=1 只有一个样本，因此该档的 P95/P99 等于单值。

| C | 模式 | E2E P95 / P99 | TTFT P95 / P99 | TPOT P95 / P99 |
|---:|---|---:|---:|---:|
| 1 | PD eager | 1.004 / 1.004 s | 15.53 / 15.53 ms | 6.24 / 6.40 ms |
| 2 | PD eager | 1.010 / 1.010 s | 18.59 / 18.82 ms | 6.20 / 6.28 ms |
| 4 | PD eager | 1.021 / 1.021 s | 17.90 / 17.99 ms | 6.25 / 6.35 ms |
| 8 | PD eager | 1.024 / 1.025 s | 25.19 / 25.34 ms | 6.27 / 6.97 ms |
| 16 | PD eager | 1.109 / 1.109 s | 31.28 / 31.54 ms | 7.64 / 7.71 ms |
| 1 | Unified graph | 0.352 / 0.352 s | 10.40 / 10.40 ms | 2.31 / 2.32 ms |
| 2 | Unified graph | 0.369 / 0.369 s | 15.93 / 16.21 ms | 2.37 / 2.48 ms |
| 4 | Unified graph | 0.381 / 0.381 s | 18.57 / 18.63 ms | 2.42 / 2.47 ms |
| 8 | Unified graph | 0.405 / 0.406 s | 19.17 / 19.29 ms | 2.56 / 2.64 ms |
| 16 | Unified graph | 0.442 / 0.442 s | 21.65 / 21.87 ms | 2.77 / 2.91 ms |

PoC 没有 API admission queue；`T_prefill_queue` 近似 0，表中的 E2E 是 LLM
request admission 到结束。并发 batch 的内部 scheduler wait 包含在 TTFT/E2E
中，没有将其伪装成独立 KV 时间。`T_kv_wait + first decode` 为 15.53 ms；
能够独立归因的 transfer submit→DONE 才记为 1.42 ms。

## Tail latency / interference

Unified 控制 A 为 162/162。A decode 到第 40 token 时注入 4,192-token
prompt B 后：

| 指标 | 控制 | 注入 B 后 |
|---|---:|---:|
| A pre-P95 / pre-P99 | 约 2.30 / 2.38 ms | — |
| A post-P95 / post-P99 | — | 2.06 / 18.71 ms |
| A max TPOT | 约 2.66 ms | 36.34 ms |
| A token correctness | 162/162 | 164/162，index 100 首次分歧 |

这证明该 workload 的 Unified prefill 会造成 decode tail spike，也可能改变
batch 下的 stochastic trajectory。

PD 侧尝试在 A 第 40 token 时启动独立长 prefill B，但注入 vLLM worker
在 GPU0 和 GPU4 都重复卡在 EngineCore 权重加载前（约 691 MiB、0%
utilization），没有产生可用 PD 干扰数据，相关进程已清理。不能据此声称
PD 已改善 P95/P99。详见 `pd_interference_results.md`。

## Resource / lifecycle

C=16 × 63 轮共 1008 请求：

- 1008/1008 raw token 逐 token 等于 baseline；
- GPU0 稳态 17,473 MiB，69 个连续采样点不增长；
- GPU2 稳态 17,395 MiB，68 个连续采样点不增长；
- 约 1.25 秒采样间隔下，稳定区 GPU0 utilization 平均 0.78%、P95 8%，
  GPU2 平均 30.46%、P95 35%；短 Prefill kernel 很容易落在采样间隔之间，
  GPU0 数字只用于说明 workload 稀疏，不能替代 kernel profiler；
- 结束后 GPU0/GPU2 都回到 0 MiB；
- 无残留 experiment worker、KV request 或 output queue。

当前资源成本是两份 LLM engine/KV allocation，约 34.9 GiB，而 Unified 只
需要一份。这也是 1P1D 的部署代价。

本轮长测记录了 GPU memory/utilization 和请求完成数，没有同步采集 worker
RSS，因此只能下“GPU memory 无持续增长”的结论，不能扩大成“所有 host
memory 均已证明无泄漏”。

Profiler 产物：

```text
pd_exp/results/pd_profile_trace/prefill_profile/*.pt.trace.json.gz
pd_exp/results/pd_profile_trace/decode_profile/*.pt.trace.json.gz
pd_exp/results/pd_profile_trace/decode_trace.jsonl
```

服务器没有 `nsys`，因此未生成 Nsight Systems trace。首次 profiler 导出
超过原 30 秒 worker 收尾窗口，已把 `--torch-profile` 收尾等待改为 180 秒；
失败留下的 EngineCore 已按 PID 清理。

实验代码包含 `COSY_PD_PREFILL`、`COSY_PD_DECODE`、`COSY_FLOW`、
`COSY_HIFT` NVTX ranges。NIXL 在本机采用 consumer GPU READ，producer 的
`COSY_PD_KV_SEND` 表示 KV/metadata 已暴露，consumer 使用
`COSY_PD_KV_RECV`/`COSY_PD_KV_RECV_DONE` marks。由于多个异步 READ 可能
乱序完成，KV 精确区间使用 connector JSONL 的 submit→DONE，而不人为构造
可能错误嵌套的 host NVTX range。

## 失败实验和决策

1. 原环境缺少 NIXL：使用独立 venv，未污染现有环境。
2. UCX 全 mlx5 设备超过 soft nofile=1024：提高到 65536，并用 loopback
   TCP 控制面 + CUDA IPC/copy 数据面。
3. 只启用 CUDA IPC/sm 时缺少 active-message transport：保留 TCP。
4. prompt embeddings 在 upstream Nixl scheduler 触发 `len(None)`：实验
   connector 使用 `request.num_prompt_tokens`。
5. producer 辅助 stop token 曾错误进入 Flow：按生产路径在 Flow 前删除。
6. 重新运行 frontend/reference encoder 导致 canonical 输入漂移：改为复用
   序列化 prompt tensor。
7. PD CUDA Graph 产生 164 vs 162 token：停止，不用速度掩盖 correctness。
8. PD 同 P 干扰注入 worker 两次加载卡住：清理并如实保留未完成状态。

vLLM V1 0.11 会强制启用其已有 chunked-prefill scheduler，即使传入
`enable_chunked_prefill=False`。Unified 和 PD 使用相同版本/行为，本实验
没有新增 chunked-prefill 策略。

## Final Gate

```text
[x] Unified baseline 可重复
[x] P/D 运行在不同进程和不同 H100
[x] KV 真正通过 NIXL GPU memory transport
[x] Decode 复用 external cached tokens，不重做完整 prefill
[x] 固定 canonical tensor 时 speech tokens 100% 一致
[x] WAV 正常且字节一致
[x] CER/WER 相对回归为 0
[x] speaker similarity 相对回归为 0（pairwise=1.0）
[x] KV transfer latency 独立测量
[x] concurrency 1/2/4/8/16 完成
[x] TPOT P50/P95/P99 完成
[x] 1008 requests 无 GPU memory 持续泄漏
[ ] PD feature flag：未集成，因性能/graph correctness 不达标
[ ] PD 干扰 P95/P99 对照：Unified 完成，PD 注入未完成
```

## 最终回答

1. PD 是否正确：eager PoC 正确。
2. speech tokens 是否一致：canonical 输入下完全一致。
3. KV overhead：热态约 1.25–1.42 ms（C=1），不是主要瓶颈。
4. 是否改善并发 P95/P99：Unified spike 已证实；PD 对照未完成，不能下结论。
5. 是否提高 req/s：没有；当前正确 PD 明显低于 Unified。
6. 改善最大的指标：热态 decoder admission 后 TTFT 较小，但 TPOT/吞吐退化。
7. 1P:1D 是否合理：作为研究架构合理，当前生产性价比不合理。
8. 是否尝试 1P:2D：暂不值得，先解决 remote-KV + graph correctness。
9. 是否做 P/D/A：暂不值得；LLM PD 本身尚未证明收益。

## Git

本轮只在隔离 worktree `feature/cosyvoice-pd-h100` 中修改实验代码和报告，
未修改生产路径，也未 push。提交前应人工审查本报告和 `pd_exp/`。

最终静态校验命令：

```bash
/SharedData/yangyu/cosyvoice_pd_env/bin/python -m compileall -q pd_exp
/SharedData/yangyu/cosyvoice_pd_env/bin/python pd_exp/test_prompt_nixl_connector.py
/SharedData/yangyu/cosyvoice_pd_env/bin/python -m ruff check pd_exp
```

三项均通过。当前没有 Git commit；实验结果保留在服务器但由
`pd_exp/results/.gitignore` 排除，只跟踪结果索引说明，避免把 WAV、tensor、
trace 和 1008 请求明细误提交到远端。
