# CosyVoice3 PD 干扰实验结果

## 实验目的

验证 Unified vLLM 中长 prompt Prefill 是否会干扰同卡正在进行的 Decode，
并尝试用 1P1D 将 Prefill 放到独立 H100 后复测同一指标。主要观察 Request A
逐 speech-token TPOT 的 P95、P99 和最大值，同时执行逐 token correctness gate。

## 固定条件

- 模型：CosyVoice3-0.5B vLLM backend；
- Request A：canonical `[131, 896]` BF16 `prompt_embeds`，seed 20260726；
- A 的 Unified reference：162 raw speech tokens；
- Request B：4,192-token 长 prompt；
- 注入时机：A 产生第 40 个 speech token 后；
- Unified 模式下 A、B 共用一个 H100 和一个 vLLM engine；
- PD 目标模式下 A Decode 使用 GPU2，B Prefill 使用独立 GPU；
- 未改变 sampling、Flow、HiFT 或 frontend。

## Unified 结果

结果文件：

```text
pd_exp/results/unified_interference.json
```

| 项目 | 无 B 的控制 A | 注入 B 的 A |
|---|---:|---:|
| 注入前 TPOT P95 | 约 2.30 ms | — |
| 注入前 TPOT P99 | 约 2.38 ms | — |
| 注入后 TPOT P95 | — | 2.06 ms |
| 注入后 TPOT P99 | — | 18.71 ms |
| 注入后最大 TPOT | — | 36.34 ms |
| raw token 数 | 162 | 164 |
| 与 reference 首次差异 | 无 | index 100 |

结论：长 Prefill 与 Decode 共卡时，A 出现明显 TPOT tail spike。由于当前
sampling trajectory 对 batch/scheduling 顺序敏感，A 还从 162 token 改变为
164 token。因此该实验能证明 Unified 存在干扰，但不能把注入请求的输出当成
正确性 baseline。

## PD 对照状态

先后尝试让 B 在 GPU0（计划中的 Prefill GPU）和 GPU4 上以独立 vLLM
EngineCore 启动。两次都停在 EngineCore 初始化、权重加载前，显存约
691 MiB、GPU utilization 为 0%，没有进入 B Prefill，也没有形成可用的 A
TPOT 对照。相关实验 PID 已清理，GPU0/GPU2/GPU4 均恢复到实验前状态。

因此当前不能声称：

- PD 已降低 Request A 的 TPOT P95/P99；
- PD 已消除 stochastic trajectory 变化；
- 独立 Prefill GPU 在这个注入方式下已通过稳定性验证。

## 判断

Unified 干扰现象成立，但 PD 干扰对照未完成。结合正确 PD eager 的 TPOT
约 6.1 ms、Unified graph 的 TPOT 约 2.1 ms，以及 PD graph 的 token 漂移，
本轮数据不足以支持将 PD 接入生产。后续应先解决 remote KV 与
`FULL_DECODE_ONLY` CUDA Graph 的 correctness，再在进程启动前预加载一个
长期存活的 B Prefill worker，避免在 A decode 中途创建新的 vLLM EngineCore。

推荐的下一次实验方法：

1. A、B worker 都在测量前完成模型加载和 warm-up；
2. B worker 阻塞在 IPC barrier，而不是收到信号后才启动 engine；
3. A 第 40 token 只释放 barrier，触发已就绪的 B 执行长 Prefill；
4. 分别记录 A 的 pre/post TPOT P50/P95/P99/max；
5. 先要求控制与注入两条 A trajectory 都通过逐 token gate；若 sampling 本身
   对调度顺序敏感，则补充 greedy/control-only 实验，但不能替代产品采样配置。
