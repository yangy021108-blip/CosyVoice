# CosyVoice SDAA vLLM 端到端延迟审核（2026-07-30）

## 审核结论

在同一台服务器、同一容器、同一模型和同一逻辑 SDAA 设备上，用真实
`/v1/audio/speech` HTTP 请求重新进行对称 A/B 后，最终优化配置相对关闭
全部优化的基线，平均客户端端到端延迟从 4.511 s 降至 2.629 s，降低
**41.71%**；20/20 个配对请求都更快。相对第四轮配置，最终两项优化仍使
平均延迟降低 **1.83%**，20 个配对请求中 19 个更快。

因此，历史文档中的总体优化方向和数量级成立。旧文档各轮在线结果是不同
日期的单次运行，适合展示演进过程，不应单独当作严格的累计 A/B 证据；本次
同环境、每轮重启服务的对称测试是当前端到端结论的主要依据。

## 测试方法

- 环境：TECO 3.2.0，CosyVoice3-0.5B，FP16，vLLM backend；
- 服务端点：`http://127.0.0.1:8021/v1/audio/speech`；
- 在线服务设备：`SDAA_VISIBLE_DEVICES=18`；
- 固定 5 条提示词，按相同顺序循环；
- 每次服务启动后预热 2 次，正式请求 10 次，并发 1，seed 为
  `20260726`；
- 每个配置独立启动服务两次，共 20 个正式请求；
- 对称顺序：
  `baseline → round4 → optimized → optimized → round4 → baseline`；
- 客户端延迟包含 HTTP、API 路由、vLLM LLM、Flow、HiFT 和 WAV 编码；
- 每个配置的 20 条结果按 cycle 和请求序号配对，报告配对相对变化和近似
  95% 置信区间。

三个配置的关键区别如下：

| 配置 | Flow 步数 | 第 1～4 轮优化 | RoPE 预计算 | HiFT F0 SDAA FP32 |
| --- | ---: | --- | --- | --- |
| baseline | 10 | 全部关闭 | 关闭 | 关闭 |
| round4 | 8 | 全部开启 | 关闭 | 关闭 |
| optimized | 8 | 全部开启 | 开启 | 开启 |

“第 1～4 轮优化”包括 Flow FlashAttention、unmasked fast path、HiFT
reflection pad / nearest contiguous upsample、vLLM SDAA graph 和 fused
norm。

## EvalScope 端到端结果

每行聚合两次独立服务启动、20 个正式 HTTP 请求：

| 配置 | 客户端均值 | P50 | P95 | 服务端均值 | 请求吞吐 | 平均 RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 4.511 s | 4.399 s | 5.420 s | 4506.21 ms | 0.2215 req/s | 0.7061 |
| round4 | 2.678 s | 2.544 s | 3.752 s | 2673.11 ms | 0.3727 req/s | 0.4176 |
| optimized | 2.629 s | 2.490 s | 3.705 s | 2623.98 ms | 0.3796 req/s | 0.4100 |

| 对比 | 客户端均值 | 服务端均值 | 请求吞吐 | 平均 RTF | 配对更快 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline → round4 | -40.62% | -40.68% | +68.30% | -40.85% | 20/20 |
| baseline → optimized | **-41.71%** | **-41.77%** | **+71.42%** | **-41.94%** | **20/20** |
| round4 → optimized | -1.83% | -1.84% | +1.85% | -1.83% | 19/20 |

配对客户端相对变化：

- baseline → optimized：均值 -41.86%，中位数 -43.90%，近似 95%
  区间 `[-44.09%, -39.64%]`；
- round4 → optimized：均值 -1.85%，中位数 -1.87%，近似 95%
  区间 `[-2.26%, -1.45%]`。

两次独立启动的均值分别为：

| 配置 | 第一次 | 第二次 | 两次差值 |
| --- | ---: | ---: | ---: |
| baseline | 4.496 s | 4.526 s | 0.031 s |
| round4 | 2.674 s | 2.683 s | 0.008 s |
| optimized | 2.638 s | 2.620 s | 0.018 s |

## 音频正确性

- 60/60 个正式 HTTP 请求成功，质量重试为 0；
- 每个配置生成 127.68 秒音频；
- baseline → round4、baseline → optimized、round4 → optimized 的
  20/20 对音频时长均一致；
- 重新启动服务后的 WAV SHA256 不一致，因此本次测试不宣称逐字节或波形
  等价；
- 从每个配置选取同一轮 10 条音频，共 30 条，用独立
  Qwen3-ASR-0.6B 检查：总 CER 为 10/852，即 1.17%；
- baseline、round4、optimized 分别为 3/284、3/284、4/284，所有转写
  都非空且语义完整，没有漏句、重复、静音、截断或异常拉长。

ASR 只证明语义正确，不能替代主观音质、音色和自然度评价。尤其是
baseline 的 10 步与最终配置的 8 步不是数值等价路径；若要声称主观质量
完全持平，仍需对保留的 WAV 做人工 ABX 或 MOS。人工试听目录：

```text
/mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/run_00_baseline_cycle_0/audio
/mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/run_01_round4_cycle_0/audio
/mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/run_02_optimized_cycle_0/audio
```

## 历史文档审核

历史 JSON 与五份优化文档中的主要表格数字一致，没有发现抄录或计算错误。
本次结果还复现了三个关键节点：

| 节点 | 历史单次在线均值 | 本次两次启动聚合 | 差异 |
| --- | ---: | ---: | ---: |
| 原始 baseline | 4.457 s | 4.511 s | +1.21% |
| 第四轮 | 2.681 s | 2.678 s | -0.09% |
| 第五轮最终配置 | 2.636 s | 2.629 s | -0.24% |

对各文档的审核意见：

| 文档 | 审核意见 |
| --- | --- |
| `cosyvoice_sdaa_vllm_optimization_report.md` | 第一轮 FlashAttention 的历史结果有效，但只代表第一轮节点 |
| `cosyvoice_sdaa_vllm_optimization_round2.md` | 历史结果和原始 JSON 一致；其中 API trace 不能证明 graph 生效的限制说明正确 |
| `cosyvoice_sdaa_vllm_optimization_round3.md` | graph 与第三轮组合配置的历史结果有效，但在线数字来自单次运行 |
| `cosyvoice_sdaa_vllm_optimization_round4.md` | fused norm 同进程交错 A/B 和在线结果均有原始数据支持 |
| `cosyvoice_sdaa_vllm_optimization_round5.md` | RoPE/F0 候选 A/B 有效；旧在线结果只有单次运行且含异常慢请求，本次对称 HTTP A/B 补强了结论 |

## 复现命令

以下命令在容器中的 `/workspace/CosyVoice` 执行。开始前必须已有一个通过
`/ready` 检查的 vLLM API 服务；脚本从该进程继承已经验证的模型、端口、
设备和 API key 环境，不会输出或写入 API key。测试期间会依次重启服务，
结束或发生异常时都会恢复最终优化服务。

```bash
cd /workspace/CosyVoice

python tools/benchmark_cosyvoice_vllm_e2e_ab.py \
  --output-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_recheck \
  --prompt-file tools/cosyvoice_perf_prompts.txt \
  --number 10 \
  --warmup 2 \
  --seed 20260726 \
  --port 8021 \
  --startup-timeout 180
```

输出目录必须为空或不存在，防止旧结果和新结果混合。脚本只在终端打印精简
摘要，完整数据写入 `results.json`。

独立 ASR 回归命令在宿主机执行：

```bash
/opt/kube/bin/docker exec \
  -e SDAA_VISIBLE_DEVICES=19 \
  -e LD_LIBRARY_PATH=/opt/tecoai/lib64 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm bash -lc '
    source /opt/tecoai/setvars.sh
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate vllm_env_py310
    python tools/cosyvoice_qwen3_asr_semantic_check.py \
      --audio \
        cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/run_00_baseline_cycle_0/audio \
        cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/run_01_round4_cycle_0/audio \
        cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/run_02_optimized_cycle_0/audio \
      --expected-text-file tools/cosyvoice_perf_prompts.txt \
      --model /tecogpfs/models/Qwen/Qwen3-ASR-0___6B \
      --gpu-memory-utilization 0.8 \
      --output-jsonl cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/asr_cycle0.jsonl \
      --summary-json cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/asr_cycle0_summary.json
  '
```

## 结果文件

```text
cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/results.json
cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/asr_cycle0.jsonl
cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/asr_cycle0_summary.json
cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/run_00_baseline_cycle_0/
cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/run_01_round4_cycle_0/
cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/run_02_optimized_cycle_0/
cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/run_03_optimized_cycle_1/
cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/run_04_round4_cycle_1/
cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_20260730/run_05_baseline_cycle_1/
```

没有为本次端到端审核新建 Perfetto trace。历史有效优化的压缩 trace 继续
保留；本次仅验证 HTTP 端到端效果，避免生成重复的大型 trace。

## 当前恢复配置

测试结束后在线服务已恢复并通过 `/ready`：

```text
SDAA_VISIBLE_DEVICES=18
COSYVOICE_SDAA_FLOW_FLASH_ATTN=1
COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH=1
COSYVOICE_SDAA_HIFT_REFLECTION_PAD=1
COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE=1
COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE=1
COSYVOICE_SDAA_FLOW_STEPS=8
COSYVOICE_VLLM_SDAA_GRAPH=1
COSYVOICE_SDAA_FLOW_FUSED_NORM=1
COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE=1
COSYVOICE_SDAA_HIFT_F0_FP32=1
```
