# CosyVoice SDAA vLLM 第二轮优化报告

> 2026-07-30 审核说明：本文保留第二轮优化的历史实验记录。历史数字与原始
> JSON 一致，但各轮单次在线运行主要用于展示演进。最终配置的同环境对称
> HTTP A/B 见
> [端到端延迟审核](cosyvoice_sdaa_vllm_e2e_latency_audit_20260730.md)。

本报告只记录 SDAA 环境。第一轮 FlashAttention 基线与实现见
`perf_md/cosyvoice_sdaa_vllm_optimization_report.md`。

## 最终结论

第二轮最终默认配置：

```bash
export COSYVOICE_SDAA_FLOW_FLASH_ATTN=1
export COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH=1
export COSYVOICE_SDAA_HIFT_REFLECTION_PAD=1
export COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE=1
export COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE=1
export COSYVOICE_SDAA_FLOW_STEPS=8
```

这些值在下文的前台启动命令中显式设置。服务运行在逻辑 SDAA 设备 17；
离线 A/B、ASR 与 Perfetto 使用空闲逻辑设备 18。

EvalScope 固定语料、10 请求、并发 1、预热 2 次的结果：

| 指标 | 原始基线 | 第一轮优化 | 第二轮最终 | 第二轮相对第一轮 | 最终相对原始基线 |
|---|---:|---:|---:|---:|---:|
| 平均延迟 | 4.4570 s | 3.3659 s | 2.8659 s | -14.85% | -35.70% |
| P50 | 4.3247 s | 3.1871 s | 2.7118 s | -14.91% | -37.30% |
| P95 | 5.1125 s | 4.1965 s | 3.5217 s | -16.08% | -31.12% |
| 请求吞吐 | 0.2241 req/s | 0.2966 req/s | 0.3482 req/s | +17.41% | +55.35% |
| 平均 RTF | 0.6976 | 0.5249 | 0.4472 | -14.79% | -35.89% |
| 成功请求 | 10/10 | 10/10 | 10/10 | 无退化 | 无退化 |
| 生成音频 | 63.84 s | 63.84 s | 63.84 s | 完全相同 | 完全相同 |

最终 10 条音频全部得到非空、完整的 Qwen3-ASR 转写，总 CER 为
`3 / 284 = 1.0563%`。这与原始基线的约 1.06% 一致，优于第一轮优化版
的约 1.41%。质量检测没有触发重试。

最终人工试听目录：

```text
/mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v3/evalscope_c1/audio/
```

## 从第一轮 trace 得到的热点

使用 `tools/analyze_perfetto_phases.py` 按请求阶段重新聚合第一轮 trace。

Flow 两个请求合计：

- `GEMM N512`：1169.86 ms，3152 次；
- 两种 SDAA FlashAttention kernel：428.73 ms 和 260.84 ms；
- 1D dilation convolution：379.06 ms；
- `sdaaLaunchKernel`：37621 次；
- `sdaa::_local_scalar_dense`：23 次、405.60 ms；
- mask 的 `masked_fill`、`bitwise_not` 与同步仍有可消除空间。

HiFT 两个请求合计：

- `nearest1d` kernel：FP16 217.24 ms、FP32 34.43 ms；
- `reflection_pad1d` CPU fallback：121.18 ms；
- D2H/H2D copy 和同步占比明显。

vLLM 的 `EngineCore` 是独立进程，当前 API 进程的 PyTorch Profiler 只把
该段表现为 `unlabeled_or_llm` 墙钟时间，不能看到 EngineCore 内部算子。
第一轮与最终 trace 中该段分别为 2715.82 ms 和 2703.71 ms，基本不变，
说明本轮收益来自 Flow 与 HiFT，而不是把 vLLM 时间错误归因到其他阶段。

服务日志同时确认当前平台仍显示：

```text
Detected eager backend, disabling AOT compile.
Skipping SDAA graph capture.
```

因此不能根据 API trace 宣称 vLLM compile/graph 已生效。下一轮若优化
vLLM，必须对 EngineCore 单独采 trace，再测试 SDAA graph、固定 shape 桶
或 prefix caching；其中 prefix caching 必须先通过固定 seed 音频确定性门槛。

## 候选实验

所有候选先运行无 trace 的交错顺序 A/B；未取得明显收益的候选没有生成
Perfetto trace。

| 候选 | 结果 | 正确性 | 决策 |
|---|---:|---|---|
| 去除 API 非流式全有效 attention mask | 约 -0.75% 至 -1.22% | 固定 seed WAV 哈希完全一致 | 保留 |
| RoPE cos/sin 缓存 | 在去 mask 后仅约 -0.06% | 哈希一致 | 移除代码 |
| QKV 单 GEMM 融合 | 在去 mask + RoPE 后约 -0.29% | 哈希一致 | 移除代码 |
| HiFT reflection/nearest 等价替换 | 初版额外约 -0.53% | 哈希一致 | 继续分析 |
| Flow 10 步降至 8 步 | -10.34% | 5 条 CER 0.70%，无退化 | 默认采用 |
| Flow 10 步降至 6 步 | -20.65% | 5 条 CER 0.70%，无退化 | 仅保留试听候选 |
| 连续化后再 `repeat_interleave` | 在 8 步基础上再 -4.55% | 10 次固定请求哈希一致 | 保留 |

6 步虽然自动语义指标通过，但 Euler 步数属于质量/速度参数，韵律和音色
仍需更大语料的主观测试，因此默认采用更保守的 8 步。

候选试听目录：

```text
cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/round2/audio/
├── unmasked_hift/         # 10 步
├── unmasked_hift_steps8/  # 8 步
└── unmasked_hift_steps6/  # 6 步
```

## 最终 Perfetto

最终 trace 与第一轮成功 trace 使用相同两条请求：

| 阶段，两请求合计 | 第一轮 | 第二轮最终 | 变化 |
|---|---:|---:|---:|
| full request | 6418.70 ms | 5571.09 ms | -13.21% |
| Flow | 3132.15 ms | 2558.80 ms | -18.31% |
| HiFT | 570.05 ms | 307.90 ms | -45.99% |
| unlabeled / vLLM | 2715.82 ms | 2703.71 ms | -0.45% |

Flow 的主要 kernel 调用按 10 步到 8 步成比例下降：

- GEMM：3152 次降到 2520 次；
- 两种 FlashAttention kernel：各 220 次降到各 176 次；
- Flow `sdaa::_local_scalar_dense`：23 次降到 4 次；
- Flow `sdaaLaunchKernel`：37621 次降到约 29066 次。

最终 HiFT trace 中已经没有高耗时的 `nearest1d` 和
`reflection_pad1d` CPU fallback。先连续化再展开避免了中间 v2 trace 中
约 248 ms 的 `copy_stride_stride_zero` 热点。

保留的压缩 trace：

```text
cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/
├── baseline/perfetto/baseline.perfetto.json.gz
├── optimized/perfetto/optimized.perfetto.json.gz
└── optimized_v3/perfetto/optimized_v3.perfetto.json.gz
```

中间 v2 trace 已删除；无收益候选没有生成 trace。三个保留文件均通过
`gzip -t`，最终 trace 大小约 16.95 MB。

## 手动复现

### 启动最终服务

宿主机终端 A：

```bash
/opt/kube/bin/docker start yy-cosyvoice-sdaa-vllm
/opt/kube/bin/docker exec -it \
  -e SDAA_VISIBLE_DEVICES=17 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  /bin/bash
```

进入容器后：

```bash
source /opt/tecoai/setvars.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_env_py310
cd /workspace/CosyVoice
set -a
source /workspace/CosyVoice/cosyvoice_sdaa_vllm_api.env
set +a
export COSYVOICE_MODEL_DIR=/workspace/model
export COSYVOICE_MODEL_ALIAS=cosyvoice3-0.5b
export COSYVOICE_HOST=127.0.0.1
export COSYVOICE_PORT=8021
export COSYVOICE_LOAD_VLLM=true
export COSYVOICE_FP16=true
export COSYVOICE_MAX_CONCURRENCY=1
export COSYVOICE_MAX_QUEUE_SIZE=16
export COSYVOICE_REQUEST_TIMEOUT_SECONDS=900
export COSYVOICE_DEFAULT_SEED=0
export COSYVOICE_QUALITY_CHECK_ENABLED=true
export COSYVOICE_SDAA_FLOW_FLASH_ATTN=1
export COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH=1
export COSYVOICE_SDAA_HIFT_REFLECTION_PAD=1
export COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE=1
export COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE=1
export COSYVOICE_SDAA_FLOW_STEPS=8
export COSYVOICE_VLLM_SDAA_GRAPH=0
exec python -m api_server.main
```

这是第二轮 `optimized_v3` 的历史复现配置，因此 graph 显式关闭。当前推荐的
第三轮配置应设置 `COSYVOICE_VLLM_SDAA_GRAPH=1`。API Key 从被忽略的
`cosyvoice_sdaa_vllm_api.env` 读取，不在文档中明文保存。服务加载和请求日志
直接显示在终端 A。

宿主机终端 B：

```bash
/opt/kube/bin/docker exec yy-cosyvoice-sdaa-vllm \
  curl -fsS http://127.0.0.1:8021/health
```

### EvalScope

```bash
/opt/kube/bin/docker exec -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  /root/miniconda3/envs/vllm_env_py310/bin/python \
  tools/cosyvoice_evalscope_perf.py \
    --prompt-file tools/cosyvoice_perf_prompts.txt \
    --outputs-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v3/evalscope_c1 \
    --api-key-env-file cosyvoice_sdaa_vllm_api.env \
    --number 10 --parallel 1 --warmup 2 --seed 20260726 \
    --run-name cosyvoice-sdaa-round2-final
```

### Perfetto，直接生成 `.json.gz`

```bash
/opt/kube/bin/docker exec \
  -e SDAA_VISIBLE_DEVICES=18 \
  -e COSYVOICE_SDAA_FLOW_FLASH_ATTN=1 \
  -e COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH=1 \
  -e COSYVOICE_SDAA_HIFT_REFLECTION_PAD=1 \
  -e COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE=1 \
  -e COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE=1 \
  -e COSYVOICE_SDAA_FLOW_STEPS=8 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  bash tools/run_cosyvoice_perfetto_sdaa.sh \
    --prompt-file tools/cosyvoice_perf_prompts.txt \
    --outputs-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v3/perfetto \
    --warmup 1 --active 2 --seed 20260726 \
    --tag optimized_v3

/opt/kube/bin/docker exec -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  /root/miniconda3/envs/vllm_env_py310/bin/python \
  tools/analyze_perfetto_trace.py \
    cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v3/perfetto/optimized_v3.perfetto.json.gz \
    --output \
    cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v3/perfetto/optimized_v3.analysis.json

/opt/kube/bin/docker exec -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  /root/miniconda3/envs/vllm_env_py310/bin/python \
  tools/analyze_perfetto_phases.py \
    cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v3/perfetto/optimized_v3.perfetto.json.gz \
    --output \
    cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v3/perfetto/optimized_v3.phase_hotspots.json
```

### ASR 语义回归

```bash
/opt/kube/bin/docker exec \
  -e SDAA_VISIBLE_DEVICES=18 \
  -e LD_LIBRARY_PATH=/opt/tecoai/lib64 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  /root/miniconda3/envs/vllm_env_py310/bin/python \
  tools/cosyvoice_qwen3_asr_semantic_check.py \
    --audio cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v3/evalscope_c1/audio \
    --expected-text-file tools/cosyvoice_perf_prompts.txt \
    --model /tecogpfs/models/Qwen/Qwen3-ASR-0___6B \
    --gpu-memory-utilization 0.8 \
    --output-jsonl cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v3/evalscope_c1/asr.jsonl \
    --summary-json cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v3/evalscope_c1/asr_summary.json
```

## 下一批优化方向

本节中的 vLLM decode graph 已在第三轮完成，结果和固定路径复现命令见
`perf_md/cosyvoice_sdaa_vllm_optimization_round3.md`。

1. 对独立的 vLLM EngineCore 采专用 trace，验证 SDAA graph 是否能覆盖稳定
   decode shape；当前 API trace 无法回答该问题。
2. Flow 剩余热点仍是 GEMM、两种 FlashAttention、1D convolution 与约
   2.9 万次 kernel launch。优先评估固定长度桶的图捕获或 AdaLN/逐元素
   融合，所有方案必须保留固定 seed 与 ASR 门槛。
3. 6 步模式可以作为显式性能档位继续做 MOS/ABX 主观测试；未完成主观
   质量评估前不应替换 8 步默认值。
4. HiFT 剩余主要是 GEMM、transpose、1D convolution、STFT/ISTFT 与少量
   CPU fallback。下一步应先给这些模块加更细的阶段标注，再逐项消除
   fallback，避免再次把热点转移到 copy/layout。
