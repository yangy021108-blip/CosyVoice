# CosyVoice SDAA vLLM 第三轮优化报告

> 2026-07-30 审核说明：本文保留第三轮优化的历史实验记录。历史数字与原始
> JSON 一致，但各轮单次在线运行主要用于展示演进。最终配置的同环境对称
> HTTP A/B 见
> [端到端延迟审核](cosyvoice_sdaa_vllm_e2e_latency_audit_20260730.md)。

本轮延续
`perf_md/cosyvoice_sdaa_vllm_optimization_round2.md` 的计划，重点验证
vLLM EngineCore 的 SDAA graph，并继续检查 Flow、HiFT 和 PyTorch 层的
可优化项。测试日期为 2026-07-28，服务运行在
`yy-cosyvoice-sdaa-vllm`，模型为 `/workspace/model`。

## 最终结论

最终保留的新增优化是 vLLM 的单序列 decode graph：

- API 最大并发为 1，设置 `max_num_seqs=1`；
- 只捕获稳定的 batch size 1 decode，使用
  `FULL_DECODE_ONLY` 和 `cudagraph_capture_sizes=[1]`；
- 变长 prompt prefill 保持 eager；
- 保持 `enable_prefix_caching=False`、
  `enable_chunked_prefill=False` 和 `async_scheduling=False`，避免固定
  seed 的 token 序列改变。

代码位置：

- `cosyvoice/cli/model.py`：构造 vLLM `EngineArgs`；
- 手动启动命令显式设置 `COSYVOICE_VLLM_SDAA_GRAPH=1`。

vLLM 日志确认：

```text
cudagraph_mode=FULL_DECODE_ONLY
cudagraph_capture_sizes=[1]
Graph capturing finished in 0 secs, took 0.01 GiB
```

### 独立 A-candidate-A

每组均为固定语料、10 请求、并发 1。两次 baseline 分别在 candidate 前后
独立重启服务，避免把启动顺序或热状态误认为优化收益。

| 运行 | decode graph | 平均服务推理延迟 | 成功 | 固定 seed WAV |
|---|---:|---:|---:|---|
| baseline A | 关闭 | 2.9519 s | 10/10 | 基准 |
| candidate | 开启 | 2.8624 s | 10/10 | 哈希一致 |
| baseline B | 关闭 | 2.9673 s | 10/10 | 哈希一致 |
| baseline A/B 合并 | 关闭 | 2.9596 s | 20/20 | 哈希一致 |

candidate 相对合并 baseline 的平均服务推理延迟下降 `3.2834%`。原始结果
保存在：

```text
cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/round3/
├── graph_baseline_a.json
├── graph_candidate.json
└── graph_baseline_b.json
```

## 最终 EvalScope 结果

固定语料 10 请求、并发 1、预热 2 次、起始 seed 20260726。

| 指标 | 原始基线 | 第二轮 optimized_v3 | 第三轮 optimized_v4 | v4 相对 v3 | v4 相对原始 |
|---|---:|---:|---:|---:|---:|
| 客户端平均延迟 | 4.4570 s | 2.8659 s | 2.7612 s | -3.66% | -38.05% |
| 客户端 P50 | 4.3247 s | 2.7118 s | 2.6183 s | -3.45% | -39.46% |
| 客户端 P95 | 5.1125 s | 3.5217 s | 3.4145 s | -3.05% | -33.21% |
| 服务平均推理延迟 | — | 2859.8147 ms | 2754.8363 ms | -3.67% | — |
| 请求吞吐 | 0.2241 req/s | 0.3482 req/s | 0.3615 req/s | +3.83% | +61.32% |
| 音频吞吐 | — | 2.2229 倍实时 | 2.3080 倍实时 | +3.83% | — |
| 平均 RTF | 0.6976 | 0.4472 | 0.4306 | -3.72% | -38.28% |
| 成功请求 | 10/10 | 10/10 | 10/10 | 无退化 | 无退化 |
| 生成音频 | 63.84 s | 63.84 s | 63.84 s | 完全相同 | 完全相同 |

`optimized_v3` 和 `optimized_v4` 的 10 个 WAV 逐文件 SHA-256 完全一致。
Qwen3-ASR 检查结果也是 10/10 非空转写，字符错误率为：

```text
3 / 284 = 1.056338%
```

因此该优化没有通过缩短音频、改变 token 序列或降低可识别性换取速度。
可人工试听的正确音频位于：

```text
/mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v4/evalscope_c1/audio/
```

## Perfetto 结果

最终 trace 由 profiler 直接写成压缩 JSON，不存在先生成大 JSON 再压缩的
中间文件：

```text
/mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v4/perfetto/optimized_v4.perfetto.json.gz
```

文件大小为 16,981,625 bytes，包含两条活动请求。两条 profiler 音频的
SHA-256 也与 EvalScope 正确基线一致。

| 两请求合计 | optimized_v3 | optimized_v4 | 变化 |
|---|---:|---:|---:|
| full request | 5571.09 ms | 5358.11 ms | -3.82% |
| Flow | 2558.80 ms | 2462.89 ms | -3.75% |
| HiFT | 307.90 ms | 322.60 ms | +4.77% |
| unlabeled / LLM | 2703.71 ms | 2571.95 ms | -4.87% |

vLLM EngineCore 是独立子进程。当前 PyTorch profiler 能统计父进程的完整
请求墙钟时间，但不能在同一 trace 中展开 EngineCore 内部 kernel。因此，
decode graph 的依据是 vLLM graph capture 日志、独立 A-candidate-A 和
`unlabeled / LLM` 墙钟段共同验证，不能把该段误解为父进程的 SDAA 算子。

最终 trace 中 Flow 仍然是下一轮的主要目标：

- `sdaaLaunchKernel`：29,066 次，合计 1162.20 ms；
- N512 GEMM：2520 次，合计 934.63 ms；
- 两类 FlashAttention：各 176 次，合计 551.03 ms；
- 1D dilation convolution：36 次，合计 303.57 ms；
- copy-stride kernel：4950 次，合计 110.64 ms。

HiFT 仍有以下可见开销：

- H2D 204 次、D2H 78 次；
- `cpufallback::convolution` 10 次，合计 39.04 ms；
- `cpufallback::elu.out` 10 次，合计 13.65 ms；
- 每请求重复执行 weight norm、STFT 和 ISTFT。

## 未保留的候选

没有明显收益的候选均未生成 Perfetto trace；失败候选生成的 WAV 已删除。
实验统计 JSON 保留在 `round3` 目录，便于避免重复尝试。

| 候选 | 性能或现象 | 正确性 | 决策 |
|---|---|---|---|
| 去除 LLM token 轮询 sleep | 约 -0.09% | 哈希一致 | 波动范围，回退 |
| 请求结束不调用 empty cache | 约 -0.02% | 哈希一致 | 无稳定收益，回退 |
| Euler 不变量与 buffer 复用 | 约 -0.06% | 20/20 哈希一致 | 无稳定收益，回退 |
| vLLM eager backend 加 Inductor 配置 | 约 -0.8% | 哈希一致 | 日志仍为 eager，不能归因，回退 |
| vLLM async scheduling | 约 -9% | 5 个固定 seed 中 4 个改变 | 拒绝 |
| vLLM prefill/piecewise graph | 约 -7.6% | 5/5 改变，BlockAttention 提示不兼容 | 拒绝 |
| vLLM chunked prefill | 无稳定收益 | 哈希一致 | 回退 |
| 整个 Flow/estimator SDAA graph | 多请求后 EngineCore 退出并产生静音 | 失败 | 删除代码和音频 |
| 整个 Flow `torch.compile` | Torch-SDAA codegen `NotImplemented` | 无结果 | 删除代码 |
| Flow FF 子模块 `torch.compile` | 稳态更慢 | WAV 哈希改变 | 删除代码和音频 |

async scheduling 和 prefill graph 的速度数字不能作为有效优化，因为它们改变
了同一 seed 的生成 token 和音频长度。除非先修复 SDAA vLLM 的 RNG/调度
确定性，否则不应启用。

## 手动复现

以下命令不定义 shell 路径变量，不使用 shell 函数。API Key 由
`cosyvoice_sdaa_vllm_api.env` 读取，不在命令或文档中明文出现。

### 终端 A：启动 vLLM 服务

在宿主机执行：

```bash
/opt/kube/bin/docker start yy-cosyvoice-sdaa-vllm
/opt/kube/bin/docker exec -it \
  -e SDAA_VISIBLE_DEVICES=18 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  /bin/bash
```

进入容器后执行：

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
export COSYVOICE_VLLM_SDAA_GRAPH=1
exec python -m api_server.main
```

服务以前台方式运行。终端会直接显示 `/workspace/model` 的加载过程、
`FULL_DECODE_ONLY` graph capture、`127.0.0.1:8021` 监听状态和请求日志。
出现 `Application startup complete` 后保持终端不动。

### 终端 B：人工试听测试

在宿主机新开 shell：

zero-shot、无 instructions、WAV：

```bash
cd /mnt/nvme/application/yangyu/CosyVoice
python3 tools/test_cosyvoice_api.py \
  --backend vllm \
  --voice-mode zero_shot \
  --voice default \
  --response-format wav \
  --seed 0 \
  --output-dir /mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/manual/vllm_zero_shot_no_instruction_wav
```

zero-shot、有 instructions、WAV：

```bash
cd /mnt/nvme/application/yangyu/CosyVoice
python3 tools/test_cosyvoice_api.py \
  --backend vllm \
  --voice-mode zero_shot \
  --voice default \
  --instructions "请用四川话、开心地说这句话。" \
  --response-format wav \
  --seed 0 \
  --output-dir /mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/manual/vllm_zero_shot_instruction_wav
```

zero-shot、无 instructions、裸 PCM：

```bash
cd /mnt/nvme/application/yangyu/CosyVoice
python3 tools/test_cosyvoice_api.py \
  --backend vllm \
  --voice-mode zero_shot \
  --voice default \
  --text "这是 PCM 输出测试。" \
  --response-format pcm \
  --seed 0 \
  --output-dir /mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/manual/vllm_zero_shot_no_instruction_pcm
```

脚本每次只执行一个推理组合，先用 `/v1/audio/voices` 校验音色 mode，再验证
WAV/PCM。HTTP 或质量门禁错误只写 `.error.json`，不会伪装成音频。
当前模型没有注册 SFT 音色；SFT 测试命令和注册条件见
`api_server/README_SDAA.md`。

2026-07-29 已分别实测上述三个 zero-shot 目录，`results.json` 均为
`ok: true`。SFT 可用性检查在请求前明确报告当前没有注册 SFT voice，且不
生成音频。

### EvalScope

```bash
/opt/kube/bin/docker exec \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  /root/miniconda3/envs/vllm_env_py310/bin/python \
  tools/cosyvoice_evalscope_perf.py \
    --prompt-file tools/cosyvoice_perf_prompts.txt \
    --outputs-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v4/evalscope_c1 \
    --api-key-env-file cosyvoice_sdaa_vllm_api.env \
    --number 10 \
    --parallel 1 \
    --warmup 2 \
    --seed 20260726 \
    --run-name cosyvoice-sdaa-round3-decode-graph
```

### Perfetto，直接生成 `.json.gz`

```bash
/opt/kube/bin/docker exec \
  -e SDAA_VISIBLE_DEVICES=17 \
  -e COSYVOICE_VLLM_SDAA_GRAPH=1 \
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
    --outputs-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v4/perfetto \
    --warmup 1 \
    --active 2 \
    --seed 20260726 \
    --tag optimized_v4
```

```bash
/opt/kube/bin/docker exec \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  /root/miniconda3/envs/vllm_env_py310/bin/python \
  tools/analyze_perfetto_trace.py \
    cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v4/perfetto/optimized_v4.perfetto.json.gz \
    --output cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v4/perfetto/optimized_v4.analysis.json
```

```bash
/opt/kube/bin/docker exec \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  /root/miniconda3/envs/vllm_env_py310/bin/python \
  tools/analyze_perfetto_phases.py \
    cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v4/perfetto/optimized_v4.perfetto.json.gz \
    --output cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v4/perfetto/optimized_v4.phase_hotspots.json
```

### ASR 语义回归

```bash
/opt/kube/bin/docker exec \
  -e SDAA_VISIBLE_DEVICES=17 \
  -e LD_LIBRARY_PATH=/opt/tecoai/lib64 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  /root/miniconda3/envs/vllm_env_py310/bin/python \
  tools/cosyvoice_qwen3_asr_semantic_check.py \
    --audio cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v4/evalscope_c1/audio \
    --expected-text-file tools/cosyvoice_perf_prompts.txt \
    --model /tecogpfs/models/Qwen/Qwen3-ASR-0___6B \
    --gpu-memory-utilization 0.8 \
    --output-jsonl cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v4/evalscope_c1/asr.jsonl \
    --summary-json cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized_v4/evalscope_c1/asr_summary.json
```

## 下一轮优化方向

1. 给 vLLM EngineCore 子进程增加独立 profiler 入口，直接比较 graph replay
   前后的 BlockAttention、GEMM 和 launch 数。
2. 优先在 Flow 内融合 AdaLN、sin/cos、mul/add/cat 和 copy-stride 路径；
   29,066 次 launch 表明小算子调度仍有明显空间。Torch-SDAA 3.2.0 的整图
   捕获不稳定，应先从单个等价自定义算子或固定 shape 小模块开始。
3. 检查 SDAA vLLM async/prefill 对 sampler RNG 状态的推进顺序。只有同一
   seed 的 token、WAV 哈希和 ASR 同时通过，才能重新考虑这两个高收益候选。
4. 在加载阶段移除 HiFT 推理用 weight norm，并为 CPU fallback 的
   convolution/ELU 做局部等价改写；每个候选先跑无 trace A/B 和 10 个
   固定 seed，再决定是否保留 trace。
