# CosyVoice3 vLLM 在 TECO SDAA 上的推理优化报告

> 2026-07-30 审核说明：本文保留第一轮优化的历史实验记录。历史数字与原始
> JSON 一致，但单次在线运行不等同于同环境累计 A/B。最终配置的对称重启、
> 真实 HTTP 端到端复测见
> [端到端延迟审核](cosyvoice_sdaa_vllm_e2e_latency_audit_20260730.md)。

## 结论

在 TECO 3.2.0、CosyVoice3-0.5B、FP16、vLLM backend 和相同固定语料下，
把 Flow/DiT 的 PyTorch math SDPA 替换为 TECO SDAA 非因果
FlashAttention 后，已经取得可重复的端到端明显收益：

| 指标 | 基线 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| EvalScope 平均请求延迟 | 4.457 s | 3.366 s | **-24.48%** |
| EvalScope P50 延迟 | 4.325 s | 3.187 s | **-26.31%** |
| EvalScope P95 延迟 | 5.112 s | 4.197 s | **-17.92%** |
| EvalScope 请求吞吐 | 0.224 req/s | 0.297 req/s | **+32.31%** |
| 音频吞吐 | 1.431× realtime | 1.893× realtime | **+32.31%** |
| 平均 RTF | 0.698 | 0.525 | **-24.76%** |
| Perfetto 端到端 | 4098.2 ms | 3209.4 ms | **-21.69%** |
| Perfetto Flow | 2449.4 ms | 1566.1 ms | **-36.06%** |

EvalScope 基线与优化版均为 10/10 成功。优化版 10 个 WAV 全部为
24 kHz、单声道、PCM16，质量检查重试次数均为 0。Qwen3-ASR-0.6B 对
10/10 音频均得到非空、完整语义转写，字符错误率为 1.41%；基线为
1.06%，相差 1 个字符，没有出现漏句、重复、纯静音、异常截断或异常拉长。

## 环境与测试方法

- 加速卡：TECO_AICARD_01；
- TECO-SMI：1.14；
- SDAA Driver/Runtime/Torch-SDAA：3.2.0；
- 模型：`/tecogpfs/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512`；
- 服务容器：`yy-cosyvoice-sdaa-vllm`；
- API 服务使用逻辑设备 `17`；离线 Perfetto 与 ASR 检查使用空闲逻辑
  设备 `18`，避免与服务争用；
- 固定 5 条提示词，按相同顺序循环；EvalScope 预热 2 次、正式请求
  10 次、并发 1、起始 seed 为 `20260726`；
- Perfetto 预热 1 次、采集 2 次，使用同一批提示词和 seed。

本平台实际可安装并验证的是 `evalscope==0.17.1`。PyPI 中不存在
`evalscope==1.17.1`；测试脚本会把每个二进制 WAV 响应作为一个完成单元，
保留 EvalScope 的调度、SQLite、延迟分位数和请求吞吐统计，并额外计算
TTS 的 RTF、音频吞吐、静音比例及 WAV 结构。

EvalScope 输出中的 token throughput、TTFT、TPOT、ITL 不适用于非流式
二进制 TTS 响应，本报告不使用这些字段作优化判断。

## Perfetto 定位依据

基线 trace 显示 Flow 是端到端主瓶颈，平均占 59.77%。两个请求合计出现：

- `aten::scaled_dot_product_attention`：880 次；
- FP32 batch GEMM SDAA kernel：880 次，总计约 1379.2 ms；
- math SDPA 的 softmax、mask 和 dtype 转换也产生额外 kernel 与调度开销。

优化后，原 FP32 batch GEMM 和 math softmax 路径消失，变为 FP16/BF16
SDAA FlashAttention kernel。Perfetto 分段结果表明：

| 阶段 | 基线 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| 完整请求 | 4098.2 ms | 3209.4 ms | -21.69% |
| Flow/DiT | 2449.4 ms | 1566.1 ms | -36.06% |
| HiFT | 297.7 ms | 285.0 ms | -4.26% |
| 未标注前后处理及 LLM | 1350.7 ms | 1357.9 ms | +0.53% |

收益几乎全部来自 Flow，未标注部分基本不变，说明不是输入变短、少生成音频
或偶然的 LLM 采样差异造成的。

## 代码实现与安全回退

实现位于：

```text
cosyvoice/flow/DiT/modules.py
cosyvoice/sdaa_ops/
```

`cosyvoice/sdaa_ops/` 中同时保存扩展源码、构建文件和当前 Python 3.10 /
TECO 3.2.0 二进制。只有满足以下条件时才使用 FlashAttention：

- `COSYVOICE_SDAA_FLOW_FLASH_ATTN=1`；
- SDAA 设备；
- FP16 或 BF16；
- `torch.no_grad()` 推理；
- Q/K/V shape 相同；
- API 单请求经 classifier-free guidance 复制后的 batch size 为 2；
- attention mask 已验证为全 `true`。

训练、其他设备、其他 dtype、一般 padded batch 或扩展缺失时不会静默使用
不兼容算子，而是回退或明确报错。关闭优化只需：

```bash
export COSYVOICE_SDAA_FLOW_FLASH_ATTN=0
```

## 从源码构建 SDAA 扩展

```bash
DOCKER=/opt/kube/bin/docker

$DOCKER exec -w /workspace/CosyVoice/cosyvoice/sdaa_ops \
  yy-cosyvoice-sdaa-vllm bash -lc '
    source /opt/tecoai/setvars.sh
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate vllm_env_py310
    python setup.py build_ext --inplace
  '
```

构建成功后会生成
`sdaa_mm_encoder_fa_poc_ext.cpython-310-x86_64-linux-gnu.so`。

## 启动优化后的 API 服务

API Key 只保存在仓库内被忽略且权限为 `600` 的环境文件，不在命令行中
展开：

宿主机终端 A：

```bash
cd /mnt/nvme/application/yangyu/CosyVoice
python3 tools/rotate_cosyvoice_api_key.py
chmod 600 cosyvoice_sdaa_vllm_api.env
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
export COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH=0
export COSYVOICE_SDAA_HIFT_REFLECTION_PAD=0
export COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE=0
export COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE=0
export COSYVOICE_SDAA_FLOW_STEPS=10
export COSYVOICE_VLLM_SDAA_GRAPH=0
exec python -m api_server.main
```

这是第一轮历史复现配置。服务以前台方式运行，启动过程和请求日志直接显示在
终端 A。出现 `Application startup complete` 后，在宿主机终端 B 检查：

```bash
curl --fail-with-body http://127.0.0.1:8021/health
curl --fail-with-body http://127.0.0.1:8021/ready
```

## 安装并运行 EvalScope perf

工具只安装到仓库内被忽略的运行目录：

```bash
DOCKER=/opt/kube/bin/docker

$DOCKER exec -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm bash -lc '
    EVALSCOPE_DIR=cosyvoice_api_outputs/perf_tools/evalscope_py
    mkdir -p "${EVALSCOPE_DIR}"
    /root/miniconda3/envs/vllm_env_py310/bin/python -m pip install \
      --target "${EVALSCOPE_DIR}" --no-deps \
      evalscope==0.17.1 jsonlines immutabledict langdetect \
      pandas word2number tabulate
  '
```

运行优化版测试：

```bash
$DOCKER exec -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  /root/miniconda3/envs/vllm_env_py310/bin/python \
  tools/cosyvoice_evalscope_perf.py \
    --prompt-file tools/cosyvoice_perf_prompts.txt \
    --outputs-dir \
      cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized/evalscope_c1 \
    --api-key-env-file cosyvoice_sdaa_vllm_api.env \
    --number 10 \
    --parallel 1 \
    --warmup 2 \
    --seed 20260726 \
    --run-name cosyvoice-sdaa-flash-attn
```

要重采基线，使用相同命令和语料，只需以
`COSYVOICE_SDAA_FLOW_FLASH_ATTN=0` 重启服务，并把输出目录改为
`baseline/evalscope_c1`。

## 运行 Perfetto

先确认逻辑设备 `18` 空闲，然后分别运行基线与优化版：

```bash
DOCKER=/opt/kube/bin/docker
ROOT=cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726

$DOCKER exec \
  -e SDAA_VISIBLE_DEVICES=18 \
  -e COSYVOICE_SDAA_FLOW_FLASH_ATTN=0 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  bash tools/run_cosyvoice_perfetto_sdaa.sh \
    --prompt-file tools/cosyvoice_perf_prompts.txt \
    --outputs-dir "${ROOT}/baseline/perfetto" \
    --warmup 1 --active 2 --seed 20260726 --tag baseline

$DOCKER exec \
  -e SDAA_VISIBLE_DEVICES=18 \
  -e COSYVOICE_SDAA_FLOW_FLASH_ATTN=1 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm \
  bash tools/run_cosyvoice_perfetto_sdaa.sh \
    --prompt-file tools/cosyvoice_perf_prompts.txt \
    --outputs-dir "${ROOT}/optimized/perfetto" \
    --warmup 1 --active 2 --seed 20260726 --tag optimized
```

分析 trace：

```bash
python3 tools/analyze_perfetto_trace.py \
  "${ROOT}/baseline/perfetto/baseline.perfetto.json.gz" \
  --output "${ROOT}/baseline/perfetto/baseline.analysis.json"

python3 tools/analyze_perfetto_trace.py \
  "${ROOT}/optimized/perfetto/optimized.perfetto.json.gz" \
  --output "${ROOT}/optimized/perfetto/optimized.analysis.json"

python3 tools/compare_cosyvoice_perf_results.py --root "${ROOT}"
```

`cosyvoice_perfetto_profile.py` 把 trace 直接导出为
`.perfetto.json.gz`，不先落盘未压缩 JSON。压缩 trace 可以直接在
Perfetto UI 中打开，`analyze_perfetto_trace.py` 也会流式解压分析。

每轮实验结束后，仅保留基线和达到明显端到端收益的优化 trace。没有明显
收益、结果退化或实验失败的 trace 应删除，只保留小体积的分析 JSON 和实验
结论，避免数百 MB 的无效文件长期占用磁盘。

## 音频正确性与人工试听

自动 ASR 语义回归：

```bash
$DOCKER exec \
  -e SDAA_VISIBLE_DEVICES=18 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa-vllm bash -lc '
    source /opt/tecoai/setvars.sh
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate vllm_env_py310
    python tools/cosyvoice_qwen3_asr_semantic_check.py \
      --audio \
        cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized/evalscope_c1/audio \
      --expected-text-file tools/cosyvoice_perf_prompts.txt \
      --model /tecogpfs/models/Qwen/Qwen3-ASR-0___6B \
      --gpu-memory-utilization 0.8 \
      --output-jsonl \
        cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized/evalscope_c1/asr.jsonl \
      --summary-json \
        cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized/evalscope_c1/asr_summary.json
  '
```

供人工试听的优化版音频：

```text
/mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/optimized/evalscope_c1/audio/
```

基线对照音频：

```text
/mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/baseline/evalscope_c1/audio/
```

建议人工重点检查“太初”、`SDAA`、`CosyVoice API` 的发音，以及韵律与音色
是否符合业务偏好。数值上 FlashAttention 与 math SDPA 不会逐样本完全相同；
两条同长度对照的频谱余弦相似度为 0.966 和 0.989，但最终正确性应以完整
语义、无退化、结构指标和人工听感共同判断。

## 结果文件

所有代码、运行配置、音频与 trace 均在 `CosyVoice/` 内；不保存通用
压缩包，Perfetto trace 使用允许的 `.json.gz`：

```text
cosyvoice_api_outputs/perf/cosyvoice_sdaa_optimization_20260726/
├── comparison_summary.json
├── baseline/
│   ├── evalscope_c1/
│   └── perfetto/baseline.perfetto.json.gz
└── optimized/
    ├── evalscope_c1/
    └── perfetto/optimized.perfetto.json.gz
```

其中 `comparison_summary.json` 是机器可读的最终 A/B 汇总。
