# CosyVoice API：SDAA 双终端部署与验收

本文只描述当前 TECO SDAA 服务器上的 PyTorch 和 vLLM backend。API
调用逻辑、字段和代码职责见 [`README.md`](README.md)。

## 1. 固定路径与端口

| 项目 | 值 |
| --- | --- |
| 宿主机仓库 | `/mnt/nvme/application/yangyu/CosyVoice` |
| 容器内仓库 | `/workspace/CosyVoice` |
| PyTorch 容器 | `yy-cosyvoice-sdaa` |
| vLLM 容器 | `yy-cosyvoice-sdaa-vllm` |
| PyTorch 端口 | `127.0.0.1:8020` |
| vLLM 端口 | `127.0.0.1:8021` |
| PyTorch 模型 | `/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B` |
| vLLM 宿主模型 | `/workspace/model` |
| API Key 文件 | `/mnt/nvme/application/yangyu/CosyVoice/cosyvoice_sdaa_vllm_api.env` |
| 人工验收输出 | `/mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/manual` |
| Docker CLI | `/opt/kube/bin/docker` |

当前 vLLM 容器的 `Cmd` 是 `/bin/bash`；已有 PyTorch 容器是旧实例，
`Cmd` 仍为 `sleep infinity`。两者的 PID 1 都只负责保持容器存活，实际服务
统一通过 `docker exec -it` 进入 backend 环境后以前台方式启动。仓库中的
`Dockerfile.sdaa` 已使用 `/bin/bash`，以后基于它新建的容器不再显示
`sleep infinity`。

## 2. 一次性准备

在宿主机执行：

```bash
cd /mnt/nvme/application/yangyu/CosyVoice
git submodule update --init --recursive third_party/Matcha-TTS
python3 tools/rotate_cosyvoice_api_key.py
chmod 600 /mnt/nvme/application/yangyu/CosyVoice/cosyvoice_sdaa_vllm_api.env
mkdir -p /mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/manual
```

Key 由服务 shell 和测试脚本直接读取该文件，不需要复制到命令行。

## 3. PyTorch backend

### 终端 A：启动服务

在宿主机执行：

```bash
/opt/kube/bin/docker start yy-cosyvoice-sdaa
/opt/kube/bin/docker exec -it \
  -e SDAA_VISIBLE_DEVICES=16 \
  -w /workspace/CosyVoice \
  yy-cosyvoice-sdaa \
  /bin/bash
```

进入容器后执行：

```bash
source /opt/tecoai/setvars.sh
conda activate vllm_env_py310
cd /workspace/CosyVoice
set -a
source /workspace/CosyVoice/cosyvoice_sdaa_vllm_api.env
set +a
export COSYVOICE_MODEL_DIR=/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
export COSYVOICE_MODEL_ALIAS=cosyvoice3-0.5b
export COSYVOICE_HOST=127.0.0.1
export COSYVOICE_PORT=8020
export COSYVOICE_LOAD_VLLM=false
export COSYVOICE_FP16=false
export COSYVOICE_MAX_CONCURRENCY=1
export COSYVOICE_MAX_QUEUE_SIZE=16
export COSYVOICE_REQUEST_TIMEOUT_SECONDS=900
export COSYVOICE_DEFAULT_SEED=0
export COSYVOICE_QUALITY_CHECK_ENABLED=true
export COSYVOICE_SDAA_FLOW_FLASH_ATTN=1
export COSYVOICE_SDAA_FLOW_FUSED_NORM=1
export COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE=1
export COSYVOICE_SDAA_FLOW_STEPS=8
export COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH=1
export COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK=1
export COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE=1
export COSYVOICE_SDAA_HIFT_F0_FP32=1
export COSYVOICE_SDAA_HIFT_CONV_STFT=1
export COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE=1
export COSYVOICE_SDAA_HIFT_REFLECTION_PAD=1
exec python -m api_server.main
```

服务日志出现 `Application startup complete` 后保持终端 A 不动。

### 终端 B：请求与验收

在宿主机新开一个 shell：

```bash
cd /mnt/nvme/application/yangyu/CosyVoice
curl --fail-with-body http://127.0.0.1:8020/health
curl --fail-with-body http://127.0.0.1:8020/ready
python3 tools/test_cosyvoice_api.py \
  --backend pytorch \
  --voice-mode zero_shot \
  --voice default \
  --response-format wav \
  --seed 0 \
  --output-dir /mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/manual/pytorch_zero_shot_no_instruction_wav
```

## 4. vLLM backend

### 终端 A：启动服务

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
export COSYVOICE_SDAA_FLOW_FUSED_NORM=1
export COSYVOICE_SDAA_FLOW_FUSED_FFN=1
export COSYVOICE_SDAA_FLOW_FUSED_FFN_MIN_ROWS=800
export COSYVOICE_SDAA_FLOW_FUSED_GEMM=1
export COSYVOICE_SDAA_FLOW_FUSED_GEMM_MIN_ROWS=512
export COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE=1
export COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH=1
export COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK=1
export COSYVOICE_SDAA_HIFT_REFLECTION_PAD=1
export COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE=1
export COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE=1
export COSYVOICE_SDAA_HIFT_F0_FP32=1
export COSYVOICE_SDAA_HIFT_CONV_STFT=1
export COSYVOICE_SDAA_FLOW_STEPS=8
export COSYVOICE_VLLM_SDAA_GRAPH=1
exec python -m api_server.main
```

命令从 `cosyvoice_sdaa_vllm_api.env` 读取 API Key，不在终端历史中展开
Key。`exec python -m api_server.main` 是前台进程，模型加载、SDAA graph
capture、端口监听和请求错误都会直接显示在终端 A。日志出现
`Application startup complete` 后保持终端 A 不动；按 `Ctrl-C` 停止服务。
当前配置启用单序列 `FULL_DECODE_ONLY` graph，变长 prefill 保持 eager。
Flow FFN 在输入矩阵不少于 800 行时使用 SDAA `ffnv2` 融合算子；短矩阵仍
走原始 `linear -> GELU -> linear`，避免融合算子在小 shape 上回退。该路径
只在 vLLM/SDAA 环境中开启，因为融合算子由镜像内的 `vllm_sdaa` 扩展提供。
Flow Attention 的 Q/K/V/out 投影在输入矩阵不少于 512 行时使用 SDAA
`blas_gemm_fusion`。服务首次预热时会将 FP16 转置权重转换为算子要求的布局，
随后缓存在对应 Linear 层的非持久 buffer 中；短矩阵、训练模式和非 SDAA
环境自动走原始 `nn.Linear`。该路径同样依赖 vLLM 镜像内的 `vllm_sdaa` 扩展。

### 终端 B：请求与验收

在宿主机新开一个 shell：

```bash
cd /mnt/nvme/application/yangyu/CosyVoice
curl --fail-with-body http://127.0.0.1:8021/health
curl --fail-with-body http://127.0.0.1:8021/ready
```

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

zero-shot、无 instructions、PCM：

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

SFT 测试：

```bash
cd /mnt/nvme/application/yangyu/CosyVoice
python3 tools/test_cosyvoice_api.py \
  --backend vllm \
  --voice-mode sft \
  --response-format wav \
  --seed 0 \
  --output-dir /mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/manual/vllm_sft_no_instruction_wav
```

当前模型没有 `spk2info.pt`，`/v1/audio/voices` 只返回 `default`
zero-shot，因此当前 SFT 命令会在推理前报告没有注册 SFT 音色。以后注册
真实 SFT `spk_id` 并重启服务后，同一命令会自动选择第一个 SFT 音色。
`instructions` 只允许和 zero-shot 组合。

每次命令只测试一个组合并写入独立目录。脚本会等待 `/ready`，再核对
`/v1/audio/voices` 返回的音色 mode，因此不会把 SFT、zero-shot 和
instruction 推理路径混在一次测试里。

2026-07-29 实测中，zero-shot 无 instructions WAV、zero-shot 有
instructions WAV 和 zero-shot PCM 三个目录均为 `ok: true`。SFT 目录为
`ok: false`、`speech_requests: []`，原因是当前模型没有注册 SFT 音色；
该目录没有任何音频文件。

## 5. 输出文件与判断标准

每次调用显式指定独立的 `--output-dir`，且该目录必须位于
`cosyvoice_api_outputs` 下。典型文件如下：

| 文件 | 含义 |
| --- | --- |
| `health.json` | 健康端点结果 |
| `models.json` | 模型列表结果 |
| `voices.json` | 音色列表结果 |
| `vllm_zero_shot_no_instruction.wav` | zero-shot、无 instructions、WAV |
| `vllm_zero_shot_instruction.wav` | zero-shot、有 instructions、WAV |
| `vllm_zero_shot_no_instruction.pcm` | zero-shot、无 instructions、裸 PCM |
| `vllm_sft_no_instruction.wav` | 注册 SFT 音色后的 SFT WAV |
| `*.error.json` | HTTP 或质量门禁失败的结构化错误，不是音频 |
| `*.headers.txt` | 对应语音请求的状态与响应头 |
| `request.json` | 本次唯一语音请求的参数 |
| `results.json` | 完整汇总及每段音频的结构指标 |

脚本只会在 HTTP 200 且 `Content-Type` 正确时写 `.wav` 或 `.pcm`，并验证
WAV 为 24 kHz、单声道、PCM16、有有效采样且非全静音。HTTP 错误保存为
`.error.json`，不会生成错误音频。`results.json` 顶层 `ok` 为 `true` 表示
本次单一组合通过；`audio_quality_failed` 会让本次命令返回非零状态。
默认终端只显示类似 curl 的两行传输统计，不再打印整份 JSON。需要在终端
查看完整结果时显式追加 `--verbose`；无论是否追加，`results.json` 都会
保存完整内容。

人工试听前先查看结果清单：

```bash
cd /mnt/nvme/application/yangyu/CosyVoice
python3 -m json.tool cosyvoice_api_outputs/manual/vllm_zero_shot_no_instruction_wav/results.json
find /mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/manual/vllm_zero_shot_no_instruction_wav \
  -maxdepth 1 \
  -type f \
  -printf '%f %s bytes\n'
```

## 6. 停止服务

终端 A 按 `Ctrl-C` 停止前台服务。需要停止常驻容器时在宿主机执行：

```bash
/opt/kube/bin/docker stop -t 15 yy-cosyvoice-sdaa
/opt/kube/bin/docker stop -t 15 yy-cosyvoice-sdaa-vllm
```

## 7. SDAA 与 ONNX Runtime

观察卡状态：

```bash
watch -n 0.5 /opt/tecoai/bin/teco-smi
```

CosyVoice 主模型、LLM、flow 和 vocoder 通过 PyTorch SDAA 或 SDAA vLLM
运行在卡上。`onnxruntime` 仅执行 CosyVoice 文本前处理中的 ONNX 辅助图；
CUDA 的 `onnxruntime-gpu` 不能驱动 SDAA，也不代表主推理退回 CPU。当前
SDAA 环境应保留 CPU 版 `onnxruntime`，不要安装 CUDA provider。

## 8. 性能测试

固定测试脚本和性能文档位于：

- `tools/cosyvoice_evalscope_perf.py`
- `tools/benchmark_cosyvoice_vllm_e2e_ab.py`
- `tools/cosyvoice_perfetto_profile.py`
- `tools/cosyvoice_qwen3_asr_semantic_check.py`
- `tools/analyze_perfetto_trace.py`
- `tools/analyze_perfetto_phases.py`
- `perf_md/cosyvoice_sdaa_vllm_optimization_report.md`
- `perf_md/cosyvoice_sdaa_vllm_optimization_round2.md`
- `perf_md/cosyvoice_sdaa_vllm_optimization_round3.md`
- `perf_md/cosyvoice_sdaa_vllm_optimization_round4.md`
- `perf_md/cosyvoice_sdaa_vllm_optimization_round5.md`
- `perf_md/cosyvoice_sdaa_vllm_e2e_latency_audit_20260730.md`

性能测试生成的报告、Perfetto trace 和候选音频统一放在仓库内的
`cosyvoice_api_outputs`。未形成有效优化结论的 trace 应删除，不提交压缩包。

真实 vLLM HTTP 端到端对称 A/B 在容器中的 `/workspace/CosyVoice`
执行。它会临时重启当前 API 服务，并在正常结束或异常时恢复最终优化配置：

```bash
cd /workspace/CosyVoice

python tools/benchmark_cosyvoice_vllm_e2e_ab.py \
  --output-dir cosyvoice_api_outputs/perf/cosyvoice_sdaa_e2e_audit_recheck \
  --prompt-file tools/cosyvoice_perf_prompts.txt \
  --variants optimized optimized_trusted_mask_conv_stft \
  --cycles 2 \
  --number 10 \
  --warmup 2 \
  --seed 20260726 \
  --port 8021 \
  --startup-timeout 180
```

输出目录必须为空或不存在。脚本不会打印或保存 API key，终端只显示精简
结果，完整统计、日志和试听 WAV 都写入指定输出目录。

当前服务还启用了 `COSYVOICE_SDAA_FLOW_FUSED_NORM=1`。它将 Flow DiT
中的门控残差、LayerNorm 和 AdaLN scale/shift 合并为 TECO LMK SDAA
算子，并跨相邻 block 复用已经生成的归一化结果。关闭该开关会自动回到
原始 PyTorch 算子序列，便于回归比较。

`COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE=1` 在每个 Flow 请求开始时只计算
一次 RoPE 的 sine/cosine，供 22 个 DiT block 和 8 个 Flow step 复用。
该路径在 20/20 个同 seed 配对中生成逐字节完全一致的 WAV。

`COSYVOICE_SDAA_HIFT_F0_FP32=1` 让 CausalHiFT 的 F0 predictor 使用
SDAA FP32，避免 float64 路径引发的 CPU fallback 和高开销 D2H/H2D
拷贝。该开关只在 SDAA、推理模式下生效；关闭时保留原始 float64 路径。

`COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK=1` 只用于当前单请求、非流式的
API 推理路径。该路径构造的 classifier-free guidance mask 全部有效，因此
可以省去每请求的 SDAA `all().item()` 同步；训练、流式推理和非 SDAA 路径
仍保留原始检查。

`COSYVOICE_SDAA_HIFT_CONV_STFT=1` 使用固定 Hann Fourier basis 的 SDAA
`conv1d`/`conv_transpose1d` 实现 HiFT 的 `n_fft=16, hop=4` STFT/ISTFT，
避免小 FFT 路径中的设备往返。关闭开关会回到原始 `torch.stft/istft`。
该路径的 40 条 ASR 与原路径逐条一致；完整结果见
`perf_md/cosyvoice_sdaa_vllm_optimization_round6.md`。
