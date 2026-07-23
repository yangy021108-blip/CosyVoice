# CosyVoice HTTP API 中文使用指南

本文说明如何启动、调用、测试和部署仓库中的 CosyVoice API v0.1。

## 1. 当前能力和限制

服务提供以下接口：

| 方法 | 路径 | 鉴权 | 用途 |
| --- | --- | --- | --- |
| `GET` | `/health` | 无 | 检查 HTTP 进程是否存活 |
| `GET` | `/ready` | 无 | 检查模型是否加载完成 |
| `GET` | `/v1/models` | Bearer API Key | 查询模型和输出格式 |
| `GET` | `/v1/audio/voices` | Bearer API Key | 查询服务端注册的音色 |
| `POST` | `/v1/audio/speech` | Bearer API Key | 文本合成 WAV 或 PCM |

当前版本支持：

- CosyVoice3 zero-shot 音色；
- CosyVoice 模型自带的 SFT speaker；
- zero-shot 音色的 `instructions` 风格控制；
- WAV 和裸 PCM16 输出；
- API Key、请求 ID、有限队列、推理超时和统一错误格式。

当前版本不支持流式输出、MP3、Opus、SSML、任意采样率、pitch、时间戳或客户端上传参考音频。未知参数会直接返回 400；传入 `stream_format` 也会返回 400。

生产配置暂时保持：

```bash
export COSYVOICE_MAX_CONCURRENCY=1
```

同一个模型实例尚未完成多线程 GPU 推理安全性验证。需要扩容时，优先使用多实例、多 GPU 或外部请求队列。

## 2. 使用现有服务器环境快速启动

现有环境位于容器 `vllm_env_yy`，Python 路径为 `/opt/conda/envs/cosyvoice/bin/python`，仓库路径为 `/SharedData/yangyu/CosyVoice`。

登录 GPU 服务器后进入容器：

```bash
docker exec -it vllm_env_yy bash
cd /SharedData/yangyu/CosyVoice
```

选择一张空闲 GPU，并设置服务参数：

```bash
export CUDA_VISIBLE_DEVICES=4
export COSYVOICE_HOST=127.0.0.1
export COSYVOICE_PORT=8000
export COSYVOICE_API_KEY='替换成随机且足够长的密钥'
export COSYVOICE_MAX_CONCURRENCY=1
export COSYVOICE_MAX_QUEUE_SIZE=16
```

启动服务：

```bash
/opt/conda/envs/cosyvoice/bin/python -m api_server.main
```

普通 PyTorch backend 必须使用 `transformers==4.51.3`。启动前先检查当前
环境，下面每条测试命令都是单独一行，可以直接复制执行：

```bash
/opt/conda/envs/cosyvoice/bin/python -c "import torch, transformers; print('torch=', torch.__version__, 'transformers=', transformers.__version__)"
```

如果输出的 Transformers 不是 `4.51.3`，执行：

```bash
/opt/conda/envs/cosyvoice/bin/python -m pip install --upgrade "transformers==4.51.3" "tokenizers>=0.21,<0.22"
```

不要在普通 PyTorch 环境中安装 `requirements-vllm.txt`。其中的
`transformers==4.57.1` 只用于 `COSYVOICE_LOAD_VLLM=true`，应放在独立环境或
镜像中。服务启动时会检查版本，因为错误的 Transformers 版本可能生成 WAV，
但语音内容会与 `input` 不一致并且断断续续。

看到以下信息后说明模型已经加载：

```text
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
```

这个启动方式仅监听容器内部回环地址，适合先做功能验证。另开一个容器终端进行检查：

```bash
docker exec -it vllm_env_yy bash

curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

预期响应：

```json
{"status":"ok"}
```

```json
{"status":"ready","model":"cosyvoice3-0.5b","sample_rate":24000}
```

注意：已经创建的 Docker 容器不能事后增加 `-p` 端口映射。需要让其他机器访问时，应使用第 8 节的 Docker 部署方式重新创建服务容器，或者在启动容器时提前配置 host network/端口映射。

## 3. 调用语音合成接口

### 3.1 查询模型和音色

```bash
export API_KEY='与服务端 COSYVOICE_API_KEY 相同的值'

curl --fail-with-body http://127.0.0.1:8000/v1/models -H "Authorization: Bearer ${API_KEY}"
curl --fail-with-body http://127.0.0.1:8000/v1/audio/voices -H "Authorization: Bearer ${API_KEY}"
```

### 3.2 生成 WAV

```bash
curl --fail-with-body --request POST http://127.0.0.1:8000/v1/audio/speech -H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json" -d '{"model":"cosyvoice3-0.5b","input":"你好，这是一次 CosyVoice API 测试。","voice":"default","response_format":"wav","speed":1.0}' --output result.wav
```

WAV 为模型原生采样率的单声道 16-bit 音频；当前 CosyVoice3 模型为 24 kHz。

### 3.3 使用风格指令

`instructions` 只支持 zero-shot 音色：

```bash
curl --fail-with-body --request POST http://127.0.0.1:8000/v1/audio/speech -H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json" -d '{"model":"cosyvoice3-0.5b","input":"欢迎使用我们的语音合成服务。","voice":"default","response_format":"wav","speed":1.0,"instructions":"请用四川话、开心地说这句话"}' --output instructed.wav
```

CosyVoice 的语音 token 生成包含随机采样。不传 `seed` 时，服务从经过当前模型验证的默认 seed 开始；如果结果明显过短、相对文本异常长或大部分接近静音，会自动换下一个 seed 重试。README 中的 `result.wav` 和 `instructed.wav` 命令不需要增加新参数。

如果需要严格复现某一次结果，可以显式传入：

```json
{
  "seed": 2
}
```

显式传入 `seed` 时不会自动换 seed。所有自动尝试都未通过质量检查时，接口返回 503 `audio_quality_failed`，不会把已知退化的音频返回给用户。

给 SFT 音色传入 `instructions` 会返回 400 `unsupported_parameter`，并且不会进入 GPU 队列。

### 3.4 获取裸 PCM

```bash
curl --fail-with-body --request POST http://127.0.0.1:8000/v1/audio/speech -H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json" -d '{"model":"cosyvoice3-0.5b","input":"这是 PCM 输出测试。","voice":"default","response_format":"pcm"}' --output result.pcm
```

PCM 格式为单声道、有符号 16-bit little-endian、模型原生采样率。服务器或 Docker 容器没有图形、声卡设备时，`ffplay` 会报 `XDG_RUNTIME_DIR`、SDL 或音频设备错误；这不表示 PCM 损坏。此时先把 24 kHz PCM 转成 WAV，再下载到本机播放：

```bash
ffmpeg -y -f s16le -ar 24000 -ac 1 -i result.pcm result_pcm.wav
```

如果当前机器有可用的音频设备，可以关闭 ffplay 的视频窗口直接播放：

```bash
ffplay -nodisp -autoexit -f s16le -ar 24000 -ac 1 result.pcm
```

### 3.5 Python 客户端示例

```python
from pathlib import Path

import requests

response = requests.post(
    "http://127.0.0.1:8000/v1/audio/speech",
    headers={"Authorization": "Bearer 替换成你的密钥"},
    json={
        "model": "cosyvoice3-0.5b",
        "input": "你好，这是 Python 客户端请求。",
        "voice": "default",
        "response_format": "wav",
        "speed": 1.0,
    },
    timeout=900,
)
response.raise_for_status()
Path("result.wav").write_bytes(response.content)

print("request_id:", response.headers.get("X-Request-ID"))
print("audio_duration:", response.headers.get("X-Audio-Duration"))
print("rtf:", response.headers.get("X-Real-Time-Factor"))
```

接口采用 OpenAI 风格 JSON，但只实现本文列出的参数，不应假设它与 OpenAI SDK 的所有语音参数完全兼容。

## 4. 请求参数和响应

`POST /v1/audio/speech` 的请求字段如下：

| 字段 | 必填 | 取值/限制 | 说明 |
| --- | --- | --- | --- |
| `model` | 是 | 当前默认 `cosyvoice3-0.5b` | 服务端模型别名 |
| `input` | 是 | 非空，默认最多 2000 字符 | 待合成文本 |
| `voice` | 是 | `/v1/audio/voices` 返回的 ID | 服务端注册音色 |
| `response_format` | 否 | `wav`、`pcm`，默认 `wav` | 音频封装格式 |
| `speed` | 否 | `0.5` 到 `2.0`，默认 `1.0` | 语速 |
| `instructions` | 否 | 最多 1000 字符 | 仅 zero-shot 音色可用 |
| `seed` | 否 | 0 到 4294967295 | 固定语音 token 采样；显式设置时关闭自动换 seed |

成功响应包含以下诊断头：

| 响应头 | 含义 |
| --- | --- |
| `X-Request-ID` | 请求唯一标识，可由客户端传入同名请求头 |
| `X-Audio-Sample-Rate` | 输出采样率 |
| `X-Audio-Channels` | 声道数，当前为 1 |
| `X-Audio-Duration` | 音频时长，单位秒 |
| `X-Queue-Wait-Ms` | 等待 GPU 推理名额的时间 |
| `X-Inference-Latency-Ms` | 模型推理和音频收集耗时 |
| `X-Real-Time-Factor` | 推理耗时除以音频时长 |
| `X-Generation-Seed` | 最终返回音频实际使用的 seed |
| `X-Quality-Retry-Count` | 质量检查触发的重试次数 |
| `X-Silent-Frame-Ratio` | 20 ms 帧中低于 -50 dBFS 的比例 |

## 5. 配置音色

客户端只能提交音色 ID，不能提交服务器文件路径。所有音色由服务端的 `api_server/voices.json` 管理，修改后需要重启服务。

### 5.1 zero-shot 音色

```json
{
  "my_voice": {
    "name": "My zero-shot voice",
    "mode": "zero_shot",
    "prompt_text": "You are a helpful assistant.<|endofprompt|>参考音频中实际说出的文字。",
    "prompt_audio": "asset/my_prompt.wav"
  }
}
```

要求：

- `prompt_audio` 必须是仓库目录内的服务端文件；
- `prompt_text` 在 CosyVoice3 系统前缀之后的部分，必须与参考音频实际说出的文字逐字一致，包括标点；
- 音色 ID 只能包含字母、数字、下划线、点和短横线；
- 服务启动时会预处理并缓存 zero-shot speaker，普通请求不会重复处理参考音频；
- 修改 `prompt_text` 或 `prompt_audio` 后必须重启 API，否则进程仍会使用旧缓存；
- 首次核对音色时不要传 `instructions`，先排除方言、情绪等风格控制对听感的影响。

参考音频应为单人、清晰、无背景音乐且没有过长静音。音量过低可能使生成结果大段静音。可以用下面的单行命令去除首尾静音、标准化响度，并转换为 16 kHz 单声道 PCM WAV：

```bash
ffmpeg -y -i asset/my_prompt.wav -af "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,areverse,loudnorm=I=-24:LRA=7:TP=-3" -ar 16000 -ac 1 -c:a pcm_s16le asset/my_prompt_clean.wav
```

### 5.2 SFT 音色

```json
{
  "sft_female": {
    "name": "Built-in female voice",
    "mode": "sft",
    "spk_id": "模型中真实存在的 speaker ID"
  }
}
```

服务启动时会调用模型的 `list_available_spks()` 验证 `spk_id`。不存在的 speaker 会导致模型保持未就绪，`/ready` 返回 503。

## 6. 环境变量

| 环境变量 | 默认值 | 建议 |
| --- | --- | --- |
| `COSYVOICE_MODEL_ALIAS` | `cosyvoice3-0.5b` | 对外暴露的模型名 |
| `COSYVOICE_MODEL_DIR` | `pretrained_models/Fun-CosyVoice3-0.5B` | 模型目录 |
| `COSYVOICE_VOICES_FILE` | `api_server/voices.json` | 音色注册文件 |
| `COSYVOICE_API_KEY` | 未设置 | 非回环监听时必须设置 |
| `COSYVOICE_HOST` | `127.0.0.1` | 容器端口映射时设为 `0.0.0.0` |
| `COSYVOICE_PORT` | `8000` | 服务端口 |
| `COSYVOICE_ALLOW_UNAUTHENTICATED` | `false` | 仅隔离测试环境可显式开启 |
| `COSYVOICE_MAX_TEXT_CHARACTERS` | `2000` | 输入长度上限 |
| `COSYVOICE_MAX_CONCURRENCY` | `1` | 当前生产环境保持 1 |
| `COSYVOICE_MAX_QUEUE_SIZE` | `16` | 排队请求上限 |
| `COSYVOICE_REQUEST_TIMEOUT_SECONDS` | `600` | HTTP 推理超时 |
| `COSYVOICE_FP16` | `false` | 是否让 backend 使用 FP16 |
| `COSYVOICE_LOAD_VLLM` | `false` | 是否启用 vLLM backend |
| `COSYVOICE_DEFAULT_SEED` | `2` | 未传 `seed` 时第一次生成使用的 seed |
| `COSYVOICE_QUALITY_CHECK_ENABLED` | `true` | 是否拒绝明显退化的音频并自动重试 |
| `COSYVOICE_QUALITY_MAX_RETRIES` | `2` | 首次生成之后最多再尝试的次数 |

无 API Key 时绑定 `0.0.0.0`、主机名或其他非回环地址会拒绝启动。只有主动设置 `COSYVOICE_ALLOW_UNAUTHENTICATED=true` 才能绕过该保护，不建议在共享内网或公网使用。

## 7. 队列、超时和错误处理

可接纳的运行中和排队中请求总数为：

```text
COSYVOICE_MAX_CONCURRENCY + COSYVOICE_MAX_QUEUE_SIZE
```

典型错误：

| HTTP 状态 | `code` | 含义 |
| --- | --- | --- |
| 400 | `invalid_parameter` | 字段类型、范围或未知字段错误 |
| 400 | `unsupported_parameter` | 参数组合不支持，例如 SFT + instructions |
| 401 | `invalid_api_key` | API Key 缺失或错误 |
| 404 | `model_not_found` / `voice_not_found` | 模型或音色不存在 |
| 413 | `input_too_large` | 输入超过长度限制 |
| 429 | `queue_full` | 推理队列已满 |
| 503 | `model_not_ready` | 模型加载失败或尚未完成 |
| 503 | `audio_quality_failed` | 多次生成均被判断为明显退化；可重试请求或指定其他 seed |
| 504 | `inference_timeout` | HTTP 请求等待推理超时 |

错误响应格式：

```json
{
  "error": {
    "message": "Voice 'missing' was not found",
    "type": "invalid_request_error",
    "param": "voice",
    "code": "voice_not_found"
  },
  "request_id": "..."
}
```

GPU 推理线程无法安全强制取消。HTTP 返回 504 或客户端断开后，已经进入后台的推理可能继续执行；服务会继续占用对应 capacity，直到后台任务真正结束，避免绕过队列上限。

## 8. Docker 部署

### 8.1 普通 PyTorch backend

从仓库根目录构建：

```bash
docker build -t cosyvoice-api:torch -f api_server/Dockerfile api_server
```

启动并映射给宿主机/内网：

```bash
docker run -d --name cosyvoice-api \
  --gpus device=4 \
  -p 8000:8000 \
  -v "$PWD:/workspace/CosyVoice" \
  -e COSYVOICE_HOST=0.0.0.0 \
  -e COSYVOICE_PORT=8000 \
  -e COSYVOICE_API_KEY='替换成随机且足够长的密钥' \
  -e COSYVOICE_MAX_CONCURRENCY=1 \
  cosyvoice-api:torch
```

查看日志和停止服务：

```bash
docker logs -f cosyvoice-api
docker stop cosyvoice-api
docker rm cosyvoice-api
```

### 8.2 vLLM backend

vLLM 镜像需要显式构建额外依赖层：

```bash
docker build --build-arg INSTALL_VLLM=true \
  -t cosyvoice-api:vllm -f api_server/Dockerfile api_server
```

启动时增加：

```bash
-e COSYVOICE_LOAD_VLLM=true
```

对应依赖固定为 `vllm==0.11.0`、`transformers==4.57.1` 和 `numpy==1.26.4`。普通镜像没有安装 vLLM；在普通镜像中强行设置 `COSYVOICE_LOAD_VLLM=true` 会得到明确的启动错误。

## 9. 测试和真实模型冒烟

不加载真实模型的测试：

```bash
python -m pip install -r api_server/requirements-test.txt
python -m unittest discover -s api_server/tests -v
python -m compileall -q api_server
```

服务启动后运行真实模型冒烟：

```bash
python -m api_server.smoke_test --base-url http://127.0.0.1:8000 --api-key '替换成你的密钥' --model cosyvoice3-0.5b --voice default --text '你好，这是一次真实模型冒烟测试。' --output smoke.wav
```

该脚本会验证：

- `Content-Type` 为 WAV；
- 单声道、16-bit、预期采样率；
- WAV 包含有效音频帧且不是全静音；
- 响应头时长与实际 WAV 时长一致；
- RTF 非负；
- clipping ratio 没有超过阈值。

当前 H100 单样例基线为 6.4 秒输出音频、RTF 约 0.40、clipping ratio 为 0。该结果只适合作为链路冒烟基线，不代表 P95 延迟、并发吞吐量或长时间稳定性。

## 10. 内网部署检查清单

上线内网试运行前确认：

- 只启动一个 Uvicorn worker；
- `COSYVOICE_MAX_CONCURRENCY=1`；
- 非回环监听配置了强 API Key；
- `/ready` 返回 200；
- `/v1/models` 和 `/v1/audio/voices` 返回预期配置；
- 至少执行一次真实模型 smoke test；
- 网关配置 HTTPS、请求体大小限制、IP/API Key 限流、连接/响应超时和访问日志；
- 监控 429、504、队列等待时间、推理延迟、RTF、GPU 显存和 GPU 利用率。

当前服务是非流式 API v0.1 MVP。对外正式发布前还建议补充真实 TCP 断连集成测试、Docker build CI、持续压测和故障恢复测试。
