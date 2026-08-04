# CosyVoice API 调用逻辑与代码结构

本文说明 `api_server` 的 HTTP 行为、内部调用链、配置方式和目录内各文件的
职责。TECO SDAA 上 PyTorch/vLLM backend 的双终端启动与推理命令见
[`README_SDAA.md`](README_SDAA.md)。

## 1. 服务定位

服务提供 OpenAI 风格的非流式文本转语音接口，但只实现本文列出的字段：

| 方法 | 路径 | 鉴权 | 作用 |
| --- | --- | --- | --- |
| `GET` | `/health` | 无 | HTTP 进程存活检查 |
| `GET` | `/ready` | 无 | 模型和音色缓存是否加载完成 |
| `GET` | `/v1/models` | Bearer API Key | 查询模型别名、采样率和输出格式 |
| `GET` | `/v1/audio/voices` | Bearer API Key | 查询服务端注册的音色 ID |
| `POST` | `/v1/audio/speech` | Bearer API Key | 生成 WAV 或裸 PCM16 |

当前支持：

- CosyVoice3 zero-shot 音色和模型自带的 SFT speaker；
- zero-shot 音色的 `instructions` 风格指令；
- 24 kHz 单声道 PCM16 WAV，以及无文件头的 PCM16 little-endian；
- API Key、请求 ID、有限队列、超时、统一错误响应和音频质量门禁。

当前不支持流式输出、MP3、Opus、SSML、任意采样率、pitch、时间戳和客户端
上传参考音频。请求中的未知字段会返回 400。

`COSYVOICE_LOAD_VLLM=true` 不是启动通用的 `vllm serve`。CosyVoice 只把
LLM/speech-token 生成阶段交给进程内的 vLLM Engine，prompt embedding、
Flow/DiT、HiFT/Vocoder 和 HTTP API 仍由同一个 CosyVoice 进程负责。

## 2. 启动调用链

```text
python -m api_server.main
        │
        ├─ Settings.from_env()             读取并校验环境变量
        ├─ VoiceStore.from_json()          读取服务端音色注册表
        ├─ create_app()                    创建 FastAPI、路由和并发控制器
        └─ FastAPI lifespan
              └─ CosyVoiceEngine.load()
                    ├─ 校验 backend 对应的 Transformers 版本
                    ├─ AutoModel(model_dir, load_vllm, fp16)
                    ├─ 验证 SFT spk_id
                    └─ 预处理并缓存 zero-shot 音色
```

`main.py` 固定只启动一个 Uvicorn worker。模型加载放在 FastAPI lifespan
中执行；如果加载失败，HTTP 进程仍可返回 `/health`，但 `/ready` 返回 503，
日志中会保留加载异常。

SDAA 环境中的 `campplus.onnx` 和 `speech_tokenizer_v3.onnx` 只用于启动期
zero-shot 音色预处理。当前 TECO 3.2.0 镜像没有 ONNX Runtime SDAA
Execution Provider，这两个辅助模型使用 CPU provider 运行一次并缓存；
LLM、Flow/DiT 和 HiFT/Vocoder 主推理仍在 SDAA 上。

## 3. 单次语音请求调用链

```text
HTTP request
    │
    ├─ request-id middleware
    ├─ Bearer API Key
    ├─ Pydantic 严格字段校验
    ├─ model / text length / voice / instructions 校验
    ├─ admission semaphore       限制“运行中 + 排队中”请求总数
    ├─ inference semaphore       限制进入推理的请求数
    └─ CosyVoiceEngine.synthesize()
          ├─ inference lock      保护单模型实例
          ├─ 设置 Python/NumPy/Torch/SDAA/vLLM seed
          ├─ 按音色模式分派 CosyVoice inference_*()
          ├─ 收集所有 tts_speech chunk
          ├─ 音频质量门禁与可选换 seed 重试
          ├─ float waveform → PCM16
          └─ WAV 封装或裸 PCM
                └─ HTTP response + 性能/质量响应头
```

详细行为：

1. `app.py` 接受客户端 `X-Request-ID`，未提供时生成 UUID，并在响应中返回。
2. 鉴权使用 `hmac.compare_digest()` 比较 Bearer token。
3. `schemas.py` 禁止未知字段，并校验语速、格式、instructions 和 seed 范围。
4. 总接纳量为
   `COSYVOICE_MAX_CONCURRENCY + COSYVOICE_MAX_QUEUE_SIZE`；没有空位时立即
   返回 429。
5. GPU 推理在线程中执行。HTTP 超时或客户端断开不能安全终止已进入 GPU
   的 Python 线程，因此并发名额会等后台推理真正结束后再释放。
6. `engine.py` 目前还有进程级推理锁；同一个模型实例保持串行最安全。
7. 推理结束后先做质量检查，只有通过的候选才会编码成音频响应。

## 4. PyTorch 与 vLLM backend 的区别

两种 backend 使用同一套 HTTP 路由、音色、Flow/DiT、HiFT、音频编码和质量
门禁，仅 speech-token 生成方式不同：

| 配置 | speech-token 阶段 | 后续阶段 |
| --- | --- | --- |
| `COSYVOICE_LOAD_VLLM=false` | CosyVoice 原生 PyTorch LLM 解码 | Flow/DiT → HiFT |
| `COSYVOICE_LOAD_VLLM=true` | 进程内 vLLM Engine 解码 | Flow/DiT → HiFT |

`engine.py` 启动时会校验依赖，防止加载错误版本后仍生成语义错误或断续音频：

- PyTorch backend：`transformers==4.51.3`；
- vLLM backend：代码接受 `transformers==4.57.1` 或 `4.57.3`。

不要在两个容器环境之间混装 requirements。

## 5. 请求字段

`POST /v1/audio/speech` 请求体：

| 字段 | 必填 | 取值/限制 | 说明 |
| --- | --- | --- | --- |
| `model` | 是 | 当前为 `cosyvoice3-0.5b` | 服务端模型别名 |
| `input` | 是 | 非空，默认最多 2000 字符 | 待合成文本 |
| `voice` | 是 | `/v1/audio/voices` 返回的 ID | 服务端注册音色 |
| `response_format` | 否 | `wav`、`pcm`，默认 `wav` | 输出格式 |
| `speed` | 否 | `0.5`–`2.0`，默认 `1.0` | 语速 |
| `instructions` | 否 | 最多 1000 字符 | 只允许 zero-shot 音色 |
| `seed` | 否 | `0`–`4294967295` | 固定 speech-token 采样 |

示例：

```json
{
  "model": "cosyvoice3-0.5b",
  "input": "你好，这是一次 CosyVoice API 测试。",
  "voice": "default",
  "response_format": "wav",
  "speed": 1.0,
  "seed": 2
}
```

给 SFT 音色传 `instructions` 会在进入推理队列前返回
`400 unsupported_parameter`。

## 6. Seed 与质量门禁

未显式传 `seed` 时：

1. 从 `COSYVOICE_DEFAULT_SEED` 开始；
2. 候选音频未通过质量门禁时依次尝试下一个 seed；
3. 最多额外尝试 `COSYVOICE_QUALITY_MAX_RETRIES` 次；
4. 所有候选都失败时返回 `503 audio_quality_failed`。

显式传 `seed` 用于复现结果，同时关闭自动换 seed。某个显式 seed 可能被
质量门禁拒绝，这是正常行为；客户端必须先检查 HTTP 状态和 `Content-Type`，
不能把 503 JSON 错误体保存成 `.wav`。

当前质量门禁检查：

- 空数组、NaN 或 Infinity；
- 音频过短或相对文本异常长；
- 20 ms 帧中低于 -50 dBFS 的静音比例是否过高；
- 按中日韩字符、英文单词和缩写估算的有效发音时长是否不足。

实际返回使用的 seed 位于 `X-Generation-Seed`，自动换 seed 次数位于
`X-Quality-Retry-Count`。

## 7. 音色注册与分派

客户端只能提交音色 ID，不能提交服务器文件路径。音色由
`api_server/voices.json` 管理，修改后必须重启服务。

当前注册表只有：

```text
default  zero_shot  Default zero-shot voice
```

zero-shot 示例：

```json
{
  "my_voice": {
    "name": "My zero-shot voice",
    "mode": "zero_shot",
    "prompt_text": "You are a helpful assistant.<|endofprompt|>参考音频文本。",
    "prompt_audio": "asset/my_prompt.wav"
  }
}
```

要求：

- 音色 ID 只允许字母、数字、下划线、点和短横线，最长 128；
- `prompt_audio` 必须位于仓库目录内；
- `prompt_text` 必须与参考音频实际内容一致；
- 服务启动时调用 `add_zero_shot_spk()` 缓存音色；
- 普通 zero-shot 请求按缓存 ID 调用 `inference_zero_shot()`；
- 带 `instructions` 的 zero-shot 请求调用 `inference_instruct2()`。

SFT 示例：

```json
{
  "sft_female": {
    "name": "Built-in female voice",
    "mode": "sft",
    "spk_id": "模型中真实存在的 speaker ID"
  }
}
```

启动时会通过 `list_available_spks()` 验证 `spk_id`。当前
`Fun-CosyVoice3-0.5B-2512` 部署没有 `spk2info.pt`，因此没有可直接注册的
模型内置 SFT speaker；当前可用音色以 `/v1/audio/voices` 返回值为准。

## 8. 成功响应头

| 响应头 | 含义 |
| --- | --- |
| `X-Request-ID` | 请求唯一 ID |
| `X-Audio-Sample-Rate` | 输出采样率 |
| `X-Audio-Channels` | 声道数，当前为 1 |
| `X-Audio-Duration` | 音频时长，秒 |
| `X-Queue-Wait-Ms` | 等待推理名额的时间 |
| `X-Inference-Latency-Ms` | 模型推理和音频收集耗时 |
| `X-Real-Time-Factor` | 推理耗时除以音频时长 |
| `X-Generation-Seed` | 最终实际使用的 seed |
| `X-Quality-Retry-Count` | 质量门禁触发的换 seed 次数 |
| `X-Silent-Frame-Ratio` | 低于 -50 dBFS 的 20 ms 帧比例 |

## 9. 错误格式

所有业务错误使用稳定 JSON：

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

常见状态：

| HTTP | `code` | 含义 |
| ---: | --- | --- |
| 400 | `invalid_parameter` | 字段类型、范围或未知字段错误 |
| 400 | `unsupported_parameter` | 不支持的参数组合 |
| 401 | `invalid_api_key` | API Key 缺失或错误 |
| 404 | `model_not_found` / `voice_not_found` | 模型或音色不存在 |
| 413 | `input_too_large` | 文本超过长度限制 |
| 429 | `queue_full` | 运行中和排队中请求已满 |
| 503 | `model_not_ready` | 模型尚未加载或加载失败 |
| 503 | `audio_quality_failed` | 所有候选音频均被质量门禁拒绝 |
| 504 | `inference_timeout` | HTTP 等待推理超时 |

## 10. 环境变量

| 环境变量 | 代码默认值 | 作用 |
| --- | --- | --- |
| `COSYVOICE_MODEL_ALIAS` | `cosyvoice3-0.5b` | 对外模型名 |
| `COSYVOICE_MODEL_DIR` | `pretrained_models/Fun-CosyVoice3-0.5B` | 模型目录 |
| `COSYVOICE_VOICES_FILE` | `api_server/voices.json` | 音色注册表 |
| `COSYVOICE_API_KEY` | 未设置 | Bearer API Key |
| `COSYVOICE_HOST` | `127.0.0.1` | 监听地址 |
| `COSYVOICE_PORT` | `8000` | 监听端口 |
| `COSYVOICE_ALLOW_UNAUTHENTICATED` | `false` | 是否显式允许非回环无鉴权 |
| `COSYVOICE_MAX_TEXT_CHARACTERS` | `2000` | 文本长度上限 |
| `COSYVOICE_MAX_CONCURRENCY` | `1` | 推理 semaphore 容量 |
| `COSYVOICE_MAX_QUEUE_SIZE` | `16` | 额外排队容量 |
| `COSYVOICE_REQUEST_TIMEOUT_SECONDS` | `600` | HTTP 推理超时 |
| `COSYVOICE_FP16` | `false` | backend 是否使用 FP16 |
| `COSYVOICE_LOAD_VLLM` | `false` | 是否启用嵌入式 vLLM |
| `COSYVOICE_DEFAULT_SEED` | `2` | 未传 seed 时的首个 seed |
| `COSYVOICE_QUALITY_CHECK_ENABLED` | `true` | 是否开启质量门禁 |
| `COSYVOICE_QUALITY_MAX_RETRIES` | `2` | 首次候选之后的最大重试数 |

没有 API Key 时，服务只允许绑定回环地址。绑定 `0.0.0.0`、主机名或其他
非回环地址必须配置 Key，除非主动设置
`COSYVOICE_ALLOW_UNAUTHENTICATED=true`；共享网络中不建议关闭鉴权。

## 11. 文件职责

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | Python package 标识 |
| `main.py` | `python -m api_server.main` 入口；启动单 worker Uvicorn |
| `config.py` | 从环境变量构造并校验 `Settings` |
| `app.py` | FastAPI 工厂、lifespan、鉴权、请求 ID、队列、路由和错误映射 |
| `schemas.py` | `SpeechRequest` 严格请求模型 |
| `errors.py` | 稳定的 `ServiceError` 类型 |
| `voice_store.py` | 解析、校验和查询服务端音色注册表 |
| `voices.json` | 当前实际音色配置 |
| `engine.py` | 加载 CosyVoice、选择 backend、seed、推理分派、质量重试和编码 |
| `audio_codec.py` | waveform 收集、质量分析、PCM16 转换和 WAV 封装 |
| `smoke_test.py` | 真实 HTTP 冒烟客户端和 WAV 结构/静音/clipping 校验 |
| `../tools/test_cosyvoice_api.py` | 固定端口的 PyTorch/vLLM 真实 API 验收客户端；保存端点结果、响应头、音频及汇总 |
| `requirements-runtime.txt` | PyTorch backend 的 API/CosyVoice 依赖 |
| `requirements-vllm.txt` | vLLM backend 的额外版本约束；不得装入 PyTorch 环境 |
| `requirements-test.txt` | 不加载真实模型的单元测试依赖 |
| `Dockerfile.sdaa` | 当前 SDAA API 镜像的 shell-first 元数据层，默认 PID 1 为 bash |
| `Dockerfile` | 通用 PyTorch/CUDA 参考构建；当前 SDAA 服务器不使用 |
| `Dockerfile.vllm` | 旧的独立 CUDA vLLM 参考构建；当前 SDAA 服务器不使用 |
| `README.md` | 本文：API 调用逻辑和代码结构 |
| `README_SDAA.md` | 当前 TECO SDAA 双终端启动、请求和验收命令 |

测试文件：

| 文件 | 覆盖范围 |
| --- | --- |
| `tests/test_config.py` | 环境变量默认值、范围和安全校验 |
| `tests/test_api.py` | 路由、鉴权、错误、队列、超时和响应头 |
| `tests/test_engine.py` | backend 分派、seed、质量重试和音色验证 |
| `tests/test_audio_codec.py` | 音频转换和质量门禁 |
| `tests/test_smoke.py` | 冒烟客户端的 WAV 验证 |

## 12. 测试

不加载真实模型：

```bash
python -m pip install -r api_server/requirements-test.txt
python -m unittest discover -s api_server/tests -v
python -m compileall -q api_server
```

服务运行后做真实模型验收。脚本从仓库根目录的
`cosyvoice_sdaa_vllm_api.env` 读取 Key，不在命令行或日志中展开。每次命令
只发送一种音色模式、instructions 和响应格式组合，避免把不同推理路径的
结果混在同一个目录。

PyTorch zero-shot、无 instructions、WAV：

```bash
cd /mnt/nvme/application/yangyu/CosyVoice
python3 tools/test_cosyvoice_api.py \
  --backend pytorch \
  --voice-mode zero_shot \
  --voice default \
  --response-format wav \
  --seed 0 \
  --output-dir /mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/manual/pytorch_zero_shot_no_instruction_wav
```

vLLM zero-shot、无 instructions、WAV：

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

vLLM zero-shot、有 instructions、WAV：

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

vLLM zero-shot、无 instructions、裸 PCM：

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

SFT 必须先在 `api_server/voices.json` 注册真实 `spk_id`。注册并重启服务后，
脚本可以自动选择第一个 SFT 音色，也可以用 `--voice` 指定：

```bash
cd /mnt/nvme/application/yangyu/CosyVoice
python3 tools/test_cosyvoice_api.py \
  --backend vllm \
  --voice-mode sft \
  --response-format wav \
  --seed 0 \
  --output-dir /mnt/nvme/application/yangyu/CosyVoice/cosyvoice_api_outputs/manual/vllm_sft_no_instruction_wav
```

当前 `/workspace/model` 没有 `spk2info.pt`，服务只注册了 `default`
zero-shot 音色。因此当前执行 SFT 命令会在发起语音请求前明确报告
`no sft voice is registered`，不会生成错误音频。SFT 不允许
`instructions`，客户端同样会在请求前拒绝这种组合。

脚本固定访问 PyTorch 的 `127.0.0.1:8020` 或 vLLM 的
`127.0.0.1:8021`，先用 `/v1/audio/voices` 核对音色 ID 和 mode，再检查
`Content-Type`、24 kHz 单声道 PCM16、有效帧和非全静音。每个输出目录
写入请求参数 `request.json` 和结果 `results.json`。HTTP 或质量门禁错误只
保存为 `.error.json`，本次单用例结果为失败，不会把错误 JSON 写成 WAV。
默认终端只显示类似 curl 的传输摘要；需要排查全部端点、响应头和指标时追加
`--verbose`，完整结果始终保存在 `results.json`。
