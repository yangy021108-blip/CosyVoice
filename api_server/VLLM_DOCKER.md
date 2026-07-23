# CosyVoice vLLM Docker（当前服务器）

本文记录 GPU 服务器上已经验证通过的 CosyVoice3 vLLM API 部署方式。

## 1. 部署结构

HTTP 入口仍然是：

```bash
/opt/conda/envs/cosyvoice/bin/python -m api_server.main
```

`COSYVOICE_LOAD_VLLM=true` 会让 CosyVoice 的 LLM/语音 token 阶段使用 vLLM；Flow、DiT、Vocoder 和 HTTP API 不会变成独立的 `vllm serve` 服务。

当前验证配置：

| 项目 | 值 |
| --- | --- |
| 镜像 | `cosyvoice-api:vllm-test` |
| 容器 | `cosyvoice_api_vllm_yy` |
| GPU | 物理 GPU 6 |
| 地址 | `127.0.0.1:8011` |
| PyTorch | `2.8.0` |
| vLLM | `0.11.0` |
| Transformers | `4.57.1` |
| NumPy | `1.26.4` |

现有 8010 PyTorch 服务不受影响。

## 2. 为什么使用挂载环境

宿主机 `/var/lib/docker` 所在根分区只剩约 5 GB，无法安全构建包含另一份 CUDA、PyTorch 和 vLLM 的完整镜像。`Dockerfile.vllm` 因此复用本机已有的 `qwenllm/qwen3-asr:latest` CUDA 基础层，并把独立 Conda 环境放在容量充足的 `/SharedData`。

该环境是原 `cosyvoice` 环境的独立副本。升级 Transformers 不会影响 8010 的 PyTorch 服务。

## 3. 一次性构建和环境准备

在仓库根目录构建薄镜像：

```bash
docker build -t cosyvoice-api:vllm-test -f api_server/Dockerfile.vllm api_server
```

第一次部署时复制已经验证的 CosyVoice 环境。下面的判断可以避免重复复制：

```bash
if [ ! -d /SharedData/yangyu/cosyvoice_vllm_env/cosyvoice ]; then
  mkdir -p /SharedData/yangyu/cosyvoice_vllm_env
  docker cp vllm_env_yy:/opt/conda/envs/cosyvoice /SharedData/yangyu/cosyvoice_vllm_env
fi
```

只在副本中安装 vLLM 后端依赖：

```bash
docker run --rm \
  -v /SharedData/yangyu/cosyvoice_vllm_env/cosyvoice:/opt/conda/envs/cosyvoice \
  -v /SharedData/yangyu/CosyVoice:/workspace/CosyVoice \
  cosyvoice-api:vllm-test \
  /opt/conda/envs/cosyvoice/bin/python \
  -m pip install \
  --no-cache-dir \
  -r /workspace/CosyVoice/api_server/requirements-vllm.txt
```

验证版本和 GPU：

```bash
docker run --rm \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=6 \
  -v /SharedData/yangyu/cosyvoice_vllm_env/cosyvoice:/opt/conda/envs/cosyvoice \
  -v /SharedData/yangyu/CosyVoice:/workspace/CosyVoice \
  cosyvoice-api:vllm-test \
  /opt/conda/envs/cosyvoice/bin/python \
  -c 'import torch; from importlib import metadata; print(torch.cuda.get_device_name(0)); print("torch", metadata.version("torch"), "vllm", metadata.version("vllm"), "transformers", metadata.version("transformers"), "numpy", metadata.version("numpy"))'
```

## 4. 配置和一次性创建容器

创建只允许当前用户读取的环境变量文件，然后使用编辑器填写内容。不要把真实 API Key 提交到 Git：

```bash
touch /SharedData/yangyu/cosyvoice_vllm_api.env
chmod 600 /SharedData/yangyu/cosyvoice_vllm_api.env
vim /SharedData/yangyu/cosyvoice_vllm_api.env
```

文件内容：

```dotenv
COSYVOICE_LOAD_VLLM=true
COSYVOICE_HOST=127.0.0.1
COSYVOICE_PORT=8011
COSYVOICE_API_KEY=替换成真实密钥
COSYVOICE_MAX_CONCURRENCY=1
COSYVOICE_MAX_QUEUE_SIZE=16
COSYVOICE_REQUEST_TIMEOUT_SECONDS=600
```

下面把镜像构建、容器创建、容器启动和 API 服务启动分开。需要注意：

- `docker build` 只构建镜像；
- `docker create` 只创建容器，而且只需执行一次；
- `docker start` 启动已存在的容器；
- `docker exec` 只能在运行中的容器内执行命令，不能直接启动一个已停止的容器。

先检查同名容器。只有查到旧容器时才执行删除：

```bash
docker ps -a --filter name=cosyvoice_api_vllm_yy
docker rm -f cosyvoice_api_vllm_yy
```

准备缓存目录：

```bash
mkdir -p /SharedData/yangyu/cosyvoice_vllm_cache
```

一次性创建容器。末尾的 `sleep infinity` 只负责让容器保持运行，不会启动 API：

```bash
docker create \
  --name cosyvoice_api_vllm_yy \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=6 \
  --ipc=host \
  --network host \
  --env-file /SharedData/yangyu/cosyvoice_vllm_api.env \
  -v /SharedData/yangyu/CosyVoice:/workspace/CosyVoice \
  -v /SharedData/yangyu/cosyvoice_vllm_env/cosyvoice:/opt/conda/envs/cosyvoice:ro \
  -v /SharedData/yangyu/cosyvoice_vllm_cache:/root/.cache \
  cosyvoice-api:vllm-test \
  sleep infinity
```

`--env-file` 的值会在创建容器时写入容器配置。如果之后修改了该文件中的端口、API Key 或其他环境变量，需要删除并重新创建容器；只执行 `docker restart` 不会重新读取环境文件。

## 5. 窗口 A：启动容器和 vLLM 后端服务

先启动容器。服务器重启或执行过 `docker stop` 后，也先运行这一条：

```bash
docker start cosyvoice_api_vllm_yy
```

然后在窗口 A 前台启动 CosyVoice API：

```bash
docker exec -it cosyvoice_api_vllm_yy \
  /opt/conda/envs/cosyvoice/bin/python \
  -m api_server.main
```

这就是 CosyVoice 场景中“启动 vLLM 服务”的正确命令，但它不是通用的 `vllm serve`。CosyVoice 只把 LLM/语音 token 阶段交给嵌入式 vLLM V1 Engine；Flow、DiT、Vocoder 和 `/v1/audio/speech` 仍由同一个 CosyVoice 进程负责。单独执行 `vllm serve` 无法生成最终语音。

首次启动服务会执行 vLLM 权重导出、`torch.compile` 和 CUDA Graph 捕获，需要等待约两分钟。窗口 A 出现下面两类日志表示 vLLM 后端和 HTTP 服务均已就绪：

```text
Initializing a V1 LLM engine (v0.11.0)
Uvicorn running on http://127.0.0.1:8011
```

保持窗口 A 不要关闭。

## 6. 窗口 B：发送请求

健康检查：

```bash
curl http://127.0.0.1:8011/health
curl http://127.0.0.1:8011/ready
```

模型列表：

```bash
API_KEY="$(sed -n 's/^COSYVOICE_API_KEY=//p' /SharedData/yangyu/cosyvoice_vllm_api.env)"
curl http://127.0.0.1:8011/v1/models \
  -H "Authorization: Bearer ${API_KEY}"
```

真实语音推理：

```bash
API_KEY="$(sed -n 's/^COSYVOICE_API_KEY=//p' /SharedData/yangyu/cosyvoice_vllm_api.env)"
curl --fail-with-body \
  --request POST \
  http://127.0.0.1:8011/v1/audio/speech \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"model":"cosyvoice3-0.5b","input":"你好，这是 CosyVoice vLLM API 的真实推理测试。","voice":"default","response_format":"wav","speed":1.0}' \
  --output vllm_result.wav
```

在运行中的容器内验证 WAV，不依赖宿主机安装 `ffprobe`：

```bash
docker exec cosyvoice_api_vllm_yy \
  /opt/conda/envs/cosyvoice/bin/python \
  -c 'import wave; f = wave.open("/workspace/CosyVoice/vllm_result.wav", "rb"); print("sample_rate", f.getframerate()); print("channels", f.getnchannels()); print("sample_width", f.getsampwidth()); print("duration", round(f.getnframes() / f.getframerate(), 2))'
```

本次真实验证结果：

```text
HTTP：200
音频：24 kHz、单声道、PCM16 WAV
音频时长：6.08 秒
RTF：0.2599
clipping ratio：0
ASR：你好，这是 Cosy Voice VLM API 的真实推理测试。
```

ASR 将英文缩写 `vLLM` 听写为 `VLM`，其余文本与输入一致。

拆分后的生命周期也已经真实验证：单独执行 `docker start` 时只有容器运行，执行 `docker exec` 后 API 才就绪；窗口 B 成功生成了 24 kHz、单声道、PCM16 WAV；执行 `docker stop` 后，再次按 `docker start`、`docker exec` 的顺序可以恢复服务。

## 7. 停止和再次启动

只停止 API 服务时，在窗口 A 按 `Ctrl+C`。此时 `sleep infinity` 仍在运行，所以容器不会退出。

要连容器一起停止，在另一个窗口执行：

```bash
docker stop cosyvoice_api_vllm_yy
```

下次使用时，先启动容器：

```bash
docker start cosyvoice_api_vllm_yy
```

再在窗口 A 重新执行服务命令：

```bash
docker exec -it cosyvoice_api_vllm_yy \
  /opt/conda/envs/cosyvoice/bin/python \
  -m api_server.main
```

然后在窗口 B 重复第 6 节的请求命令。

## 8. 日常管理

查看容器状态：

```bash
docker ps --filter name=cosyvoice_api_vllm_yy
```

API 通过交互式 `docker exec` 在窗口 A 前台运行，日志也直接显示在窗口 A，不使用 `docker logs`。删除测试容器不会删除模型、独立环境或缓存：

```bash
docker rm -f cosyvoice_api_vllm_yy
```

不要在原 `/opt/conda/envs/cosyvoice` 环境中安装 Transformers 4.57.1；8010 的 PyTorch 后端必须继续使用 Transformers 4.51.3。
