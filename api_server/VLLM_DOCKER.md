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

## 4. 配置和启动

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

先检查同名容器。只有查到旧容器时才执行删除：

```bash
docker ps -a --filter name=cosyvoice_api_vllm_yy
docker rm -f cosyvoice_api_vllm_yy
```

准备缓存目录：

```bash
mkdir -p /SharedData/yangyu/cosyvoice_vllm_cache
```

下面是一个 `docker run` 命令，按参数分行展示，可以在 Bash 中逐行输入：

```bash
docker run -d \
  --name cosyvoice_api_vllm_yy \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=6 \
  --ipc=host \
  --network host \
  --env-file /SharedData/yangyu/cosyvoice_vllm_api.env \
  -v /SharedData/yangyu/CosyVoice:/workspace/CosyVoice \
  -v /SharedData/yangyu/cosyvoice_vllm_env/cosyvoice:/opt/conda/envs/cosyvoice:ro \
  -v /SharedData/yangyu/cosyvoice_vllm_cache:/root/.cache \
  cosyvoice-api:vllm-test
```

首次启动会执行 vLLM 权重导出、`torch.compile` 和 CUDA Graph 捕获，需要等待约两分钟：

```bash
docker logs -f cosyvoice_api_vllm_yy
```

看到下面两类日志表示 vLLM 后端和 HTTP 服务均已就绪：

```text
Initializing a V1 LLM engine (v0.11.0)
Uvicorn running on http://127.0.0.1:8011
```

## 5. 在另一个 Shell 测试

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

## 6. 日常管理

查看状态和日志：

```bash
docker ps --filter name=cosyvoice_api_vllm_yy
docker logs --tail 100 cosyvoice_api_vllm_yy
```

停止和重新启动：

```bash
docker stop cosyvoice_api_vllm_yy
```

```bash
docker start cosyvoice_api_vllm_yy
```

删除测试容器不会删除模型、独立环境或缓存：

```bash
docker rm -f cosyvoice_api_vllm_yy
```

不要在原 `/opt/conda/envs/cosyvoice` 环境中安装 Transformers 4.57.1；8010 的 PyTorch 后端必须继续使用 Transformers 4.51.3。
