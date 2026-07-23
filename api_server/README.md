# CosyVoice HTTP API

[中文使用指南](README_zh.md) |
[当前服务器 vLLM Docker 详解](VLLM_DOCKER.md)

This package adds a small product-facing HTTP layer without changing the model
implementation under `cosyvoice/`.

## Current scope

- `POST /v1/audio/speech`: OpenAI-style JSON request and WAV/PCM response.
- `GET /health` and `GET /ready`: process and model readiness checks.
- `GET /v1/models` and `GET /v1/audio/voices`: discover loaded capabilities.
- One model process and one GPU inference at a time by default.
- Bearer authentication for non-loopback deployments, bounded admission,
  request IDs, stable errors, and basic latency/RTF response headers.

The first version is intentionally non-streaming. It does not pretend to
support MP3/Opus, SSML, pitch control, arbitrary sample rates, or word
timestamps. These parameters are rejected instead of silently ignored.

## Start the server

Run from the repository root in the existing CosyVoice Python environment:

```bash
export CUDA_VISIBLE_DEVICES=0
export COSYVOICE_API_KEY='replace-with-a-secret'
python -m api_server.main
```

The regular PyTorch backend must use the runtime dependency set, including
`transformers==4.51.3`. Verify the active environment before starting:

```bash
python -c "import torch, transformers; print('torch=', torch.__version__, 'transformers=', transformers.__version__)"
```

If the command reports another Transformers version, restore the regular
backend dependencies with this single-line command:

```bash
python -m pip install --upgrade "transformers==4.51.3" "tokenizers>=0.21,<0.22"
```

Do not install `requirements-vllm.txt` into the regular PyTorch environment.
Its `transformers==4.57.1` pin is for `COSYVOICE_LOAD_VLLM=true` and must use a
separate environment or image. The server checks this at startup because the
wrong Transformers backend can produce fluent-looking WAV files whose speech
does not match the input text.

For an isolated CUDA environment, build the API image from the small
`api_server` context and mount the repository (the model stays outside the
image):

```bash
docker build -t cosyvoice-api:dev -f api_server/Dockerfile api_server

docker run --rm --gpus device=4 \
  -p 8000:8000 \
  -v "$PWD:/workspace/CosyVoice" \
  -e COSYVOICE_HOST=0.0.0.0 \
  -e COSYVOICE_API_KEY='replace-with-a-secret' \
  cosyvoice-api:dev
```

The image intentionally supplies PyTorch/torchaudio as a matched pair in the
base layer instead of reinstalling the older CUDA wheels from the repository's
training-oriented `requirements.txt`.

### Verified vLLM Docker workflow on the current server

The HTTP entry point remains `python -m api_server.main`. Setting
`COSYVOICE_LOAD_VLLM=true` replaces only CosyVoice's LLM/speech-token stage
with the embedded vLLM V1 engine; it does not start a separate `vllm serve`
process.

The Docker data root on the current server has less than 5 GB free, so a
second self-contained CUDA/PyTorch image cannot be built safely there. The
verified deployment uses the thin `Dockerfile.vllm`, reuses the locally
available CUDA base image, and mounts a dedicated copy of the Python
environment from `/SharedData`. The copied environment occupies about 31 GB
on `/SharedData` and does not change the PyTorch environment used by port
8010.

Run every command below from `/SharedData/yangyu/CosyVoice`. Build the thin
image:

```bash
docker build -t cosyvoice-api:vllm-test -f api_server/Dockerfile.vllm api_server
```

Copy the known-good environment once. Run the following block line by line;
the conditional prevents a second copy:

```bash
if [ ! -d /SharedData/yangyu/cosyvoice_vllm_env/cosyvoice ]; then
  mkdir -p /SharedData/yangyu/cosyvoice_vllm_env
  docker cp vllm_env_yy:/opt/conda/envs/cosyvoice /SharedData/yangyu/cosyvoice_vllm_env
fi
```

Install the vLLM dependency set only in that copy:

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

Confirm both the isolated versions and GPU access:

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

The verified versions are PyTorch 2.8.0, vLLM 0.11.0, Transformers 4.57.1,
and NumPy 1.26.4. Do not install this Transformers version into the original
PyTorch environment, which must remain on Transformers 4.51.3.

Choose one of the following API-key modes.

#### Mode A: protected host environment file

Create the file outside the repository on the Docker host, then open it in an
editor:

```bash
touch /SharedData/yangyu/cosyvoice_vllm_api.env
chmod 600 /SharedData/yangyu/cosyvoice_vllm_api.env
vim /SharedData/yangyu/cosyvoice_vllm_api.env
```

Put the following values in the file, replacing the API key placeholder:

```dotenv
COSYVOICE_MODEL_DIR=/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
COSYVOICE_PORT=8011
COSYVOICE_API_KEY=replace-with-a-secret
```

`--env-file` reads these values while creating the container; it does not
mount the file into the container. Therefore, it is normal that
`/SharedData/yangyu/cosyvoice_vllm_api.env` cannot be found from a shell
inside `cosyvoice_api_vllm_yy`.

#### Mode B: specify the API key when starting the service

This mode does not require `cosyvoice_vllm_api.env`. Omit the following line
from the `docker create` command below:

```text
--env-file /SharedData/yangyu/cosyvoice_vllm_api.env
```

The model directory, port, and API key will instead be passed to `docker exec`
in shell A. The vLLM image sets `COSYVOICE_LOAD_VLLM=true` and the
real-model-tested default seed `0`; all other service settings use the
conservative defaults from `api_server/config.py`. Do not store API keys in
`api_server/voices.json`: that file is the voice registry, may be committed
to Git, and is not an authentication configuration file.

The remaining steps deliberately separate image construction, container
creation, container startup, and API-process startup. A stopped container
cannot accept `docker exec`; always run `docker start` before `docker exec`.

Check whether an older test container exists. Remove it only when it appears
in the first command. Container creation is a one-time operation:

```bash
docker ps -a --filter name=cosyvoice_api_vllm_yy
docker rm -f cosyvoice_api_vllm_yy
```

Prepare the persistent cache:

```bash
mkdir -p /SharedData/yangyu/cosyvoice_vllm_cache
```

Create a persistent container on physical GPU 6. The final `sleep infinity`
keeps only the container alive; it does not start the API service:

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

In mode A, the environment-file values are copied into the container
configuration at creation time. Recreate the container after changing those
values. This does not apply to mode B.

Start the container. Run the same command after a host reboot or after
`docker stop`:

```bash
docker start cosyvoice_api_vllm_yy
```

In shell A, start the CosyVoice API process in the foreground. With mode A,
run:

```bash
docker exec -it cosyvoice_api_vllm_yy \
  /opt/conda/envs/cosyvoice/bin/python \
  -m api_server.main
```

With mode B, enter a key without echoing it to the terminal:

```bash
read -rsp "API Key: " COSYVOICE_API_KEY
```

```bash
echo
```

Then pass only the model directory, port, and API key to the new process:

```bash
docker exec -it \
  -e COSYVOICE_MODEL_DIR=/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
  -e COSYVOICE_PORT=8011 \
  -e COSYVOICE_API_KEY="${COSYVOICE_API_KEY}" \
  cosyvoice_api_vllm_yy \
  /opt/conda/envs/cosyvoice/bin/python \
  -m api_server.main
```

The three explicit settings are the only ones required for this image.
`COSYVOICE_HOST` defaults to `127.0.0.1`; text length, concurrency, queue,
timeout, and quality-check settings retain their documented defaults. The
vLLM image uses the tested default seed `0`.

This is the CosyVoice equivalent of the requested "vLLM serve" window.
Do not replace it with the generic `vllm serve` command: CosyVoice uses vLLM
only for its LLM/speech-token stage, while the same process must also run
Flow, DiT, the vocoder, and `/v1/audio/speech`. The first service startup
exports vLLM weights, runs `torch.compile`, and captures CUDA Graphs. Keep
shell A open until both `Initializing a V1 LLM engine (v0.11.0)` and
`Uvicorn running on http://127.0.0.1:8011` appear.

In shell B, check readiness:

```bash
curl --fail http://127.0.0.1:8011/health
curl --fail http://127.0.0.1:8011/ready
```

Set the client key with the same mode used in shell A. For mode A:

```bash
API_KEY="$(sed -n 's/^COSYVOICE_API_KEY=//p' /SharedData/yangyu/cosyvoice_vllm_api.env)"
```

For mode B, type the same key:

```bash
read -rsp "API Key: " API_KEY
```

```bash
echo
```

Then perform authenticated model discovery:

```bash
curl --fail-with-body http://127.0.0.1:8011/v1/models \
  -H "Authorization: Bearer ${API_KEY}"
```

Run a real vLLM-backed synthesis request:

```bash
curl --fail-with-body \
  --request POST \
  http://127.0.0.1:8011/v1/audio/speech \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"model":"cosyvoice3-0.5b","input":"你好，这是 CosyVoice vLLM API 的真实推理测试。","voice":"default","response_format":"wav","speed":1.0}' \
  --output vllm_result.wav
```

Validate the generated WAV with Python inside the running container; this does
not require `ffprobe` on the host:

```bash
docker exec cosyvoice_api_vllm_yy \
  /opt/conda/envs/cosyvoice/bin/python \
  -c 'import wave; f = wave.open("/workspace/CosyVoice/vllm_result.wav", "rb"); print("sample_rate", f.getframerate()); print("channels", f.getnchannels()); print("sample_width", f.getsampwidth()); print("duration", round(f.getnframes() / f.getframerate(), 2))'
```

The tested request returned HTTP 200, a 24 kHz mono PCM16 WAV, 6.08 seconds
of audio, RTF 0.2599, and zero clipped samples. An independent ASR check
matched the requested sentence (apart from spelling the spoken `vLLM`
abbreviation as `VLM`).

The separated lifecycle was also tested: `docker start` brought up only the
idle container, `docker exec` made the API ready, shell B generated a valid
24 kHz mono PCM16 WAV, and the same start/exec sequence restored the service
after `docker stop`. Mode B was tested with a container that had no
environment file or persistent API key: an unauthenticated request returned
HTTP 401, while the key passed to `docker exec -e` authorized a real
synthesis request.

The vLLM path forwards the request seed to `SamplingParams`. Repeating the
same mixed Chinese/English request with the same seed produced byte-identical
WAV files in the real-model test. The quality guard also accounts for spoken
Latin words and acronyms such as `CosyVoice`, `vLLM`, and `API`.

To stop only the API process, press `Ctrl+C` in shell A. The persistent
container remains running. To stop it as well:

```bash
docker stop cosyvoice_api_vllm_yy
```

To use it again, first start the container, then repeat the mode A or mode B
`docker exec` command in shell A:

```bash
docker start cosyvoice_api_vllm_yy
```

Useful settings:

| Environment variable | Default |
| --- | --- |
| `COSYVOICE_MODEL_ALIAS` | `cosyvoice3-0.5b` |
| `COSYVOICE_MODEL_DIR` | `pretrained_models/Fun-CosyVoice3-0.5B` |
| `COSYVOICE_VOICES_FILE` | `api_server/voices.json` |
| `COSYVOICE_HOST` / `COSYVOICE_PORT` | `127.0.0.1` / `8000` |
| `COSYVOICE_ALLOW_UNAUTHENTICATED` | `false` |
| `COSYVOICE_MAX_TEXT_CHARACTERS` | `2000` |
| `COSYVOICE_MAX_CONCURRENCY` | `1` |
| `COSYVOICE_MAX_QUEUE_SIZE` | `16` |
| `COSYVOICE_REQUEST_TIMEOUT_SECONDS` | `600` |
| `COSYVOICE_FP16` / `COSYVOICE_LOAD_VLLM` | `false` / `false` |
| `COSYVOICE_DEFAULT_SEED` | `2` |
| `COSYVOICE_QUALITY_CHECK_ENABLED` | `true` |
| `COSYVOICE_QUALITY_MAX_RETRIES` | `2` |

Use exactly one Uvicorn worker. Multiple workers load multiple copies of the
model and duplicate GPU memory.

Binding to a non-loopback address without `COSYVOICE_API_KEY` is rejected.
`COSYVOICE_ALLOW_UNAUTHENTICATED=true` is an explicit escape hatch for a
trusted, isolated environment; it should not be used for an external service.
Put public deployments behind an HTTPS gateway that enforces request-size
limits, per-key/IP rate limits, connection/response timeouts, and access logs.

## Synthesize speech

The commands in this section are intentionally one physical line so they can
be pasted into Bash without line-continuation whitespace errors.

```bash
curl --fail-with-body --request POST http://127.0.0.1:8000/v1/audio/speech -H "Authorization: Bearer replace-with-a-secret" -H "Content-Type: application/json" -d '{"model":"cosyvoice3-0.5b","input":"你好，这是一次 CosyVoice API 测试。","voice":"default","response_format":"wav","speed":1.0}' --output result.wav
```

Optional style control uses the OpenAI-compatible plural field name:

```json
{
  "instructions": "请用四川话、开心地说这句话"
}
```

CosyVoice speech-token generation is stochastic. Requests without `seed` start
from the tested default seed and automatically retry with the next seed when
the generated audio is obviously too short, too long for the input, or mostly
silent. Set an unsigned 32-bit `seed` explicitly when exact reproducibility is
more important than automatic retry. The selected seed, retry count, and
silence ratio are returned in `X-Generation-Seed`, `X-Quality-Retry-Count`, and
`X-Silent-Frame-Ratio`.

If all automatic attempts fail the quality guard, the API returns 503 with
`code=audio_quality_failed` instead of serving known-degenerate audio. The
quality guard can be disabled for diagnostics with
`COSYVOICE_QUALITY_CHECK_ENABLED=false`.

The built-in `default` voice is configured from `asset/zero_shot_prompt.wav`.
Clients never submit server-side paths. Add server-owned voices in
`api_server/voices.json`; zero-shot prompt features are cached once at model
startup for ordinary synthesis. `prompt_text` must contain the exact spoken
transcript of `prompt_audio` after the CosyVoice3 system prefix, including
punctuation. Restart the API after changing either value. To verify speaker
identity first, send a request without `instructions`, because dialect and
emotion instructions intentionally alter delivery.

Use a clear, single-speaker reference without music or long silence. A very
quiet reference can produce mostly silent output. This one-line command trims
leading/trailing silence, normalizes loudness, and writes a 16 kHz mono PCM WAV:

```bash
ffmpeg -y -i asset/my_prompt.wav -af "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,areverse,loudnorm=I=-24:LRA=7:TP=-3" -ar 16000 -ac 1 -c:a pcm_s16le asset/my_prompt_clean.wav
```

PCM responses are mono signed 16-bit little-endian at the model-native 24 kHz
sample rate. WAV responses contain the same samples with a complete WAV header.
On a headless server, convert raw PCM to WAV and play or download the WAV:

```bash
ffmpeg -y -f s16le -ar 24000 -ac 1 -i result.pcm result_pcm.wav
```

On a machine with an audio device, PCM can also be played without an SDL video
window:

```bash
ffplay -nodisp -autoexit -f s16le -ar 24000 -ac 1 result.pcm
```

## Errors

Errors use one stable shape:

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

## Tests and smoke check

Contract and audio encoding tests do not load the model:

```bash
python -m pip install -r api_server/requirements-test.txt
python -m unittest discover -s api_server/tests -v
python -m compileall -q api_server
```

After starting the real service:

```bash
python -m api_server.smoke_test --api-key replace-with-a-secret --output smoke.wav
```

The smoke client validates the content type, mono 16-bit WAV structure, sample
rate, non-empty/non-silent audio, duration headers, RTF, and clipping ratio
before saving the result.
