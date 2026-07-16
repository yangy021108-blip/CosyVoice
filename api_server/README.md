# CosyVoice HTTP API

This package adds a small product-facing HTTP layer without changing the model
implementation under `cosyvoice/`.

## Current scope

- `POST /v1/audio/speech`: OpenAI-style JSON request and WAV/PCM response.
- `GET /health` and `GET /ready`: process and model readiness checks.
- `GET /v1/models` and `GET /v1/audio/voices`: discover loaded capabilities.
- One model process and one GPU inference at a time by default.
- Optional Bearer authentication, bounded admission, request IDs, stable errors,
  and basic latency/RTF response headers.

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

For an isolated CUDA environment, build the API image from the small
`api_server` context and mount the repository (the model stays outside the
image):

```bash
docker build -t cosyvoice-api:dev -f api_server/Dockerfile api_server

docker run --rm --gpus device=4 \
  -p 8000:8000 \
  -v "$PWD:/workspace/CosyVoice" \
  -e COSYVOICE_API_KEY='replace-with-a-secret' \
  cosyvoice-api:dev
```

The image intentionally supplies PyTorch/torchaudio as a matched pair in the
base layer instead of reinstalling the older CUDA wheels from the repository's
training-oriented `requirements.txt`.

Useful settings:

| Environment variable | Default |
| --- | --- |
| `COSYVOICE_MODEL_ALIAS` | `cosyvoice3-0.5b` |
| `COSYVOICE_MODEL_DIR` | `pretrained_models/Fun-CosyVoice3-0.5B` |
| `COSYVOICE_VOICES_FILE` | `api_server/voices.json` |
| `COSYVOICE_HOST` / `COSYVOICE_PORT` | `0.0.0.0` / `8000` |
| `COSYVOICE_MAX_TEXT_CHARACTERS` | `2000` |
| `COSYVOICE_MAX_CONCURRENCY` | `1` |
| `COSYVOICE_MAX_QUEUE_SIZE` | `16` |
| `COSYVOICE_REQUEST_TIMEOUT_SECONDS` | `600` |
| `COSYVOICE_FP16` / `COSYVOICE_LOAD_VLLM` | `false` / `false` |

Use exactly one Uvicorn worker. Multiple workers load multiple copies of the
model and duplicate GPU memory.

## Synthesize speech

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Authorization: Bearer replace-with-a-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3-0.5b",
    "input": "你好，这是一次 CosyVoice API 测试。",
    "voice": "default",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output result.wav
```

Optional style control uses the OpenAI-compatible plural field name:

```json
{
  "instructions": "请用四川话、开心地说这句话"
}
```

The built-in `default` voice is configured from `asset/zero_shot_prompt.wav`.
Clients never submit server-side paths. Add server-owned voices in
`api_server/voices.json`; zero-shot prompt features are cached once at model
startup for ordinary synthesis.

PCM responses are mono signed 16-bit little-endian at the model-native 24 kHz
sample rate. WAV responses contain the same samples with a complete WAV header.

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
python -m unittest discover -s api_server/tests -v
```

After starting the real service:

```bash
python -m api_server.smoke_test \
  --api-key replace-with-a-secret \
  --output smoke.wav
```
