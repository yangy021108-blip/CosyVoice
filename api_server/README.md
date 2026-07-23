# CosyVoice HTTP API

[中文使用指南](README_zh.md)

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

The base image uses the regular PyTorch backend. Build the explicit vLLM
variant when `COSYVOICE_LOAD_VLLM=true` is required:

```bash
docker build --build-arg INSTALL_VLLM=true \
  -t cosyvoice-api:vllm -f api_server/Dockerfile api_server
```

The vLLM layer is pinned separately in `requirements-vllm.txt`. Enabling vLLM
without installing that layer fails at startup with an actionable error.

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
