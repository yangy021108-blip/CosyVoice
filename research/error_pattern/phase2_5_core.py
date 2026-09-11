#!/usr/bin/env python3
"""Research-only primitives for separating CosyVoice LLM and acoustic stages."""

from __future__ import annotations

import contextlib
import os
import hashlib
import json
import math
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from api_server.audio_codec import analyze_audio_quality, encode_wav, float_to_pcm16
from api_server.voice_store import VoiceSpec, VoiceStore


SCHEMA_VERSION = 1


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def load_research_backend(
    *,
    model_dir: Path,
    voices_json: Path,
    voice_id: str,
    repository_root: Path,
    load_vllm: bool,
    fp16: bool,
) -> tuple[Any, VoiceSpec]:
    """Load one offline model and register the same server-owned voice."""
    local_wetext = os.environ.get("COSY_PD_WETEXT_LOCAL", "").strip()
    if local_wetext:
        import wetext.wetext as _wetext_impl
        _wetext_impl.snapshot_download = lambda _name: local_wetext
    if os.environ.get("COSY_PD_SOUND_FILE_FALLBACK") == "1":
        import torchaudio
        import soundfile as sf
        def _soundfile_load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None, buffer_size=4096, backend=None):
            frames = -1 if num_frames is None or num_frames < 0 else num_frames
            data, sample_rate = sf.read(path, start=frame_offset, frames=frames, dtype="float32", always_2d=True)
            array = data.T.copy() if channels_first else data.copy()
            return torch.from_numpy(np.asarray(array)), sample_rate
        torchaudio.load = _soundfile_load
    matcha_path = repository_root / "third_party" / "Matcha-TTS"
    if str(matcha_path) not in sys.path:
        sys.path.append(str(matcha_path))

    from cosyvoice.cli.cosyvoice import AutoModel

    voice_store = VoiceStore.from_json(voices_json, repository_root)
    voice = voice_store.get(voice_id)
    if voice is None:
        raise ValueError(f"Unknown voice ID: {voice_id}")
    if voice.mode != "zero_shot":
        raise ValueError("Phase 2.5 controls currently require a zero-shot voice")
    if voice.prompt_audio is None or voice.prompt_text is None:
        raise ValueError(f"Voice {voice_id!r} is missing prompt configuration")

    backend = AutoModel(
        model_dir=str(model_dir),
        load_vllm=load_vllm,
        load_trt=False,
        fp16=fp16,
    )
    backend.add_zero_shot_spk(
        voice.prompt_text,
        str(voice.prompt_audio),
        voice.voice_id,
    )
    return backend, voice


def build_model_inputs(
    backend: Any,
    *,
    text: str,
    voice_id: str,
    text_frontend: bool,
) -> tuple[list[str], list[dict[str, torch.Tensor]]]:
    chunks = list(
        backend.frontend.text_normalize(
            text,
            split=True,
            text_frontend=text_frontend,
        )
    )
    if not chunks:
        raise ValueError("Frontend produced no text chunks")
    if any(not isinstance(chunk, str) for chunk in chunks):
        raise TypeError("Streaming text generators are outside Phase 2.5 scope")

    model_inputs = [
        backend.frontend.frontend_zero_shot(
            chunk,
            "",
            "",
            backend.sample_rate,
            voice_id,
        )
        for chunk in chunks
    ]
    return chunks, model_inputs


def filter_silent_tokens(
    raw_tokens: list[int],
    silent_tokens: list[int],
    *,
    maximum_consecutive: int = 5,
) -> tuple[list[int], list[int]]:
    """Mirror CosyVoiceModel.llm_job's silent-token suppression exactly."""
    filtered: list[int] = []
    dropped_indexes: list[int] = []
    current_silent_count = 0
    silent_set = set(silent_tokens)
    for index, token in enumerate(raw_tokens):
        if token in silent_set:
            current_silent_count += 1
            if current_silent_count > maximum_consecutive:
                dropped_indexes.append(index)
                continue
        else:
            current_silent_count = 0
        filtered.append(token)
    return filtered, dropped_indexes


def _prepare_lm_input(
    llm: Any,
    model_input: dict[str, torch.Tensor],
    *,
    min_token_text_ratio: float,
    max_token_text_ratio: float,
) -> tuple[torch.Tensor, int, int, dict[str, Any]]:
    device = llm.speech_embedding.weight.device
    target_text = model_input["text"].to(device)
    prompt_text = model_input["prompt_text"].to(device)
    prompt_speech = model_input["llm_prompt_speech_token"].to(device)

    target_count = int(target_text.shape[1])
    prompt_text_count = int(prompt_text.shape[1])
    prompt_speech_count = int(prompt_speech.shape[1])
    combined_text = torch.concat([prompt_text, target_text], dim=1)
    if llm.__class__.__name__ == "CosyVoice3LM" and 151646 not in combined_text:
        raise ValueError("<|endofprompt|> token was not found in prompt/target text")

    text_embedding = llm.llm.model.model.embed_tokens(combined_text)
    sos_embedding = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
    task_embedding = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
    if prompt_speech_count:
        prompt_speech_embedding = llm.speech_embedding(prompt_speech)
    else:
        prompt_speech_embedding = torch.zeros(
            1,
            0,
            llm.llm_input_size,
            dtype=text_embedding.dtype,
            device=device,
        )
    lm_input = torch.concat(
        [sos_embedding, text_embedding, task_embedding, prompt_speech_embedding],
        dim=1,
    )
    minimum_tokens = int(target_count * min_token_text_ratio)
    maximum_tokens = int(target_count * max_token_text_ratio)

    cursor = 0
    spans: dict[str, Any] = {}
    spans["sos"] = [cursor, cursor + 1]
    cursor += 1
    spans["prompt_text"] = [cursor, cursor + prompt_text_count]
    cursor += prompt_text_count
    spans["target_text"] = [cursor, cursor + target_count]
    cursor += target_count
    spans["task"] = [cursor, cursor + 1]
    cursor += 1
    spans["prompt_speech"] = [cursor, cursor + prompt_speech_count]
    cursor += prompt_speech_count
    spans["generation_start"] = cursor
    spans["total_prompt_context_tokens"] = cursor
    spans["prompt_text_token_ids"] = prompt_text.squeeze(0).tolist()
    spans["target_text_token_ids"] = target_text.squeeze(0).tolist()
    spans["prompt_speech_token_ids"] = prompt_speech.squeeze(0).tolist()
    return lm_input, minimum_tokens, maximum_tokens, spans


def _serialize_logprob_step(
    step_logprobs: Any,
    chosen_token: int,
) -> dict[str, Any] | None:
    if not step_logprobs:
        return None
    entries: list[dict[str, Any]] = []
    for token_id, value in step_logprobs.items():
        logprob = float(value.logprob)
        entries.append(
            {
                "token_id": int(token_id),
                "logprob": logprob,
                "rank": (
                    int(value.rank) if getattr(value, "rank", None) is not None else None
                ),
                "decoded_token": getattr(value, "decoded_token", None),
            }
        )
    entries.sort(key=lambda item: item["logprob"], reverse=True)
    probabilities = [math.exp(item["logprob"]) for item in entries]
    returned_mass = sum(probabilities)
    lower_bound_entropy = -sum(
        probability * item["logprob"]
        for probability, item in zip(probabilities, entries)
    )
    if returned_mass > 0:
        normalized = [probability / returned_mass for probability in probabilities]
        renormalized_entropy = -sum(
            probability * math.log(max(probability, 1e-45))
            for probability in normalized
        )
    else:
        renormalized_entropy = None
    chosen = next(
        (item for item in entries if item["token_id"] == chosen_token), None
    )
    return {
        "chosen_logprob": None if chosen is None else chosen["logprob"],
        "chosen_rank": None if chosen is None else chosen["rank"],
        "returned_probability_mass": returned_mass,
        "approx_entropy_truncated_lower_bound": lower_bound_entropy,
        "approx_entropy_topk_renormalized": renormalized_entropy,
        "top_logprobs": entries,
        "entropy_is_exact": False,
    }


@torch.inference_mode()
def generate_chunk_trajectory(
    backend: Any,
    model_input: dict[str, torch.Tensor],
    *,
    chunk_index: int,
    normalized_text: str,
    llm_seed: int,
    top_k: int,
    logprobs: int | None,
    min_token_text_ratio: float,
    max_token_text_ratio: float,
) -> dict[str, Any]:
    """Generate one raw vLLM trajectory without entering Flow/HiFT."""
    llm = backend.model.llm
    if not hasattr(llm, "vllm"):
        raise RuntimeError("Phase 2.5 token capture currently requires vLLM")
    lm_input, minimum_tokens, maximum_tokens, spans = _prepare_lm_input(
        llm,
        model_input,
        min_token_text_ratio=min_token_text_ratio,
        max_token_text_ratio=max_token_text_ratio,
    )
    llm.set_inference_seed(llm_seed)

    from vllm import SamplingParams

    sampling_kwargs: dict[str, Any] = {
        "top_k": top_k,
        "stop_token_ids": llm.stop_token_ids,
        "min_tokens": minimum_tokens,
        "max_tokens": maximum_tokens,
        "seed": llm_seed,
    }
    if logprobs is not None:
        sampling_kwargs["logprobs"] = logprobs
    sampling_params = SamplingParams(**sampling_kwargs)
    request_id = f"phase2_5-{uuid.uuid4()}"
    prompt_embeds = lm_input.squeeze(0).to(torch.bfloat16).to(lm_input.device)

    started = time.perf_counter()
    with llm.lock:
        llm.vllm.add_request(
            request_id,
            {"prompt_embeds": prompt_embeds},
            sampling_params,
        )
    final_output = None
    # One engine step normally emits one token in this single-request runner,
    # but scheduler bookkeeping can also yield an empty step. Keep the bound
    # finite without assuming a strict one-step/one-token relationship.
    maximum_steps = maximum_tokens * 4 + 128
    for _ in range(maximum_steps):
        with llm.lock:
            request_outputs = llm.vllm.step()
        for request_output in request_outputs:
            if request_output.request_id != request_id:
                raise RuntimeError(
                    "Unexpected concurrent vLLM request in offline research runner"
                )
            final_output = request_output
        if final_output is not None and final_output.finished:
            break
    if final_output is None or not final_output.finished:
        raise RuntimeError("vLLM request did not finish within the expected steps")

    completion = final_output.outputs[0]
    returned_tokens = [int(token) for token in completion.token_ids]
    returned_logprobs = list(completion.logprobs or [])
    stop_token = getattr(completion, "stop_reason", None)
    if isinstance(stop_token, str) and stop_token.isdigit():
        stop_token = int(stop_token)
    stop_index = next(
        (
            index
            for index, token in enumerate(returned_tokens)
            if token in set(llm.stop_token_ids)
        ),
        None,
    )
    if stop_index is not None:
        if stop_token is None:
            stop_token = returned_tokens[stop_index]
        raw_tokens = returned_tokens[:stop_index]
        returned_logprobs = returned_logprobs[:stop_index]
    else:
        raw_tokens = returned_tokens
    if any(token >= llm.speech_token_size for token in raw_tokens):
        raise RuntimeError("Reserved stop token remained inside raw speech trajectory")

    filtered_tokens, dropped_indexes = filter_silent_tokens(
        raw_tokens,
        backend.model.silent_tokens,
    )
    token_statistics: list[dict[str, Any]] = []
    for index, token in enumerate(raw_tokens):
        metrics = (
            _serialize_logprob_step(returned_logprobs[index], token)
            if index < len(returned_logprobs)
            else None
        )
        token_statistics.append(
            {
                "step": index,
                "effective_position": spans["generation_start"] + index,
                "token_id": token,
                "dropped_by_silent_filter": index in set(dropped_indexes),
                "logprob_metrics": metrics,
            }
        )

    finish_reason = getattr(completion, "finish_reason", None)
    max_length_reached = len(returned_tokens) >= maximum_tokens
    if max_length_reached:
        stop_reason = "max_tokens"
    elif stop_token is not None:
        stop_reason = "reserved_stop_token"
    elif finish_reason is not None:
        stop_reason = str(finish_reason)
    else:
        stop_reason = "unknown"

    return {
        "chunk_index": chunk_index,
        "frontend_normalized_text": normalized_text,
        "spans": spans,
        "raw_speech_tokens": raw_tokens,
        "filtered_speech_tokens": filtered_tokens,
        "dropped_silent_token_indexes": dropped_indexes,
        "raw_speech_token_count": len(raw_tokens),
        "filtered_speech_token_count": len(filtered_tokens),
        "stop_token": stop_token,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "max_length_reached": max_length_reached,
        "generation_steps": len(raw_tokens),
        "generation_latency_seconds": time.perf_counter() - started,
        "token_statistics": token_statistics,
        "sampling": {
            "top_k": top_k,
            "min_tokens": minimum_tokens,
            "max_tokens": maximum_tokens,
            "seed": llm_seed,
            "requested_logprobs": logprobs,
            "entropy_is_exact": False,
        },
    }


def generate_trajectory(
    backend: Any,
    *,
    sample: dict[str, Any],
    voice_id: str,
    llm_seed: int,
    repeat_index: int,
    text_frontend: bool,
    top_k: int,
    logprobs: int | None,
    min_token_text_ratio: float,
    max_token_text_ratio: float,
) -> dict[str, Any]:
    chunks, model_inputs = build_model_inputs(
        backend,
        text=sample["text"],
        voice_id=voice_id,
        text_frontend=text_frontend,
    )
    chunk_records = [
        generate_chunk_trajectory(
            backend,
            model_input,
            chunk_index=index,
            normalized_text=chunk,
            llm_seed=llm_seed,
            top_k=top_k,
            logprobs=logprobs,
            min_token_text_ratio=min_token_text_ratio,
            max_token_text_ratio=max_token_text_ratio,
        )
        for index, (chunk, model_input) in enumerate(zip(chunks, model_inputs))
    ]
    trajectory_payload = [
        {
            "chunk_index": chunk["chunk_index"],
            "filtered_speech_tokens": chunk["filtered_speech_tokens"],
            "stop_token": chunk["stop_token"],
            "stop_reason": chunk["stop_reason"],
        }
        for chunk in chunk_records
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generation_id": (
            f"{sample['sample_id']}__llm_{llm_seed}__repeat_{repeat_index}"
        ),
        "sample_id": sample["sample_id"],
        "challenge_category": sample.get("challenge_category"),
        "length_group": sample.get("length_group"),
        "tags": sample.get("tags", []),
        "text": sample["text"],
        "voice": voice_id,
        "llm_seed": llm_seed,
        "repeat_index": repeat_index,
        "text_frontend_requested": text_frontend,
        "text_frontend_backend": backend.frontend.text_frontend or "none",
        "frontend_chunks": chunks,
        "chunks": chunk_records,
        "trajectory_sha256": stable_json_hash(trajectory_payload),
    }


@contextlib.contextmanager
def isolated_flow_seed(backend: Any, seed: int) -> Iterator[dict[str, Any]]:
    """Set explicit CFM noise for one offline decode and restore all state.

    CosyVoice3's ``CausalConditionalCFM`` does not draw fresh noise during
    inference. It creates ``rand_noise`` once with seed zero in ``__init__``.
    For seed zero this reproduces the production buffer; non-zero seeds are a
    research-only counterfactual used to measure acoustic sensitivity.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    decoder = getattr(getattr(backend.model, "flow", None), "decoder", None)
    original_noise = getattr(decoder, "rand_noise", None)
    noise_metadata: dict[str, Any] = {
        "flow_noise_mode": "global_rng",
        "flow_noise_matches_production": None,
        "flow_noise_sha256": None,
    }
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if isinstance(original_noise, torch.Tensor):
                generator = torch.Generator(device=original_noise.device)
                generator.manual_seed(seed)
                seeded_noise = torch.randn(
                    original_noise.shape,
                    dtype=original_noise.dtype,
                    device=original_noise.device,
                    generator=generator,
                )
                decoder.rand_noise = seeded_noise
                noise_metadata = {
                    "flow_noise_mode": "seeded_causal_cfm_override",
                    "flow_noise_matches_production": bool(
                        torch.equal(seeded_noise, original_noise)
                    ),
                    "flow_noise_sha256": sha256_bytes(
                        seeded_noise.detach().cpu().numpy().tobytes()
                    ),
                }
            yield noise_metadata
    finally:
        if isinstance(original_noise, torch.Tensor):
            decoder.rand_noise = original_noise
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@torch.inference_mode()
def decode_fixed_trajectory(
    backend: Any,
    *,
    trajectory: dict[str, Any],
    flow_seed: int,
    voice_id: str,
    text_frontend: bool,
    speed: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode saved filtered speech tokens with one isolated Flow RNG seed."""
    chunks, model_inputs = build_model_inputs(
        backend,
        text=trajectory["text"],
        voice_id=voice_id,
        text_frontend=text_frontend,
    )
    if chunks != trajectory["frontend_chunks"]:
        raise ValueError("Frontend chunks changed since speech-token capture")
    if len(model_inputs) != len(trajectory["chunks"]):
        raise ValueError("Frontend chunk count differs from saved trajectory")

    decoded_chunks: list[np.ndarray] = []
    started = time.perf_counter()
    flow_seed_metadata: dict[str, Any] = {}
    with isolated_flow_seed(backend, flow_seed) as active_flow_seed_metadata:
        flow_seed_metadata = active_flow_seed_metadata
        for chunk_record, model_input in zip(trajectory["chunks"], model_inputs):
            expected_target_ids = chunk_record["spans"]["target_text_token_ids"]
            actual_target_ids = model_input["text"].squeeze(0).tolist()
            if actual_target_ids != expected_target_ids:
                raise ValueError("Frontend target token IDs changed before decode")
            filtered_tokens = chunk_record["filtered_speech_tokens"]
            if not filtered_tokens:
                raise ValueError("Cannot decode an empty filtered token trajectory")
            token_tensor = torch.tensor(filtered_tokens, dtype=torch.int32).unsqueeze(0)
            session_id = f"phase2_5-decode-{uuid.uuid4()}"
            with backend.model.lock:
                backend.model.hift_cache_dict[session_id] = None
            try:
                speech = backend.model.token2wav(
                    token=token_tensor,
                    prompt_token=model_input["flow_prompt_speech_token"],
                    prompt_feat=model_input["prompt_speech_feat"],
                    embedding=model_input["flow_embedding"],
                    token_offset=0,
                    uuid=session_id,
                    stream=False,
                    finalize=True,
                    speed=speed,
                )
                decoded_chunks.append(
                    speech.detach().cpu().numpy().astype(np.float32).reshape(-1)
                )
            finally:
                with backend.model.lock:
                    backend.model.hift_cache_dict.pop(session_id, None)
    waveform = np.concatenate(decoded_chunks)
    quality = analyze_audio_quality(
        waveform,
        trajectory["text"],
        sample_rate=int(backend.sample_rate),
        speed=speed,
    )
    metadata = {
        "flow_seed": flow_seed,
        "decode_latency_seconds": time.perf_counter() - started,
        "sample_rate": int(backend.sample_rate),
        "audio_duration_seconds": len(waveform) / int(backend.sample_rate),
        "quality_acceptable": quality.acceptable,
        "quality_reason": quality.reason,
        "silent_frame_ratio": quality.silent_frame_ratio,
        "voiced_seconds_per_text_unit": quality.voiced_seconds_per_text_unit,
        **flow_seed_metadata,
    }
    return waveform, metadata


def write_waveform(path: Path, waveform: np.ndarray, sample_rate: int) -> dict[str, Any]:
    pcm = float_to_pcm16(waveform)
    wav_bytes = encode_wav(pcm, sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes)
    return {
        "audio_path": str(path.resolve()),
        "wav_sha256": sha256_bytes(wav_bytes),
        "pcm_sha256": sha256_bytes(pcm.tobytes()),
        "clipping_ratio": float(np.mean(np.abs(pcm.astype(np.int32)) >= 32767)),
    }
