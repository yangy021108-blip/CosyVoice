#!/usr/bin/env python3
"""Rotate the local CosyVoice API key without printing the secret."""

from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "cosyvoice_sdaa_vllm_api.env",
    )
    args = parser.parse_args()

    env_file = args.env_file.resolve()
    values: dict[str, str] = {}
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    values["COSYVOICE_API_KEY"] = secrets.token_hex(32)
    if "COSYVOICE_OUTPUT_DIR" not in values:
        values["COSYVOICE_OUTPUT_DIR"] = str(
            env_file.parent / "cosyvoice_api_outputs"
        )

    temp_file = env_file.with_name(f".{env_file.name}.tmp")
    previous_umask = os.umask(0o077)
    try:
        temp_file.write_text(
            "".join(f"{key}={value}\n" for key, value in values.items()),
            encoding="utf-8",
        )
        temp_file.chmod(0o600)
        os.replace(temp_file, env_file)
    finally:
        os.umask(previous_umask)
        if temp_file.exists():
            temp_file.unlink()

    print(f"Rotated API key in {env_file}; secret was not printed.")


if __name__ == "__main__":
    main()
