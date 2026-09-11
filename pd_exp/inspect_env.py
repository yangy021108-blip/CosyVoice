#!/usr/bin/env python3
"""Emit machine-readable versions and CUDA P2P capability."""

from __future__ import annotations

import json
import platform

import nixl
import torch
import transformers
import vllm


def main() -> int:
    pairs = {}
    for left in range(torch.cuda.device_count()):
        for right in range(left + 1, torch.cuda.device_count()):
            pairs[f"{left}-{right}"] = torch.cuda.can_device_access_peer(left, right)
    print(json.dumps({
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "vllm": vllm.__version__,
        "transformers": transformers.__version__,
        "nixl_module": nixl.__file__,
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "p2p": pairs,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
