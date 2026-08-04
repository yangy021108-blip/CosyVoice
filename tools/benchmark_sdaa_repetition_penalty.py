#!/usr/bin/env python3
"""Benchmark an SDAA repetition-only penalty fast path."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch_sdaa  # noqa: F401

from vllm.model_executor.layers.utils import apply_penalties


def repetition_only_fast(
    logits: torch.Tensor,
    prompt_tokens: torch.Tensor,
    output_tokens: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    num_seqs, vocab_size = logits.shape
    prompt_mask = torch.zeros(
        (num_seqs, vocab_size + 1),
        dtype=torch.bool,
        device=logits.device,
    )
    output_mask = torch.zeros_like(prompt_mask)
    prompt_mask.scatter_(1, prompt_tokens, True)
    output_mask.scatter_(1, output_tokens, True)
    mask = prompt_mask[:, :vocab_size] | output_mask[:, :vocab_size]
    penalty = repetition_penalties.unsqueeze(1).expand(-1, vocab_size)
    penalties = torch.where(mask, penalty, 1.0)
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    logits.mul_(scaling)
    return logits


def measure(fn, base_logits, iterations: int) -> list[float]:
    values = []
    for _ in range(iterations):
        torch.sdaa.synchronize()
        started = time.perf_counter()
        fn(base_logits.clone())
        torch.sdaa.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--vocab-size", type=int, default=6784)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(20260803)
    device = torch.device("sdaa")
    logits = torch.randn(1, args.vocab_size, device=device)
    prompt = torch.randint(
        0, args.vocab_size, (1, args.prompt_len), device=device
    )
    output = torch.randint(
        0, args.vocab_size, (1, args.output_len), device=device
    )
    presence = torch.zeros(1, device=device)
    frequency = torch.zeros(1, device=device)
    repetition = torch.full((1,), 1.1, device=device)

    def reference(value: torch.Tensor) -> torch.Tensor:
        return apply_penalties(
            value,
            prompt,
            output,
            presence,
            frequency,
            repetition,
        )

    def candidate(value: torch.Tensor) -> torch.Tensor:
        return repetition_only_fast(
            value,
            prompt,
            output,
            repetition,
        )

    for _ in range(10):
        reference(logits.clone())
        candidate(logits.clone())
    torch.sdaa.synchronize()
    expected = reference(logits.clone())
    actual = candidate(logits.clone())
    max_abs_diff = float((expected - actual).abs().max().cpu())
    exact = bool(torch.equal(expected.cpu(), actual.cpu()))

    reference_ms = measure(reference, logits, args.iterations)
    candidate_ms = measure(candidate, logits, args.iterations)
    result = {
        "iterations": args.iterations,
        "shape": list(logits.shape),
        "reference_mean_ms": statistics.fmean(reference_ms),
        "candidate_mean_ms": statistics.fmean(candidate_ms),
        "change_percent": (
            statistics.fmean(candidate_ms) / statistics.fmean(reference_ms)
            - 1.0
        ) * 100.0,
        "max_abs_diff": max_abs_diff,
        "exact": exact,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
