#!/usr/bin/env python3
"""Validate and benchmark fixed-size HiFT STFT paths on SDAA."""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=150000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def measure(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.sdaa.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.sdaa.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def main() -> None:
    args = parse_args()
    n_fft = 16
    hop = 4
    torch.manual_seed(20260802)
    device = torch.device("sdaa")
    inputs = torch.randn(
        1, args.samples, device=device, dtype=torch.float16
    )
    window = torch.hann_window(
        n_fft, periodic=True, dtype=torch.float32, device=device
    )
    frequency = torch.arange(
        n_fft // 2 + 1, dtype=torch.float32, device=device
    )[:, None]
    sample = torch.arange(
        n_fft, dtype=torch.float32, device=device
    )[None, :]
    angle = 2.0 * math.pi * frequency * sample / n_fft
    real_weight = (torch.cos(angle) * window).unsqueeze(1)
    imag_weight = (-torch.sin(angle) * window).unsqueeze(1)
    combined_weight = torch.cat((real_weight, imag_weight), dim=0)
    inverse_real = torch.cos(angle) / n_fft
    inverse_real[1:-1] *= 2.0
    inverse_imag = -torch.sin(angle) / n_fft
    inverse_imag[1:-1] *= 2.0
    inverse_imag[0].zero_()
    inverse_imag[-1].zero_()
    inverse_weight = torch.cat(
        (inverse_real, inverse_imag), dim=0
    ).mul(window).unsqueeze(1)

    def native():
        return torch.stft(
            inputs,
            n_fft,
            hop,
            n_fft,
            window=window,
            return_complex=True,
        )

    def convolution():
        padded = F.pad(
            inputs[:, None].float(),
            (n_fft // 2, n_fft // 2),
            mode="reflect",
        )
        real = F.conv1d(padded, real_weight, stride=hop)
        imag = F.conv1d(padded, imag_weight, stride=hop)
        return torch.complex(real, imag)

    def combined_convolution_parts():
        padded = F.pad(
            inputs[:, None].float(),
            (n_fft // 2, n_fft // 2),
            mode="reflect",
        )
        spectrum = F.conv1d(padded, combined_weight, stride=hop)
        return spectrum[:, : n_fft // 2 + 1], spectrum[:, n_fft // 2 + 1 :]

    native_output = native()
    convolution_output = convolution()
    combined_real, combined_imag = combined_convolution_parts()
    combined_output = torch.complex(combined_real, combined_imag)
    frame_count = native_output.shape[-1]
    envelope = F.conv_transpose1d(
        torch.ones(1, 1, frame_count, device=device),
        window.square()[None, None],
        stride=hop,
    )
    inverse_envelope = envelope[..., n_fft // 2 : -n_fft // 2].reciprocal()

    def native_inverse():
        return torch.istft(
            native_output,
            n_fft,
            hop,
            n_fft,
            window=window,
        )

    def convolution_inverse():
        spectrum = torch.cat(
            (native_output.real, native_output.imag), dim=1
        )
        waveform = F.conv_transpose1d(
            spectrum, inverse_weight, stride=hop
        )
        waveform = waveform[..., n_fft // 2 : -n_fft // 2]
        return (waveform * inverse_envelope).squeeze(1)

    native_inverse_output = native_inverse()
    convolution_inverse_output = convolution_inverse()
    torch.sdaa.synchronize()
    results = {
        "native_device": str(native_output.device),
        "native_dtype": str(native_output.dtype),
        "convolution_device": str(convolution_output.device),
        "convolution_dtype": str(convolution_output.dtype),
        "output_shape": list(native_output.shape),
        "max_abs_diff": float(
            (native_output - convolution_output).abs().max().cpu()
        ),
        "mean_abs_diff": float(
            (native_output - convolution_output).abs().mean().cpu()
        ),
        "combined_max_abs_diff": float(
            (native_output - combined_output).abs().max().cpu()
        ),
        "inverse_output_shape": list(native_inverse_output.shape),
        "inverse_max_abs_diff": float(
            (native_inverse_output - convolution_inverse_output)
            .abs()
            .max()
            .cpu()
        ),
        "inverse_mean_abs_diff": float(
            (native_inverse_output - convolution_inverse_output)
            .abs()
            .mean()
            .cpu()
        ),
        "native_mean_ms": measure(native, args.warmup, args.iterations),
        "convolution_mean_ms": measure(
            convolution, args.warmup, args.iterations
        ),
        "combined_parts_mean_ms": measure(
            combined_convolution_parts, args.warmup, args.iterations
        ),
        "native_inverse_mean_ms": measure(
            native_inverse, args.warmup, args.iterations
        ),
        "convolution_inverse_mean_ms": measure(
            convolution_inverse, args.warmup, args.iterations
        ),
    }
    results["change_percent"] = (
        results["convolution_mean_ms"] / results["native_mean_ms"] - 1.0
    ) * 100.0
    results["combined_parts_change_percent"] = (
        results["combined_parts_mean_ms"] / results["native_mean_ms"]
        - 1.0
    ) * 100.0
    results["convolution_inverse_change_percent"] = (
        results["convolution_inverse_mean_ms"]
        / results["native_inverse_mean_ms"]
        - 1.0
    ) * 100.0
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
