
"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
from typing import Optional
import weakref

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

from x_transformers.x_transformers import apply_rotary_pos_emb, rotate_half


_SDAA_FLOW_FLASH_ATTN_ENV = "COSYVOICE_SDAA_FLOW_FLASH_ATTN"
_SDAA_FLOW_FLASH_ATTN_PATH_ENV = "COSYVOICE_SDAA_FLOW_FLASH_ATTN_PATH"
_SDAA_FLOW_UNMASKED_ENV = "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH"
_SDAA_FLOW_TRUST_UNMASKED_ENV = "COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK"
_SDAA_FLOW_FUSED_NORM_ENV = "COSYVOICE_SDAA_FLOW_FUSED_NORM"
_SDAA_FLOW_PRECOMPUTE_ROPE_ENV = "COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE"
_SDAA_FLOW_FUSED_FFN_ENV = "COSYVOICE_SDAA_FLOW_FUSED_FFN"
_SDAA_FLOW_FUSED_GEMM_ENV = "COSYVOICE_SDAA_FLOW_FUSED_GEMM"
_VALIDATED_ALL_TRUE_MASKS: dict[
    int, weakref.ReferenceType[torch.Tensor]
] = {}


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _load_sdaa_flow_flash_attention() -> None:
    default_path = (
        Path(__file__).resolve().parents[3]
        / "cosyvoice"
        / "sdaa_ops"
        / "sdaa_mm_encoder_fa_poc_ext.cpython-310-x86_64-linux-gnu.so"
    )
    extension_path = Path(
        os.getenv(_SDAA_FLOW_FLASH_ATTN_PATH_ENV, str(default_path))
    ).expanduser()
    if not extension_path.is_file():
        raise FileNotFoundError(
            f"{_SDAA_FLOW_FLASH_ATTN_ENV} is enabled but the SDAA "
            f"flash-attention extension is missing: {extension_path}"
        )
    torch.ops.load_library(str(extension_path))


def _all_true_inference_mask(attn_mask: torch.Tensor | None) -> bool:
    if attn_mask is None:
        return True
    cache_key = id(attn_mask)
    cached = _VALIDATED_ALL_TRUE_MASKS.get(cache_key)
    if cached is not None and cached() is attn_mask:
        return True
    # CosyVoice API inference synthesizes one unpadded utterance and then
    # duplicates it for classifier-free guidance. Validate each mask tensor
    # once before using an attention kernel that has no padding-mask input.
    is_all_true = bool(attn_mask.all().item())
    if is_all_true:
        _VALIDATED_ALL_TRUE_MASKS[cache_key] = weakref.ref(
            attn_mask,
            lambda reference: (
                _VALIDATED_ALL_TRUE_MASKS.pop(cache_key, None)
                if _VALIDATED_ALL_TRUE_MASKS.get(cache_key) is reference
                else None
            ),
        )
    return is_all_true


def can_use_sdaa_unmasked_flow_attention(
    mask: torch.Tensor | None,
    streaming: bool,
) -> bool:
    """Return whether the API's all-valid CFG mask can be omitted safely."""
    return (
        _env_enabled(_SDAA_FLOW_FLASH_ATTN_ENV)
        and _env_enabled(_SDAA_FLOW_UNMASKED_ENV)
        and not streaming
        and mask is not None
        and mask.device.type == "sdaa"
        and mask.shape[0] == 2
        and not torch.is_grad_enabled()
        and (
            _env_enabled(_SDAA_FLOW_TRUST_UNMASKED_ENV)
            or _all_true_inference_mask(mask)
        )
    )


def can_use_sdaa_fused_flow_norm(x: torch.Tensor) -> bool:
    return (
        _env_enabled(_SDAA_FLOW_FUSED_NORM_ENV)
        and x.device.type == "sdaa"
        and x.dtype == torch.float16
        and x.shape[0] == 2
        and not torch.is_grad_enabled()
    )


@dataclass(frozen=True)
class PrecomputedRotaryEmbedding:
    cosine: torch.Tensor
    sine: torch.Tensor
    query_scale: torch.Tensor | float
    key_scale: torch.Tensor | float


def precompute_sdaa_flow_rope(
    rope,
    x: torch.Tensor,
):
    use_precomputed_rope = (
        _env_enabled(_SDAA_FLOW_PRECOMPUTE_ROPE_ENV)
        and x.device.type == "sdaa"
        and not torch.is_grad_enabled()
    )
    if not use_precomputed_rope:
        return rope

    freqs, xpos_scale = rope
    if xpos_scale is None:
        query_scale = 1.0
        key_scale = 1.0
    else:
        query_scale = xpos_scale
        key_scale = xpos_scale**-1.0
    return PrecomputedRotaryEmbedding(
        cosine=freqs.cos(),
        sine=freqs.sin(),
        query_scale=query_scale,
        key_scale=key_scale,
    )


def _apply_precomputed_rotary_pos_emb(
    tensor: torch.Tensor,
    rope: PrecomputedRotaryEmbedding,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    seq_len = tensor.shape[-2]
    cosine = rope.cosine[:, -seq_len:, :]
    sine = rope.sine[:, -seq_len:, :]
    if torch.is_tensor(scale):
        scale = scale[:, -seq_len:, :]

    if tensor.ndim == 4 and cosine.ndim == 3:
        cosine = cosine.unsqueeze(1)
        sine = sine.unsqueeze(1)
        if torch.is_tensor(scale):
            scale = scale.unsqueeze(1)

    rot_dim = cosine.shape[-1]
    rotated, unrotated = tensor[..., :rot_dim], tensor[..., rot_dim:]
    if torch.is_tensor(scale):
        rotated = (
            rotated * cosine * scale
            + rotate_half(rotated) * sine * scale
        )
    else:
        rotated = rotated * cosine + rotate_half(rotated) * sine
    return torch.cat((rotated, unrotated), dim=-1).type(tensor.dtype)


def _scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    use_sdaa_flash = (
        _env_enabled(_SDAA_FLOW_FLASH_ATTN_ENV)
        and query.device.type == "sdaa"
        and query.dtype in (torch.float16, torch.bfloat16)
        and not torch.is_grad_enabled()
        and query.shape == key.shape == value.shape
        and query.shape[0] == 2
        and _all_true_inference_mask(attn_mask)
    )
    if not use_sdaa_flash:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

    _load_sdaa_flow_flash_attention()
    scale = 1.0 / math.sqrt(query.shape[-1])
    return torch.ops._C_sdaa_poc.mm_encoder_flash_attention_fixed(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        scale,
    ).transpose(1, 2)


def _gated_residual_layer_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm: nn.LayerNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    use_sdaa_fused_norm = can_use_sdaa_fused_flow_norm(residual)
    if not use_sdaa_fused_norm:
        residual = residual + gate.unsqueeze(1) * update
        normalized = (
            norm(residual) * (1 + scale[:, None]) + shift[:, None]
        )
        return residual, normalized

    _load_sdaa_flow_flash_attention()
    gated_update = gate.unsqueeze(1) * update
    normalized, residual = torch.ops._C_sdaa_poc.flow_fused_norm(
        gated_update,
        residual,
        scale,
        shift,
        norm.eps,
    )
    return residual, normalized


# raw wav to mel spec
class MelSpec(nn.Module):
    def __init__(
        self,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        normalize=False,
        power=1,
        norm=None,
        center=True,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels

        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=filter_length,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mel_channels,
            power=power,
            center=center,
            normalized=normalize,
            norm=norm,
        )

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, inp):
        if len(inp.shape) == 3:
            inp = inp.squeeze(1)  # 'b 1 nw -> b nw'

        assert len(inp.shape) == 2

        if self.dummy.device != inp.device:
            self.to(inp.device)

        mel = self.mel_stft(inp)
        mel = mel.clamp(min=1e-5).log()
        return mel


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# convolutional position embedding


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None):  # noqa: F722
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out


class CausalConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )

    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None):  # noqa: F722
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        x = self.conv1(x)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        x = self.conv2(x)
        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out


# rotary positional embedding related


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


# Global Response Normalization layer (Instance Normalization ?)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# AdaLayerNormZero
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def modulation(self, emb):
        emb = self.linear(self.silu(emb))
        return torch.chunk(emb, 6, dim=1)

    def forward(self, x, emb=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(emb)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNormZero for final layer
# return only with modulated x for attn input, cuz no more mlp modulation


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# FeedForward


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))
        self.register_buffer(
            "_sdaa_fused_weight_in", None, persistent=False
        )
        self.register_buffer(
            "_sdaa_fused_weight_out", None, persistent=False
        )
        self.register_buffer(
            "_sdaa_fused_bias_in", None, persistent=False
        )
        self.register_buffer(
            "_sdaa_fused_bias_out", None, persistent=False
        )

    def _prepare_sdaa_fused_weights(self) -> None:
        if self._sdaa_fused_weight_in is not None:
            return
        import vllm_sdaa._C  # noqa: F401

        linear_in = self.ff[0][0]
        linear_out = self.ff[2]
        original_weight_in = linear_in.weight.to(dtype=torch.float16)
        original_weight_out = linear_out.weight.to(dtype=torch.float16)
        weight_in = torch.empty_like(original_weight_in)
        weight_out = torch.empty_like(original_weight_out)
        torch.ops._C_sdaa.weight_permute(
            weight_in,
            None,
            original_weight_in.permute(1, 0).contiguous(),
            2,
            0,
            -1,
        )
        torch.ops._C_sdaa.weight_permute(
            weight_out,
            None,
            original_weight_out.permute(1, 0).contiguous(),
            3,
            0,
            -1,
        )
        self._sdaa_fused_weight_in = weight_in
        self._sdaa_fused_weight_out = weight_out
        self._sdaa_fused_bias_in = (
            linear_in.bias.to(dtype=torch.float16)
            if linear_in.bias is not None
            else None
        )
        self._sdaa_fused_bias_out = (
            linear_out.bias.to(dtype=torch.float16)
            if linear_out.bias is not None
            else None
        )

    def _can_use_sdaa_fused_ffn(self, x: torch.Tensor) -> bool:
        threshold = int(os.getenv("COSYVOICE_SDAA_FLOW_FUSED_FFN_MIN_ROWS", "800"))
        return (
            _env_enabled(_SDAA_FLOW_FUSED_FFN_ENV)
            and x.device.type == "sdaa"
            and x.dtype == torch.float16
            and not self.training
            and not torch.is_grad_enabled()
            and x.shape[-1] == 1024
            and x.numel() // x.shape[-1] >= threshold
        )

    def forward(self, x):
        if self._can_use_sdaa_fused_ffn(x):
            self._prepare_sdaa_fused_weights()
            linear_in = self.ff[0][0]
            linear_out = self.ff[2]
            flattened = x.reshape(-1, x.shape[-1])
            output = torch.empty_like(flattened)
            torch.ops._C_sdaa.ffnv2(
                output,
                flattened,
                self._sdaa_fused_weight_in,
                self._sdaa_fused_weight_out,
                self._sdaa_fused_bias_in,
                "default",
                "gelu",
            )
            if self._sdaa_fused_bias_out is not None:
                output.add_(self._sdaa_fused_bias_out)
            return output.view(*x.shape[:-1], linear_out.out_features)
        return self.ff(x)


def _sdaa_flow_linear(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    min_rows = int(
        os.getenv("COSYVOICE_SDAA_FLOW_FUSED_GEMM_MIN_ROWS", "512")
    )
    rows = x.numel() // x.shape[-1]
    use_fused_gemm = (
        _env_enabled(_SDAA_FLOW_FUSED_GEMM_ENV)
        and x.device.type == "sdaa"
        and x.dtype == torch.float16
        and not linear.training
        and not torch.is_grad_enabled()
        and rows >= min_rows
        and linear.in_features % 32 == 0
        and linear.out_features % 32 == 0
    )
    if not use_fused_gemm:
        return linear(x)

    packed_weight = linear._sdaa_fused_gemm_weight
    if packed_weight is None:
        import vllm_sdaa._C  # noqa: F401

        packed_weight = torch.ops._C_sdaa.blas_gemm_fusion_weight(
            linear.weight.detach()
            .to(dtype=torch.float16)
            .T.contiguous()
            .cpu()
        ).to(device=x.device)
        linear._sdaa_fused_gemm_weight = packed_weight
        linear._sdaa_fused_gemm_bias = (
            linear.bias.detach().to(device=x.device, dtype=x.dtype)
            if linear.bias is not None
            else None
        )

    flattened = x.reshape(-1, x.shape[-1])
    output = torch.ops._C_sdaa.blas_gemm_fusion(
        flattened,
        packed_weight,
        False,
        False,
    )
    if linear._sdaa_fused_gemm_bias is not None:
        output.add_(linear._sdaa_fused_gemm_bias)
    return output.view(*x.shape[:-1], linear.out_features)


# Attention with possible joint part
# modified from diffusers/src/diffusers/models/attention_processor.py


class Attention(nn.Module):
    def __init__(
        self,
        processor: JointAttnProcessor | AttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,  # if not None -> joint attention
        context_pre_only=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if self.context_dim is not None:
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if self.context_pre_only is not None:
                self.to_q_c = nn.Linear(context_dim, self.inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, dim)

        fused_gemm_linears = [self.to_q, self.to_k, self.to_v, self.to_out[0]]
        if hasattr(self, "to_q_c"):
            fused_gemm_linears.append(self.to_q_c)
        if hasattr(self, "to_k_c"):
            fused_gemm_linears.append(self.to_k_c)
        if hasattr(self, "to_v_c"):
            fused_gemm_linears.append(self.to_v_c)
        if hasattr(self, "to_out_c"):
            fused_gemm_linears.append(self.to_out_c)
        for linear in fused_gemm_linears:
            linear.register_buffer(
                "_sdaa_fused_gemm_weight", None, persistent=False
            )
            linear.register_buffer(
                "_sdaa_fused_gemm_bias", None, persistent=False
            )

    def forward(
        self,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b n d"] = None,  # context c  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        else:
            return self.processor(self, x, mask=mask, rope=rope)


# Attention processor


class AttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        # `sample` projections.
        query = _sdaa_flow_linear(attn.to_q, x)
        key = _sdaa_flow_linear(attn.to_k, x)
        value = _sdaa_flow_linear(attn.to_v, x)

        # apply rotary position embedding
        if rope is not None:
            if isinstance(rope, PrecomputedRotaryEmbedding):
                query = _apply_precomputed_rotary_pos_emb(
                    query,
                    rope,
                    rope.query_scale,
                )
                key = _apply_precomputed_rotary_pos_emb(
                    key,
                    rope,
                    rope.key_scale,
                )
            else:
                freqs, xpos_scale = rope
                q_xpos_scale, k_xpos_scale = (
                    (xpos_scale, xpos_scale**-1.0)
                    if xpos_scale is not None
                    else (1.0, 1.0)
                )
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = mask
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
                attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = _scaled_dot_product_attention(query, key, value, attn_mask)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # linear proj
        x = _sdaa_flow_linear(attn.to_out[0], x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            else:
                mask = mask[:, 0, -1].unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Joint Attention processor for MM-DiT
# modified from diffusers/src/diffusers/models/attention_processor.py


class JointAttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b nt d"] = None,  # context c, here text # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.FloatTensor:
        residual = x

        batch_size = c.shape[0]

        # `sample` projections.
        query = _sdaa_flow_linear(attn.to_q, x)
        key = _sdaa_flow_linear(attn.to_k, x)
        value = _sdaa_flow_linear(attn.to_v, x)

        # `context` projections.
        c_query = _sdaa_flow_linear(attn.to_q_c, c)
        c_key = _sdaa_flow_linear(attn.to_k_c, c)
        c_value = _sdaa_flow_linear(attn.to_v_c, c)

        # apply rope for context and noised input independently
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (
                (xpos_scale, xpos_scale**-1.0)
                if xpos_scale is not None
                else (1.0, 1.0)
            )
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_xpos_scale, k_xpos_scale = (
                (xpos_scale, xpos_scale**-1.0)
                if xpos_scale is not None
                else (1.0, 1.0)
            )
            c_query = apply_rotary_pos_emb(
                c_query, freqs, q_xpos_scale
            )
            c_key = apply_rotary_pos_emb(c_key, freqs, k_xpos_scale)

        # attention
        query = torch.cat([query, c_query], dim=1)
        key = torch.cat([key, c_key], dim=1)
        value = torch.cat([value, c_value], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = F.pad(mask, (0, c.shape[1]), value=True)  # no mask for c (text)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = _scaled_dot_product_attention(query, key, value, attn_mask)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # Split the attention outputs.
        x, c = (
            x[:, : residual.shape[1]],
            x[:, residual.shape[1]:],
        )

        # linear proj
        x = _sdaa_flow_linear(attn.to_out[0], x)
        # dropout
        x = attn.to_out[1](x)
        if not attn.context_pre_only:
            c = _sdaa_flow_linear(attn.to_out_c, c)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
            # c = c.masked_fill(~mask, 0.)  # no mask for c (text)

        return x, c


# DiT Block


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor=AttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x, ff_norm = _gated_residual_layer_norm(
            x,
            attn_output,
            gate_msa,
            scale_mlp,
            shift_mlp,
            self.ff_norm,
        )
        ff_output = self.ff(ff_norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x

    def forward_prepared(
        self,
        x,
        t,
        prepared,
        mask=None,
        rope=None,
        next_block=None,
    ):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = prepared
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x, ff_norm = _gated_residual_layer_norm(
            x,
            attn_output,
            gate_msa,
            scale_mlp,
            shift_mlp,
            self.ff_norm,
        )
        ff_output = self.ff(ff_norm)

        if next_block is None:
            x = x + gate_mlp.unsqueeze(1) * ff_output
            return x, None

        (
            next_shift_msa,
            next_scale_msa,
            next_gate_msa,
            next_shift_mlp,
            next_scale_mlp,
            next_gate_mlp,
        ) = next_block.attn_norm.modulation(t)
        x, next_norm = _gated_residual_layer_norm(
            x,
            ff_output,
            gate_mlp,
            next_scale_msa,
            next_shift_msa,
            next_block.attn_norm.norm,
        )
        return x, (
            next_norm,
            next_gate_msa,
            next_shift_mlp,
            next_scale_mlp,
            next_gate_mlp,
        )


# MMDiT Block https://arxiv.org/abs/2403.03206


class MMDiTBlock(nn.Module):
    r"""
    modified from diffusers/src/diffusers/models/attention.py

    notes.
    _c: context related. text, cond, etc. (left part in sd3 fig2.b)
    _x: noised input related. (right part)
    context_pre_only: last layer only do prenorm + modulation cuz no more ffn
    """

    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, context_pre_only=False):
        super().__init__()

        self.context_pre_only = context_pre_only

        self.attn_norm_c = AdaLayerNormZero_Final(dim) if context_pre_only else AdaLayerNormZero(dim)
        self.attn_norm_x = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor=JointAttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=dim,
            context_pre_only=context_pre_only,
        )

        if not context_pre_only:
            self.ff_norm_c = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_c = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")
        else:
            self.ff_norm_c = None
            self.ff_c = None
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, c, t, mask=None, rope=None, c_rope=None):  # x: noised input, c: context, t: time embedding
        # pre-norm & modulation for attention input
        if self.context_pre_only:
            norm_c = self.attn_norm_c(c, t)
        else:
            norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)

        # attention
        x_attn_output, c_attn_output = self.attn(x=norm_x, c=norm_c, mask=mask, rope=rope, c_rope=c_rope)

        # process attention output for context c
        if self.context_pre_only:
            c = None
        else:  # if not last layer
            c = c + c_gate_msa.unsqueeze(1) * c_attn_output

            norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            c_ff_output = self.ff_c(norm_c)
            c = c + c_gate_mlp.unsqueeze(1) * c_ff_output

        # process attention output for input x
        x = x + x_gate_msa.unsqueeze(1) * x_attn_output

        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x_ff_output = self.ff_x(norm_x)
        x = x + x_gate_mlp.unsqueeze(1) * x_ff_output

        return c, x


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: float["b"]):  # noqa: F821
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time
