"""NVFP4-weight, BF16-activation routed-expert MoE for Hopper.

The op consumes the NVFP4 Marlin layout written by
``prepare_moe_nvfp4_layer_for_marlin``, so Marlin stays a zero-copy fallback
for shapes and row counts the optimized kernel does not take. Per local
expert ``e``, with hidden size ``H`` and intermediate size ``I``:

- ``w13[e]``: ``[H/16, 4I]`` int32, the ``[2I, H]`` gate-then-up FP4 codes
  repacked by ``gptq_marlin_repack``; ``w2[e]``: ``[I/16, 2H]`` int32.
- ``w13_scale[e]``: ``[H/16, 2I]`` float8_e4m3fn block scales (one per 16 K)
  after ``marlin_permute_scales`` and ``nvfp4_marlin_process_scales``;
  ``w2_scale[e]``: ``[I/16, H]``.
- ``w13_global_scale[e]``, ``w2_global_scale[e]``: bfloat16 global scales
  after ``nvfp4_marlin_process_global_scale``.

For each token ``t`` and slot ``j`` whose expert is local::

    y = W2[e] @ silu_and_mul(W13[e] @ x[t])
    out[t] = sum_j topk_weights[t, j] * norm(y)

with GEMM outputs and activations rounded to BF16 between stages, and
``norm`` an optional per-expert RMSNorm epilogue (identity when absent).
"""

from __future__ import annotations

import functools
from typing import Optional

import msgspec
import torch

from sglang.kernels.fused_op import BaseFusedOp, get_fused_op_backend
from sglang.kernels.spec import CapabilityRequirement, KernelBackend

_SM90 = frozenset({CapabilityRequirement.cuda(min_sm=(9, 0), max_sm=(9, 0))})

# FP4 E2M1 code -> value; bit 3 is the sign.
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_VALUES = _E2M1_VALUES + tuple(-v for v in _E2M1_VALUES)
# nvfp4_marlin_process_global_scale folds 2**(126 - 7) into the BF16 global
# scale so Marlin can skip the FP4 exponent bias while dequantizing.
_MARLIN_BF16_GLOBAL_SCALE_BIAS = 2.0**-119
_FP4_GROUP_SIZE = 16


class MarlinNvfp4MoeWeights(msgspec.Struct, frozen=True):
    """Local experts' NVFP4 weights in the Marlin layout (see module docstring)."""

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w13_global_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w2_global_scale: torch.Tensor


class PostExpertNorm(msgspec.Struct, frozen=True):
    """RMSNorm applied to each expert output before the top-k weighted sum."""

    weight: torch.Tensor  # [H] bfloat16
    eps: float
    # rms_norm(weight * y) when True, rms_norm(y) * weight otherwise.
    weight_first: bool


@functools.cache
def _marlin_inverse_weight_perm() -> torch.Tensor:
    """Inverse of gptq_marlin_repack's 4-bit permutation of each 1024-code group."""
    perm = []
    for i in range(32):
        col = i // 4
        rows = (2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1)
        tile = [16 * row + col + 8 * block for block in (0, 1) for row in rows]
        for j in range(4):
            perm.extend(p + 256 * j for p in tile)
    perm = torch.tensor(perm).view(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].flatten()
    return torch.argsort(perm)


@functools.cache
def _marlin_inverse_scale_perm() -> torch.Tensor:
    """Inverse of marlin_permute_scales' grouped permutation of each 64 columns."""
    perm = torch.tensor([i + 8 * j for i in range(8) for j in range(8)])
    return torch.argsort(perm)


def decode_marlin_fp4_codes(
    qweight: torch.Tensor, size_k: int, size_n: int
) -> torch.Tensor:
    """Invert gptq_marlin_repack for one expert: ``[K/16, 2N]`` int32 -> ``[N, K]`` codes."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qweight.device)
    codes = (qweight.unsqueeze(-1) >> shifts) & 0xF
    inverse = _marlin_inverse_weight_perm().to(qweight.device)
    codes = codes.reshape(-1, inverse.numel())[:, inverse]
    codes = codes.reshape(size_k // 16, size_n // 16, 16, 16).permute(0, 2, 1, 3)
    return codes.reshape(size_k, size_n).T.to(torch.uint8)


def decode_marlin_fp4_scales(
    scales: torch.Tensor, size_k: int, size_n: int
) -> torch.Tensor:
    """Invert the NVFP4 Marlin scale processing: ``[K/16, N]`` -> ``[N, K/16]`` fp32."""
    # Each byte holds FP16 bits 14..7 of 128 * scale; the dropped bits are zero
    # for non-negative normal E4M3 inputs, so this round trip is exact.
    raw = scales.view(torch.uint8)
    s = (raw.to(torch.int16) << 7).view(torch.float16).float() * 2.0**-7
    # Bytes below 0x80 are subnormal E4M3, which Marlin decodes about 2^-105 too small.
    s = torch.where(raw >= 0x80, s, 0.0)
    s = s.reshape(-1, 4)[:, [0, 2, 1, 3]]
    inverse = _marlin_inverse_scale_perm().to(scales.device)
    s = s.reshape(-1, inverse.numel())[:, inverse]
    return s.reshape(size_k // _FP4_GROUP_SIZE, size_n).T


def dequantize_marlin_nvfp4(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    """One expert's Marlin-layout NVFP4 weight as an exact fp32 ``[N, K]`` tensor."""
    values = torch.tensor(_E2M1_VALUES, device=qweight.device)
    codes = decode_marlin_fp4_codes(qweight, size_k, size_n)
    block_scales = decode_marlin_fp4_scales(scales, size_k, size_n)
    weight = values[codes.long()] * block_scales.repeat_interleave(
        _FP4_GROUP_SIZE, dim=1
    )
    return weight * (global_scale.float() * _MARLIN_BF16_GLOBAL_SCALE_BIAS)


def _marlin_linear(x, qweight, scales, global_scale, size_k, size_n):
    """Marlin rounds the exponent-biased dot to BF16 before its global scale."""
    unit = global_scale.new_tensor(1 / _MARLIN_BF16_GLOBAL_SCALE_BIAS)
    w = dequantize_marlin_nvfp4(qweight, scales, unit, size_k=size_k, size_n=size_n)
    # Removing the bias before this cast changes subnormal BF16 rounding.
    dot = (x.float() @ w.T) * _MARLIN_BF16_GLOBAL_SCALE_BIAS
    return (dot.to(x.dtype).float() * global_scale.float()).to(x.dtype)


def _local_topk_ids(
    topk_ids: torch.Tensor, expert_map: Optional[torch.Tensor], num_experts: int
) -> torch.Tensor:
    """Local expert per slot; -1 (dropped) for padding, remote or out-of-range ids."""
    ids = topk_ids.to(torch.int32)
    if expert_map is not None:
        ids = torch.where(ids >= 0, expert_map[ids.clamp(min=0)], -1)
    return torch.where((ids >= 0) & (ids < num_experts), ids, -1)


def _validate_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weights: MarlinNvfp4MoeWeights,
) -> None:
    if hidden_states.dim() != 2 or hidden_states.dtype != torch.bfloat16:
        raise ValueError(
            "nvfp4_w4a16_moe expects [tokens, hidden] bfloat16 activations, got "
            f"{tuple(hidden_states.shape)} {hidden_states.dtype}."
        )
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != len(hidden_states):
        raise ValueError(
            "nvfp4_w4a16_moe expects matching [tokens, top_k] ids and weights, got "
            f"{tuple(topk_ids.shape)} and {tuple(topk_weights.shape)}."
        )
    hidden_size = hidden_states.shape[1]
    num_experts, w13_k_tiles, w13_cols = weights.w13.shape
    if (
        weights.w13.dtype != torch.int32
        or weights.w2.dtype != torch.int32
        or w13_k_tiles * 16 != hidden_size
        or weights.w2.shape != (num_experts, w13_cols // 4 // 16, 2 * hidden_size)
        or weights.w13_scale.dtype != torch.float8_e4m3fn
        or weights.w2_scale.dtype != torch.float8_e4m3fn
        or weights.w13_global_scale.shape != (num_experts,)
        or weights.w2_global_scale.shape != (num_experts,)
        or weights.w13_global_scale.dtype != torch.bfloat16
        or weights.w2_global_scale.dtype != torch.bfloat16
    ):
        raise ValueError(
            "nvfp4_w4a16_moe expects BF16-activation NVFP4 Marlin weights from "
            "prepare_moe_nvfp4_layer_for_marlin matching the hidden size "
            f"{hidden_size}."
        )


def _post_norm_weighted_sum(
    expert_outputs: torch.Tensor,
    topk_weights: torch.Tensor,
    live: torch.Tensor,
    norm: Optional[PostExpertNorm],
    routed_scaling_factor: float,
) -> torch.Tensor:
    y = torch.where(live.unsqueeze(-1), expert_outputs, 0).float()
    if norm is not None and norm.weight_first:
        y = y * norm.weight.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + norm.eps)
    elif norm is not None:
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + norm.eps)
        y = y * norm.weight.float()
    y = y * topk_weights.float().unsqueeze(-1)
    # Marlin's moe_sum_reduce scales the fp32 sum before the cast.
    return (y.sum(dim=1) * routed_scaling_factor).to(expert_outputs.dtype)


class Nvfp4W4A16MoeOp(BaseFusedOp):
    """Routed NVFP4 W4A16 experts with an optional fused post-expert norm."""

    op = "moe.nvfp4_w4a16_moe"
    priority = (KernelBackend.CUTE_DSL,)
    capabilities = {KernelBackend.CUTE_DSL: _SM90}

    def supports(self, **kwargs) -> bool:
        """Whether an optimized backend takes this call; callers keep Marlin otherwise."""
        # A forced non-candidate backend would dispatch to forward_native, which
        # syncs the host; Marlin is the stock path for that case.
        if get_fused_op_backend() not in (None, *self._auto_backend_candidates()):
            return False
        return any(
            self.backend_eligible(backend, **kwargs)
            for backend in self._auto_backend_candidates()
        )

    def backend_eligible(self, backend: KernelBackend, *args, **kwargs) -> bool:
        if not super().backend_eligible(backend, *args, **kwargs):
            return False
        if backend is KernelBackend.CUTE_DSL:
            from sglang.kernels.ops.moe.kawaii_moe import kawaii_moe_supported

            return kawaii_moe_supported(*args, **kwargs)
        return True

    def forward_cute_dsl(self, *args, **kwargs) -> torch.Tensor:
        # forward(backend=...) skips backend_eligible, so check it here too.
        if not self.backend_eligible(KernelBackend.CUTE_DSL, *args, **kwargs):
            raise ValueError(
                "nvfp4_w4a16_moe: the cute_dsl backend needs SM90 and a shape it "
                "supports (see kawaii_moe_supported); use the default dispatch."
            )
        from sglang.kernels.ops.moe.kawaii_moe import kawaii_moe

        return kawaii_moe(*args, **kwargs)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: MarlinNvfp4MoeWeights,
        expert_map: Optional[torch.Tensor] = None,
        norm: Optional[PostExpertNorm] = None,
        routed_scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        _validate_inputs(hidden_states, topk_weights, topk_ids, weights)
        dtype = hidden_states.dtype
        hidden_size = hidden_states.shape[1]
        intermediate_size = weights.w2.shape[1] * 16
        local_ids = _local_topk_ids(topk_ids, expert_map, weights.w13.shape[0])
        expert_outputs = hidden_states.new_zeros((*local_ids.shape, hidden_size))
        for expert in local_ids.unique().tolist():
            if expert < 0:
                continue
            tokens, slots = (local_ids == expert).nonzero(as_tuple=True)
            gate_up = _marlin_linear(
                hidden_states[tokens],
                weights.w13[expert],
                weights.w13_scale[expert],
                weights.w13_global_scale[expert],
                hidden_size,
                2 * intermediate_size,
            )
            gate, up = gate_up.float().chunk(2, dim=-1)
            act = (torch.nn.functional.silu(gate) * up).to(dtype)
            expert_outputs[tokens, slots] = _marlin_linear(
                act,
                weights.w2[expert],
                weights.w2_scale[expert],
                weights.w2_global_scale[expert],
                intermediate_size,
                hidden_size,
            )
        return _post_norm_weighted_sum(
            expert_outputs,
            topk_weights,
            live=local_ids >= 0,
            norm=norm,
            routed_scaling_factor=routed_scaling_factor,
        )
