"""nvfp4_w4a16 fused MoE path: the Hopper W4A16 op where it applies, else Marlin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.moe import (
    MarlinNvfp4MoeWeights,
    nvfp4_w4a16_moe,
    nvfp4_w4a16_moe_supported,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig, register_fused_func
from sglang.srt.layers.moe.moe_runner.marlin import (
    MarlinMoeQuantInfo,
    fused_experts_none_to_marlin,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, StandardTopKOutputPacked

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


def _op_inputs(
    dispatch_output: StandardDispatchOutput,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> Optional[dict]:
    """The op's inputs when it computes what the Marlin path would, else None."""
    topk_output = dispatch_output.topk_output
    if (
        not isinstance(topk_output, (StandardTopKOutput, StandardTopKOutputPacked))
        # The op always returns the combined [tokens, hidden] sum.
        or runner_config.no_combine
        or not runner_config.is_gated
        or runner_config.activation != "silu"
        or runner_config.gemm1_alpha is not None
        or runner_config.gemm1_clamp_limit is not None
        or runner_config.swiglu_limit is not None
        or quant_info.weight_bits != 4
        or quant_info.w13_scales.dtype != torch.float8_e4m3fn
        or quant_info.w13_global_scale is None
        or quant_info.w2_global_scale is None
        or quant_info.w2_scales.dtype != torch.float8_e4m3fn
        or quant_info.w13_qzeros is not None
        or quant_info.w2_qzeros is not None
        or quant_info.w13_g_idx is not None
        or quant_info.w2_g_idx is not None
        or quant_info.w13_bias is not None
        or quant_info.w2_bias is not None
        or dispatch_output.hidden_states.dtype != torch.bfloat16
    ):
        return None
    return dict(
        hidden_states=dispatch_output.hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        weights=MarlinNvfp4MoeWeights(
            w13=quant_info.w13_qweight,
            w13_scale=quant_info.w13_scales,
            w13_global_scale=quant_info.w13_global_scale,
            w2=quant_info.w2_qweight,
            w2_scale=quant_info.w2_scales,
            w2_global_scale=quant_info.w2_global_scale,
        ),
        # StandardDispatcher already mapped the ids to local experts (-1 = remote).
        expert_map=None,
        routed_scaling_factor=(
            1.0
            if runner_config.routed_scaling_factor is None
            else runner_config.routed_scaling_factor
        ),
    )


@register_fused_func("none", "nvfp4_w4a16")
def fused_experts_none_to_nvfp4_w4a16(
    dispatch_output: StandardDispatchOutput,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    op_inputs = _op_inputs(dispatch_output, quant_info, runner_config)
    if op_inputs is None or not nvfp4_w4a16_moe_supported(**op_inputs):
        return fused_experts_none_to_marlin(dispatch_output, quant_info, runner_config)
    return StandardCombineInput(hidden_states=nvfp4_w4a16_moe(**op_inputs))
