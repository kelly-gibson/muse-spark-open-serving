"""Startup admission evidence; no device reads or request-path instrumentation."""

import logging

logger = logging.getLogger(__name__)


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def record_admission(model, reason, *, reserved_bytes=0, admitted=False):
    model._expert_balancing_status = (
        "admitted" if admitted else "declined",
        " ".join(str(reason).split()),
        reserved_bytes,
    )


def report_balancing(model, group):
    """One borrowing summary per EP group, after graph capture and final KV sizing.

    Bytes are persistent local payload retained by borrowing, excluding canonical
    weights, peer mappings and allocator overhead. Admission is not evidence that
    a later request moved expert work.
    """
    from sglang.srt.environ import envs

    if not envs.SGLANG_ENABLE_MUSE_SPARK_PREFILL_BORROW.get() and not hasattr(
        model, "_expert_balancing_status"
    ):
        return
    if group.rank_in_group != 0 or getattr(model, "_expert_balancing_reported", False):
        return
    from .muse_spark_fp4_moe import MuseSpark1xFp4Experts

    # Setup records a status on every model; only Muse models report it.
    if not any(isinstance(m, MuseSpark1xFp4Experts) for m in model.modules()):
        return
    enabled = envs.SGLANG_ENABLE_MUSE_SPARK_PREFILL_BORROW.get()
    state, reason, size = getattr(
        model,
        "_expert_balancing_status",
        (
            "declined",
            "setup not reached" if enabled else "disabled by configuration",
            0,
        ),
    )
    logger.info(
        "Expert balancing EP%d: borrowing=%s reason=%r reserved_bytes=%d",
        group.world_size,
        state,
        reason,
        size,
    )
    model._expert_balancing_reported = True
