"""Collectively admitted, transient peer weights for wide prefill only."""

import ctypes
import hashlib
import logging
import os

import msgspec
import torch

from .expert_balancing_status import record_admission, tensor_bytes

logger = logging.getLogger(__name__)


def collective_stage(group, action, on_error):
    """No rank enters the next collective after another rank's local failure."""
    value, error = None, None
    try:
        value = action()
    except Exception as exc:
        error = repr(exc)
    errors = group.all_gather_object(error)
    if any(e is not None for e in errors):
        logger.warning("Prefill borrowing unavailable: %s", errors)
        on_error(f"Prefill borrowing: {errors}")
        return None
    return value


def _ipc_export(t, lib):
    """IPC handle of ``t``'s cudaMalloc segment and ``t``'s byte offset in it."""
    from cuda.bindings import driver as cu

    err, base, _ = cu.cuMemGetAddressRange(t.data_ptr())
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"prefill borrowing: CUDA driver error {err}")
    return lib.cudaIpcGetMemHandle(ctypes.c_void_p(int(base))), t.data_ptr() - int(base)


def allocator_supported():
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        for setting in os.environ.get(name, "").split(","):
            key, _, value = setting.partition(":")
            if key.strip() == "expandable_segments" and value.strip().lower() == "true":
                return False
    return True


def _owner_pointers(weights, group):
    """Collectively map canonical fields, or collectively refuse before serving."""
    from cuda.bindings import driver as cu

    from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib = None
    mine, error = None, None
    try:
        if not allocator_supported():
            raise RuntimeError("peer weight IPC does not support expandable segments")
        lib = CudaRTLibrary()
        mine = [
            [_ipc_export(t, lib) for t in msgspec.structs.astuple(w)] for w in weights
        ]
    except Exception as exc:
        error = repr(exc)
    exported = group.all_gather_object((mine, error))
    if any(err is not None for _, err in exported):
        return None
    opened = {}

    def open_handle(handle):
        key = bytes(handle)
        if key not in opened:
            opened[key] = lib.cudaIpcOpenMemHandle(handle).value
        return opened[key]

    result, error = None, None
    try:
        result = [
            [[t.data_ptr() for t in msgspec.structs.astuple(w)] for w in weights]
            if r == group.rank_in_group
            else [[open_handle(h) + offset for h, offset in layer] for layer in fields]
            for r, (fields, _) in enumerate(exported)
        ]
    except Exception as exc:
        error = repr(exc)
    if any(err is not None for err in group.all_gather_object(error)):
        for pointer in opened.values():
            cu.cuIpcCloseMemHandle(pointer)
        return None
    return result


def checksum(fields):
    digest = hashlib.sha256()
    for tensor in fields:
        digest.update(tensor.detach().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


class ExpertBorrow:
    def __init__(self, layers, group):
        from sglang.kernels.ops.moe import MarlinNvfp4MoeWeights
        from sglang.kernels.ops.moe.kawaii_borrow import compiled_copy, compiled_plan

        self.layers, self.group = layers, group
        self.rank, self.ranks = group.rank_in_group, group.world_size
        self.local = layers[0].num_local_experts
        self.weights = [layer._slot_routes()[1] for layer in layers]
        self.fields = [msgspec.structs.astuple(w) for w in self.weights]
        geometry = [(tuple(t.shape), t.dtype) for t in self.fields[0]]
        if any(
            [(tuple(t.shape), t.dtype) for t in row] != geometry for row in self.fields
        ):
            raise ValueError("borrowing requires uniform expert geometry")
        if any(
            not t.is_contiguous() or t.data_ptr() % 16
            for row in self.fields
            for t in row
        ):
            raise ValueError("borrowing requires aligned canonical fields")
        self.device = self.fields[0][0].device
        self.scratch_fields = [torch.empty_like(t[:1]) for t in self.fields[0]]
        self.scratch = MarlinNvfp4MoeWeights(*self.scratch_fields)
        self.sizes = torch.tensor(
            [t[0].numel() * t.element_size() for t in self.fields[0]],
            dtype=torch.int64,
            device=self.device,
        )
        if any(n != 2 and n % 16 for n in self.sizes.cpu().tolist()):
            raise ValueError(
                "borrowing field stride must be 16-byte aligned or a BF16 scalar"
            )
        self.destinations = torch.tensor(
            [t.data_ptr() for t in self.scratch_fields],
            dtype=torch.int64,
            device=self.device,
        )
        self.enabled = torch.zeros(1, dtype=torch.int32, device=self.device)
        # Every expert stays on its home rank: one identity row serves every layer.
        self.owners = torch.arange(
            self.local * self.ranks, dtype=torch.int32, device=self.device
        )
        self.plan_fn = compiled_plan(self.local * self.ranks, self.ranks)
        self.copy_fn = compiled_copy(self.local)
        self.peers = None
        self.reserved_bytes = tensor_bytes(
            [
                *self.scratch_fields,
                self.sizes,
                self.destinations,
                self.enabled,
                self.owners,
            ]
        )

    def supports(self, ids):
        from sglang.kernels.ops.moe.kawaii_borrow import MAX_ENTRIES

        return ids.ndim == 2 and ids.shape[1] == 8 and ids.numel() <= MAX_ENTRIES

    def plan(self, ids):
        import cuda.bindings.driver as cuda
        from cutlass import Int32
        from cutlass.cute.runtime import make_ptr

        from sglang.kernels.ops.moe.kawaii_borrow import BORROW_TILE, borrow_capacity

        cap = max(1, borrow_capacity(ids.numel(), self.ranks))
        main, final = torch.empty_like(ids), torch.empty_like(ids)
        borrowed = torch.empty(cap * BORROW_TILE, dtype=torch.int32, device=ids.device)
        tiles = torch.empty(cap, dtype=torch.int32, device=ids.device)
        count = torch.empty(1, dtype=torch.int32, device=ids.device)
        plan = torch.empty((self.ranks, 3), dtype=torch.int32, device=ids.device)
        tensors = (
            ids,
            self.owners,
            self.enabled,
            main,
            final,
            borrowed,
            tiles,
            count,
            plan,
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.plan_fn(
            *(make_ptr(Int32, t.data_ptr()) for t in tensors),
            Int32(ids.numel()),
            Int32(cap),
            Int32(self.rank),
            stream,
        )
        return main, final, (borrowed, tiles, count), plan

    def copy(self, layer, plan):
        import cuda.bindings.driver as cuda
        from cutlass import Int32, Int64
        from cutlass.cute.runtime import make_ptr

        tensors = (self.peers[layer], self.destinations, self.sizes, plan)
        self.copy_fn(
            *(
                make_ptr(d, t.data_ptr())
                for d, t in zip((Int64, Int64, Int64, Int32), tensors)
            ),
            Int32(self.rank),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )

    def warm_and_test(self):
        from sglang.kernels.ops.moe.kawaii_wide_moe import wide_fc_warm

        if not (
            wide_fc_warm(True, *self.scratch_fields[:3])
            and wide_fc_warm(False, *self.scratch_fields[3:])
        ):
            raise RuntimeError("one-expert wide specialization unavailable")
        # Empty plan warms every node. Test a dynamic peer/slot selected on device.
        _, _, _, plan = self.plan(
            torch.empty((0, 8), dtype=torch.int32, device=self.device)
        )
        peer = (self.rank + 1) % self.ranks
        plan[self.rank] = torch.tensor(
            [0, peer * self.local, 128], dtype=torch.int32, device=self.device
        )
        self.copy(0, plan)
        torch.cuda.synchronize(self.device)
        return checksum(self.scratch_fields), checksum([t[:1] for t in self.fields[0]])


def maybe_create_expert_borrow(*, model, is_draft_worker, group):
    from sglang.srt.environ import envs

    if not envs.SGLANG_ENABLE_MUSE_SPARK_PREFILL_BORROW.get() or is_draft_worker:
        return None
    from .muse_spark_fp4_moe import MuseSpark1xFp4Experts

    def decline(reason):
        return record_admission(model, reason)

    decline("setup not completed")
    layers = [m for m in model.modules() if isinstance(m, MuseSpark1xFp4Experts)]
    eligible = bool(layers) and 2 <= group.world_size <= 8
    eligible = eligible and all(
        m.moe_runner_backend.is_nvfp4_w4a16()
        and m.num_local_experts == 32
        and m.global_num_experts == 32 * group.world_size
        and m.top_k == 8
        and m.hidden_size == 4096
        and m.intermediate_size_per_partition == 4096
        for m in layers
    )
    eligible = (
        eligible
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (9, 0)
    )
    if not all(group.all_gather_object(eligible)):
        logger.warning(
            "Prefill borrowing off: unsupported shape/device/allocator on a peer"
        )
        return decline("unsupported shape/device/allocator on a peer")
    manager = collective_stage(
        group, lambda: ExpertBorrow(layers, group), on_error=decline
    )
    if manager is None:
        return None
    geometry = [[(tuple(t.shape), str(t.dtype)) for t in row] for row in manager.fields]
    if any(other != geometry for other in group.all_gather_object(geometry)):
        logger.warning("Prefill borrowing off: peer weight geometry mismatch")
        return decline("peer weight geometry mismatch")
    peers = _owner_pointers(manager.weights, group)
    if peers is None:
        logger.warning("Prefill borrowing off: peer mapping unavailable")
        return decline("peer IPC mapping unavailable")

    def build_table():
        manager.peers = (
            torch.tensor(peers, dtype=torch.int64, device=manager.device)
            .transpose(0, 1)
            .contiguous()
        )
        manager.reserved_bytes += tensor_bytes([manager.peers])
        return True

    if collective_stage(group, build_table, on_error=decline) is None:
        return None
    hashes = collective_stage(group, manager.warm_and_test, on_error=decline)
    if hashes is None:
        return None
    votes = group.all_gather_object(hashes)
    if not all(
        got == votes[(r + 1) % len(votes)][1] for r, (got, _) in enumerate(votes)
    ):
        logger.warning("Prefill borrowing off: peer test-copy checksum mismatch")
        return decline("peer test-copy checksum mismatch")
    manager.enabled.fill_(1)
    for index, layer in enumerate(layers):
        layer._borrow = (manager, index)
        # Captured graphs keep peer weight pointers: refuse reloads (stock updater check).
        layer._derived_weight_cache_error = "Online weight updates are not supported with Muse prefill borrowing: restart the server."
        layer._weight_update_session_error = layer._derived_weight_cache_error
    record_admission(
        model,
        "setup and peer-copy test passed",
        reserved_bytes=manager.reserved_bytes,
        admitted=True,
    )
    return manager
