"""Hopper BF16 GEMMs on the image's cuBLASLt with fixed, pre-tuned algorithms.

Each config entry names one algorithm by its readable cuBLASLt configuration.
It is rebuilt with ``cublasLtMatmulAlgoInit`` plus the config attributes and
validated with ``cublasLtMatmulAlgoCheck``. A device, library version, shape
or row count without a valid entry returns ``None`` and the caller keeps its
stock GEMM. ``y = x @ weight.T`` runs as the column-major ``y.T = weight @ x.T``.
The table is generated offline, not tuned at server startup. The development
script ``benchmark/kernels/cublaslt_fixed/tune_cublaslt_fixed.py`` searches the
library's readable heuristics and validates accuracy and repeatability before
paired graph timing. The H200 entries were selected in a joint projection/router
race on cuBLASLt 130101; startup only rebuilds and validates those algorithms.
No model weights or outputs are encoded in the table.
"""

from __future__ import annotations

import bisect
import ctypes
import functools
import logging
from pathlib import Path
from typing import Optional

import msgspec
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).with_name("configs") / "cublaslt"
_SM90 = (9, 0)

# Enum values from cublasLt.h / library_types.h (CUDA 13).
_CUDA_R_32F = 0
_CUDA_R_16BF = 14
_COMPUTE_32F = 68
_OP_N, _OP_T = 0, 1
_DESC_TRANSA, _DESC_TRANSB, _DESC_SM_COUNT_TARGET = 3, 4, 15
_DTYPE_IDS = {torch.bfloat16: _CUDA_R_16BF, torch.float32: _CUDA_R_32F}
OUT_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
# Tuning queries heuristics with the default 256-byte operand alignment.
_ALIGN_BYTES = 256
STOCK_ALGO = -1


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _HeuristicResult(ctypes.Structure):
    _fields_ = [
        ("algo", _Algo),
        ("workspace_size", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves_count", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


class AlgoConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    # Serves row counts above the previous entry's m, up to this m.
    m: int
    # STOCK_ALGO marks a bucket where the stock GEMM measured faster.
    algo: int
    tile: int
    splitk: int
    reduction: int
    swizzle: int
    custom: int
    stages: int
    inner: int
    cluster: int
    workspace: int = 0
    # 0 = every SM; a side-stream GEMM may leave SMs to the GEMM it overlaps.
    sm_count: int = 0


class ShapeConfigs(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    n: int
    k: int
    out: str
    entries: list[AlgoConfig]


class DeviceConfigs(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    cublaslt_version: int
    shapes: list[ShapeConfigs]


# (attribute id, AlgoConfig field, C type) for cublasLtMatmulAlgoConfigSetAttribute.
ALGO_CONFIG_FIELDS = (
    (1, "tile", ctypes.c_uint32),
    (2, "splitk", ctypes.c_int32),
    (3, "reduction", ctypes.c_uint32),
    (4, "swizzle", ctypes.c_uint32),
    (5, "custom", ctypes.c_uint32),
    (6, "stages", ctypes.c_uint32),
    (7, "inner", ctypes.c_uint16),
    (8, "cluster", ctypes.c_uint16),
)


class _Plan(msgspec.Struct, frozen=True):
    desc: ctypes.c_void_p
    a_layout: ctypes.c_void_p
    b_layout: ctypes.c_void_p
    d_layout: ctypes.c_void_p
    algo: _Algo
    workspace: int


def check(status: int, what: str) -> None:
    if status != 0:
        raise RuntimeError(f"{what} failed with cuBLAS status {status}")


@functools.cache
def lib() -> Optional[ctypes.CDLL]:
    # torch already loaded this soname, so dlopen returns the image's copy.
    name = f"libcublasLt.so.{torch.version.cuda.split('.')[0]}"
    try:
        cdll = ctypes.CDLL(name)
    except OSError:
        logger.warning("%s not found; keeping stock GEMMs", name)
        return None
    vp, size, i32 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    pvp, palgo = ctypes.POINTER(vp), ctypes.POINTER(_Algo)
    signatures = {
        "cublasLtGetVersion": [],
        "cublasLtCreate": [pvp],
        "cublasLtMatmulDescCreate": [pvp, i32, i32],
        "cublasLtMatmulDescSetAttribute": [vp, i32, vp, size],
        "cublasLtMatrixLayoutCreate": [
            pvp,
            i32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int64,
        ],
        "cublasLtMatmulAlgoInit": [vp, i32, i32, i32, i32, i32, i32, i32, palgo],
        "cublasLtMatmulAlgoConfigSetAttribute": [palgo, i32, vp, size],
        "cublasLtMatmulAlgoCheck": [vp] * 6 + [palgo, ctypes.POINTER(_HeuristicResult)],
        "cublasLtMatmul": [vp] * 12 + [palgo, vp, size, vp],
    }
    for fn_name, argtypes in signatures.items():
        fn = getattr(cdll, fn_name)
        fn.argtypes = argtypes
        fn.restype = i32
    cdll.cublasLtGetVersion.restype = size
    return cdll


@functools.cache
def handle(device_index: int) -> ctypes.c_void_p:
    light_handle = ctypes.c_void_p()
    with torch.cuda.device(device_index):
        check(lib().cublasLtCreate(ctypes.byref(light_handle)), "cublasLtCreate")
    return light_handle


def config_path(device_index: int) -> Path:
    name = torch.cuda.get_device_name(device_index).replace(" ", "_")
    return CONFIG_DIR / f"device_name={name}.json"


@functools.cache
def _device_table(
    device_index: int,
) -> dict[tuple[int, int, torch.dtype], tuple[list[int], list[AlgoConfig]]]:
    if (
        not envs.SGLANG_ENABLE_MUSE_SPARK_CUBLASLT_FIXED_GEMM.get()
        or torch.cuda.get_device_capability(device_index) != _SM90
    ):
        return {}
    path = config_path(device_index)
    if not path.is_file() or lib() is None:
        return {}
    configs = msgspec.json.decode(path.read_bytes(), type=DeviceConfigs)
    version = lib().cublasLtGetVersion()
    if configs.cublaslt_version != version:
        logger.warning(
            "cuBLASLt %d differs from the tuned %d; keeping stock GEMMs",
            version,
            configs.cublaslt_version,
        )
        return {}
    table = {}
    for shape in configs.shapes:
        entries = sorted(shape.entries, key=lambda entry: entry.m)
        table[(shape.n, shape.k, OUT_DTYPES[shape.out])] = (
            [entry.m for entry in entries],
            entries,
        )
    return table


def _entry(
    device_index: int, m: int, n: int, k: int, out_dtype: torch.dtype
) -> Optional[AlgoConfig]:
    found = _device_table(device_index).get((n, k, out_dtype))
    if found is None:
        return None
    bounds, entries = found
    index = bisect.bisect_left(bounds, m)
    if index == len(bounds) or entries[index].algo == STOCK_ALGO:
        return None
    return entries[index]


def has_fixed_algo(
    *, device: torch.device, m: int, n: int, k: int, out_dtype: torch.dtype
) -> bool:
    return (
        device.type == "cuda" and _entry(device.index, m, n, k, out_dtype) is not None
    )


@functools.cache
def _workspace_bytes(device_index: int) -> int:
    entries = [e for _, group in _device_table(device_index).values() for e in group]
    return max((entry.workspace for entry in entries), default=0)


def _set_desc(desc: ctypes.c_void_p, attr: int, value: int) -> None:
    c_value = ctypes.c_int32(value)
    check(
        lib().cublasLtMatmulDescSetAttribute(
            desc, attr, ctypes.byref(c_value), ctypes.sizeof(c_value)
        ),
        "cublasLtMatmulDescSetAttribute",
    )


def _layout(dtype_id: int, rows: int, cols: int, ld: int) -> ctypes.c_void_p:
    layout = ctypes.c_void_p()
    check(
        lib().cublasLtMatrixLayoutCreate(
            ctypes.byref(layout), dtype_id, rows, cols, ld
        ),
        "cublasLtMatrixLayoutCreate",
    )
    return layout


def make_operation(
    *, m: int, n: int, k: int, ldx: int, ldw: int, out_dtype: torch.dtype, sm_count: int
) -> tuple[ctypes.c_void_p, ...]:
    """Matmul descriptor and A/B/D layouts; descriptors live for the process."""
    desc = ctypes.c_void_p()
    check(
        lib().cublasLtMatmulDescCreate(ctypes.byref(desc), _COMPUTE_32F, _CUDA_R_32F),
        "cublasLtMatmulDescCreate",
    )
    _set_desc(desc, _DESC_TRANSA, _OP_T)
    _set_desc(desc, _DESC_TRANSB, _OP_N)
    if sm_count:
        _set_desc(desc, _DESC_SM_COUNT_TARGET, sm_count)
    # Row-major weight [N, K] and x [M, K] are column-major [K, N] and [K, M].
    a_layout = _layout(_CUDA_R_16BF, k, n, ldw)
    b_layout = _layout(_CUDA_R_16BF, k, m, ldx)
    d_layout = _layout(_DTYPE_IDS[out_dtype], n, m, n)
    return desc, a_layout, b_layout, d_layout


def build_algo(
    *, device_index: int, config: AlgoConfig, out_dtype: torch.dtype
) -> _Algo:
    algo = _Algo()
    out_id = _DTYPE_IDS[out_dtype]
    check(
        lib().cublasLtMatmulAlgoInit(
            handle(device_index),
            _COMPUTE_32F,
            _CUDA_R_32F,
            _CUDA_R_16BF,
            _CUDA_R_16BF,
            out_id,
            out_id,
            config.algo,
            ctypes.byref(algo),
        ),
        "cublasLtMatmulAlgoInit",
    )
    fields = msgspec.structs.asdict(config)
    for attr, field, c_type in ALGO_CONFIG_FIELDS:
        value = c_type(fields[field])
        check(
            lib().cublasLtMatmulAlgoConfigSetAttribute(
                ctypes.byref(algo), attr, ctypes.byref(value), ctypes.sizeof(value)
            ),
            "cublasLtMatmulAlgoConfigSetAttribute",
        )
    return algo


def algo_workspace(
    *, device_index: int, operation: tuple[ctypes.c_void_p, ...], algo: _Algo
) -> Optional[int]:
    """Workspace bytes the algorithm needs for this operation, or None if invalid."""
    desc, a_layout, b_layout, d_layout = operation
    result = _HeuristicResult()
    status = lib().cublasLtMatmulAlgoCheck(
        handle(device_index),
        desc,
        a_layout,
        b_layout,
        d_layout,
        d_layout,
        ctypes.byref(algo),
        ctypes.byref(result),
    )
    if status != 0 or result.state != 0:
        return None
    return result.workspace_size


@functools.cache
def _plan(
    device_index: int,
    m: int,
    n: int,
    k: int,
    ldx: int,
    ldw: int,
    out_dtype: torch.dtype,
) -> Optional[_Plan]:
    config = _entry(device_index, m, n, k, out_dtype)
    if config is None:
        return None
    operation = make_operation(
        m=m, n=n, k=k, ldx=ldx, ldw=ldw, out_dtype=out_dtype, sm_count=config.sm_count
    )
    algo = build_algo(device_index=device_index, config=config, out_dtype=out_dtype)
    workspace = algo_workspace(
        device_index=device_index, operation=operation, algo=algo
    )
    if workspace is None or workspace > config.workspace:
        logger.warning(
            "cuBLASLt config for m=%d n=%d k=%d is invalid here; keeping stock GEMM",
            m,
            n,
            k,
        )
        return None
    return _Plan(*operation, algo=algo, workspace=workspace)


# Per (device, stream): concurrent streams must not share split-K partials.
# Sized once to the largest tuned need, so a captured graph's buffer never moves.
_workspaces: dict[tuple[int, int], torch.Tensor] = {}
_warmed: set[tuple] = set()


def _workspace_ptr(device: torch.device, stream: torch.cuda.Stream) -> int:
    key = (device.index, stream.cuda_stream)
    buffer = _workspaces.get(key)
    if buffer is None:
        buffer = torch.empty(
            _workspace_bytes(device.index), dtype=torch.uint8, device=device
        )
        _workspaces[key] = buffer
    return buffer.data_ptr()


_ONE = ctypes.c_float(1.0)
_ZERO = ctypes.c_float(0.0)


def launch(
    *,
    device_index: int,
    operation: tuple[ctypes.c_void_p, ...],
    algo: _Algo,
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    workspace_ptr: int,
    workspace: int,
    stream: int,
) -> None:
    desc, a_layout, b_layout, d_layout = operation
    check(
        lib().cublasLtMatmul(
            handle(device_index),
            desc,
            ctypes.byref(_ONE),
            weight.data_ptr(),
            a_layout,
            x.data_ptr(),
            b_layout,
            ctypes.byref(_ZERO),
            out.data_ptr(),
            d_layout,
            out.data_ptr(),
            d_layout,
            ctypes.byref(algo),
            workspace_ptr or None,
            workspace,
            stream,
        ),
        "cublasLtMatmul",
    )


def _aligned(tensor: torch.Tensor) -> bool:
    return tensor.data_ptr() % _ALIGN_BYTES == 0


def fixed_gemm(
    x: torch.Tensor, weight: torch.Tensor, *, out_dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """``x @ weight.T`` with a tuned cuBLASLt algorithm, or None to keep stock.

    x is BF16 ``[M, K]`` and weight BF16 ``[N, K]``, both with unit inner
    stride; the output is a new contiguous ``[M, N]`` tensor of ``out_dtype``.
    """
    if torch.compiler.is_compiling():
        return None
    if not (
        x.is_cuda
        and x.dim() == 2
        and weight.dim() == 2
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.device == weight.device
        and x.shape[0] > 0
        and x.shape[1] == weight.shape[1]
        and x.stride(1) == 1
        and weight.stride(1) == 1
        and _aligned(x)
        and _aligned(weight)
    ):
        return None
    m, k = x.shape
    n = weight.shape[0]
    # A single row has no meaningful row stride.
    ldx = x.stride(0) if m > 1 else k
    ldw = weight.stride(0) if n > 1 else k
    if ldx < k or ldw < k or ldx % 8 or ldw % 8:
        return None
    device_index = x.device.index
    stream = torch.cuda.current_stream(x.device)
    key = (device_index, stream.cuda_stream, m, n, k, ldx, ldw, out_dtype)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and key not in _warmed:
        return None
    config = _entry(device_index, m, n, k, out_dtype)
    if config is None:
        return None
    plan = _plan(device_index, m, n, k, ldx, ldw, out_dtype)
    if plan is None:
        return None
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    launch(
        device_index=device_index,
        operation=(plan.desc, plan.a_layout, plan.b_layout, plan.d_layout),
        algo=plan.algo,
        x=x,
        weight=weight,
        out=out,
        workspace_ptr=_workspace_ptr(x.device, stream) if plan.workspace else 0,
        workspace=plan.workspace,
        stream=stream.cuda_stream,
    )
    if not capturing:
        _warmed.add(key)
    return out
