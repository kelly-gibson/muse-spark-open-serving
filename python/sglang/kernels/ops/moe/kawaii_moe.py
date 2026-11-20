"""Kawaii-MoE: NVFP4-weight, BF16-activation routed experts for Hopper in CuTe DSL.

The kernel reads the NVFP4 Marlin layout documented in ``nvfp4_w4a16_moe`` in
place and reproduces Marlin's BF16 dequantization bits, so stock Marlin stays a
zero-copy fallback. Weights are the MMA M operand: one Marlin int32 word is one
lane's m16n8k16 A fragment (16 weight rows x 16 K).

One persistent cooperative launch per layer, one CTA per SM:

1. Route: every CTA stably sorts the local ``(token, slot)`` entries by expert
   and cuts them into jobs of at most ``_JOB_TOKENS`` entries, in expert order.
   A job decodes its weights once and feeds one m16n8k16 token tile per
   ``_TILE_TOKENS`` entries.
2. Units: one flat list of ``J * I/64`` FC1 units, then ``J * H/64`` FC2 units,
   dealt round-robin. FC1 unit ``(job, p)`` makes act columns ``64p..64p+63``
   (gate group ``p`` with up group ``p + I/64``); FC2 unit ``(job, g)`` makes y
   columns ``64g..64g+63``. FC2 of a job waits for its FC1 units through a
   per-job counter, which round-robin order leaves about a phase of slack.
3. Combine: each token's local slots are normed and summed in slot order.

Warp 0 fetches each weight record as TMA boxes of k-tile rows, warp 1 streams
the job's activation rows into the same slots, and 16 consumer warps decode and
MMA.
Counters use two parities flipped by an epoch the last CTA bumps, so nothing
is cleared per call.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint16, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu import warp as warp_ops
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.memory import SmemAllocator

# BF16x2 bits of an E2M1 pair: sign at bit 15, exponent and mantissa at bits 8..6.
_FP4_BF16X2_MASK = 0x81C081C0

_CONSUMER_WARPS = 16
_CONSUMERS = 32 * _CONSUMER_WARPS
_THREADS = _CONSUMERS + 64  # warp 0 weights, warp 1 activations
_CONSUMER_BARRIER = 1
_TILE_TOKENS = 8  # one m16n8k16 N tile
# Token tiles per decoded weight fragment. Measured on H200: the consumer is bound by
# decode issue slots, and a second tile adds 16% to a job (a third and fourth 46%, with spills).
_JOB_TILES = 2
_JOB_TOKENS = _JOB_TILES * _TILE_TOKENS
_TILE_BYTES = 512  # Marlin: 16 K x 64 columns of FP4
_SCALE_BYTES = 64  # one block scale per column for those 16 K
_RECORD_TILES = 32
_RECORD_WEIGHT_BYTES = _RECORD_TILES * _TILE_BYTES
_RECORD_BYTES = _RECORD_WEIGHT_BYTES + _RECORD_TILES * _SCALE_BYTES
_FC1_RECORD_K = 256  # 2 column groups x 16 k-tiles
_FC2_RECORD_K = 512  # 1 column group x 32 k-tiles
# Activation chunk rows are padded by 16 B so the 8 token rows of a B fragment
# load land in distinct banks.
_FC1_ROW_BYTES = 2 * _FC1_RECORD_K + 16
_FC2_ROW_BYTES = 2 * _FC2_RECORD_K + 16
_STAGES = 4  # measured on H200: ring depths 4 and 6 stream alike
_MAX_EXPERTS = 64  # local weight slots; route ranks 16 per ballot pass
# Shape envelope the job/entry field widths and smem plan are sized for; ptxas
# compiles H = 8192 spill-free, while H = 65536 spills the combine's norm registers.
_MAX_TOP_K = 16
_MAX_HIDDEN = 8192
_MAX_INTER = 128 * _FC2_RECORD_K
_NO_EXPERT = 255
_NO_ENTRY = 0xFFFF
# Smem tables: per-expert count, first entry, first job; then job count, epoch.
_COUNT, _START, _JOB_BASE = 0, _MAX_EXPERTS, 2 * _MAX_EXPERTS
_N_JOBS, _EPOCH = 3 * _MAX_EXPERTS, 3 * _MAX_EXPERTS + 1
# Largest call the narrow path takes; larger calls go to stock Marlin or the wide kernel.
# Set from H200 measurements at H = I = 4096, top-8; not a hardware limit.
MAX_ROWS = 256
_SMEM_LIMIT = 227 * 1024  # Hardware constraint: sm_90 dynamic shared memory per block.


def fp4_fragment_bits(q):
    """One Marlin FP4 word -> A registers a0..a3 as BF16x2 bits of e2m1 * 2**-126.

    Bit-identical to Marlin's ``dequant<nv_bfloat162, kFE2M1f>``. Plain operators,
    so the same code runs traced in a kernel or on numpy uint32 arrays.
    """
    hi = q >> 8
    # One isolated nibble pair keeps the multiply's two shifted copies disjoint:
    # 4160 = 2**12 + 2**6 and 260 = 2**8 + 2**2 land the sign at 15, the rest at 8..6.
    pair0 = ((q & 0x000F000F) * 4160) & _FP4_BF16X2_MASK
    pair1 = ((q & 0x00F000F0) * 260) & _FP4_BF16X2_MASK
    pair2 = ((hi & 0x000F000F) * 4160) & _FP4_BF16X2_MASK
    pair3 = ((hi & 0x00F000F0) * 260) & _FP4_BF16X2_MASK
    # Pair p holds K (2t, 2t+1) of row r + 8 * (p // 2) at K offset 8 * (p % 2).
    return pair0, pair2, pair1, pair3


def scale_pair_bits(word, shift):
    """Four Marlin scale bytes -> BF16x2 bits of bytes (0, 2) for shift 8, (1, 3) for shift 0.

    Marlin's ``dequant_fp8_scales<nv_bfloat162, kFE4M3fn>``: a byte holds FP16
    bits 14..7 of 128 * s and decodes to 128 * s when that is 0 or at least 2,
    which covers every normal E4M3 scale.
    """
    v = word << shift
    return ((v & 0x80008000) >> 1) | ((v & 0x7F007F00) >> 4)


def fp8_scale_bits(w):
    return scale_pair_bits(w, 8), scale_pair_bits(w, 0)


# Static index maps, in plain operators so they also evaluate on CPU ints.


def fc1_tile_origin(tile, act_group, record, act_groups):
    """FC1 record tile -> (k tile, W13 column group): 16 gate tiles, then 16 up tiles."""
    return 16 * record + tile % 16, act_group + (tile // 16) * act_groups


def fc2_tile_origin(tile, y_group, record):
    """FC2 record tile -> (k tile, W2 column group)."""
    return 32 * record + tile, y_group


def consumer_tile(warp, kk, fc1):
    """Record tile of a consumer warp's kk-th k tile: FC1 splits groups by warp parity."""
    if fc1:
        return (warp % 2) * 16 + 2 * (warp // 2) + kk
    return 2 * warp + kk


def activation_k_tile(tile, fc1):
    """K tile of a record tile inside the record's activation chunk."""
    return tile % 16 if fc1 else tile


def partial_index(warp, lane, j, q, tile=0):
    """Float index of accumulator element q of subblock j of a token tile in the unit-end partials.

    The partials are [token tile][warp][weight row 0..63][token 0..7]; m16n8 C element q
    of a lane is row lane/4 + 8 (q/2) and token 2 (lane%4) + q%2 of its tile.
    """
    row = 16 * j + lane // 4 + 8 * (q // 2)
    token = _TILE_TOKENS * tile + 2 * (lane % 4) + q % 2
    return partial_offset(warp, row, token)


def partial_offset(warp, row, token):
    tile = token // _TILE_TOKENS
    return ((tile * _CONSUMER_WARPS + warp) * 64 + row) * 8 + token % _TILE_TOKENS


def epilogue_coords(thread, tile=0):
    """Consumer thread -> (token, column) it finalizes in a token tile; a warp reads 32 consecutive partials."""
    return _TILE_TOKENS * tile + thread % 8, thread // 8


def pack_job(start, count, expert):
    return start | (count << 16) | (expert << 24)


def unpack_job(word):
    # Uint32 tensor elements read back as Int32, so `>> 24` alone would sign-extend.
    return word & 0xFFFF, (word >> 16) & 0xFF, (word >> 24) & 0xFF


# Inline PTX for operations the DSL does not expose with the needed form.


def _asm(result, args, text, constraints, side_effects, loc, ip):
    return llvm.inline_asm(
        result,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _lds(addr, n, loc, ip):
    regs = ",".join(f"${i}" for i in range(n))
    vec = f".v{n}" if n > 1 else ""
    struct = llvm.StructType.get_literal([T.i32()] * n)
    res = _asm(
        struct if n > 1 else T.i32(),
        [Int32(addr)],
        f"ld.shared{vec}.u32 {{{regs}}}, [${n}];"
        if n > 1
        else "ld.shared.u32 $0, [$1];",
        ",".join(["=r"] * n + ["r"]),
        True,
        loc,
        ip,
    )
    if n == 1:
        return Uint32(res)
    return [
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(n)
    ]


@dsl_user_op
def _lds1(addr, *, loc=None, ip=None):
    return _lds(addr, 1, loc, ip)


@dsl_user_op
def _lds2(addr, *, loc=None, ip=None):
    return _lds(addr, 2, loc, ip)


@dsl_user_op
def _lds4(addr, *, loc=None, ip=None):
    return _lds(addr, 4, loc, ip)


@dsl_user_op
def _ldsf(addr, *, loc=None, ip=None):
    text = "ld.shared.f32 $0, [$1];"
    return Float32(_asm(T.f32(), [Int32(addr)], text, "=f,r", True, loc, ip))


@dsl_user_op
def _sts2f(addr, a, b, *, loc=None, ip=None):
    text = "st.shared.v2.f32 [$0], {$1, $2};"
    _asm(None, [Int32(addr), Float32(a), Float32(b)], text, "r,f,f", True, loc, ip)


@dsl_user_op
def _ldg4(addr, *, loc=None, ip=None):
    """16 B global load through L2 only, for rows other CTAs wrote this launch."""
    struct = llvm.StructType.get_literal([T.i32()] * 4)
    text = "ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4];"
    res = _asm(struct, [Int64(addr)], text, "=r,=r,=r,=r,l", True, loc, ip)
    return [
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def _ldgf(addr, *, loc=None, ip=None):
    text = "ld.global.cg.f32 $0, [$1];"
    return Float32(_asm(T.f32(), [Int64(addr)], text, "=f,l", True, loc, ip))


@dsl_user_op
def _stg4(addr, a, b, c, d, *, loc=None, ip=None):
    text = "st.global.v4.u32 [$0], {$1, $2, $3, $4};"
    args = [Int64(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)]
    _asm(None, args, text, "l,r,r,r,r", True, loc, ip)


@dsl_user_op
def _bulk_load(dst, src, size, bar, *, loc=None, ip=None):
    """cp.async.bulk of ``size`` bytes into shared memory, completing tx on ``bar``."""
    text = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];"
    args = [Int32(dst), Int64(src), Int32(size), Int32(bar)]
    _asm(None, args, text, "r,l,r,r", True, loc, ip)


@cute.jit
def _tma_box(
    atom, tma, row, group, dst, rows: cutlass.Constexpr, cols: cutlass.Constexpr, bar
):
    """TMA box of k-tile rows ``row..row+rows-1``, columns ``group*cols..`` into dst."""
    box = cute.make_layout((rows, cols), stride=(cols, 1))
    src = cute.local_tile(tma, (rows, cols), (row // rows, group))
    s, g = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(cute.make_tensor(dst, box), 0, 2),
        cute.group_modes(src, 0, 2),
    )
    cute.copy(atom, g, s, tma_bar_ptr=bar)


@dsl_user_op
def mul_bf16x2_by_half(a, s, high, *, loc=None, ip=None):
    """BF16x2 ``a`` times one BF16 half of ``s`` in both lanes, as Marlin's ``__hmul2``."""
    half = "h" if high else "l"
    text = (
        "{ .reg .b16 l, h; .reg .b32 t; mov.b32 {l, h}, $2; "
        f"mov.b32 t, {{{half}, {half}}}; mul.rn.bf16x2 $0, $1, t; }}"
    )
    return Uint32(_asm(T.i32(), [Uint32(a), Uint32(s)], text, "=r,r,r", False, loc, ip))


@dsl_user_op
def _bf16_scale(acc, scale, *, loc=None, ip=None):
    """Marlin's epilogue: bf16(acc) * bf16 global scale in BF16 arithmetic."""
    text = (
        "{ .reg .b16 a, s; cvt.rn.bf16.f32 a, $1; cvt.rn.bf16.f32 s, $2; "
        "mul.rn.bf16 a, a, s; cvt.f32.bf16 $0, a; }"
    )
    return Float32(
        _asm(T.f32(), [Float32(acc), Float32(scale)], text, "=f,f,f", False, loc, ip)
    )


@dsl_user_op
def _silu_mul(gate, up, *, loc=None, ip=None):
    """Stock's FP32 SwiGLU: its activation JIT builds ``x / (1 + expf(-x)) * up`` with
    ``--use_fast_math``, i.e. ``__fdividef(x, 1 + __expf(-x)) * up`` with flush-to-zero."""
    text = (
        "{ .reg .f32 e; mul.ftz.f32 e, $1, 0fBFB8AA3B; ex2.approx.ftz.f32 e, e; "
        "add.ftz.f32 e, e, 0f3F800000; div.approx.ftz.f32 e, $1, e; mul.ftz.f32 $0, e, $2; }"
    )
    return Float32(
        _asm(T.f32(), [Float32(gate), Float32(up)], text, "=f,f,f", False, loc, ip)
    )


@dsl_user_op
def _pack_bf16x2(lo, hi, *, loc=None, ip=None):
    text = "cvt.rn.bf16x2.f32 $0, $2, $1;"
    return Uint32(
        _asm(T.i32(), [Float32(lo), Float32(hi)], text, "=r,f,f", False, loc, ip)
    )


@dsl_user_op
def _bf16x2_to_f32(word, high, *, loc=None, ip=None):
    bits = Uint32(word) & 0xFFFF0000 if high else Uint32(word) << 16
    return Float32(_asm(T.f32(), [bits], "mov.b32 $0, $1;", "=f,r", False, loc, ip))


def _load_f32(ptr):
    """Load one element of any float dtype as fp32; arch.load takes no BF16."""
    return cute.make_tensor(ptr, cute.make_layout(1))[0].to(Float32)


def _store_bf16(ptr, value):
    cute.make_tensor(ptr, cute.make_layout(1))[0] = BFloat16(value)


@cute.jit
def _wait_barrier(bar: cute.Pointer, phase: Int32):
    # A spin, not mbarrier_wait: that one suspends through NANOSLEEP on every miss.
    ready = cute.arch.mbarrier_try_wait(bar, phase)
    while ready == 0:
        ready = cute.arch.mbarrier_try_wait(bar, phase)


@cute.jit
def _wait_counter(counter: cute.Pointer, target: Int32):
    seen = cute.arch.load(counter, Int32, sem="acquire", scope="gpu")
    while seen < target:
        seen = cute.arch.load(counter, Int32, sem="acquire", scope="gpu")


@cute.jit
def _warp_exclusive_scan(value: Int32, lane: Int32) -> Int32:
    total = value
    for d in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_up(total, 1 << d)
        if lane >= (1 << d):
            total = total + other
    return total - value


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    # Butterfly order: every lane ends with the same bits.
    for d in cutlass.range_constexpr(5):
        value = value + cute.arch.shuffle_sync_bfly(value, 16 >> d)
    return value


def check_fragment_order(mma):
    """A regs a0..a3 = (row, K) (0,0) (8,0) (0,8) (8,8); B regs b0, b1 = K 0 and 8;
    C elements c0..c3 = (row, token) (0,0) (0,1) (8,0) (8,1). Checked at trace time."""
    thr = mma.get_slice(0)
    a = thr.partition_A(cute.make_identity_tensor((16, 16)))
    b = thr.partition_B(cute.make_identity_tensor((8, 16)))
    c = thr.partition_C(cute.make_identity_tensor((16, 8)))
    got_a = [tuple(int(x) for x in a[2 * u]) for u in range(4)]
    got_b = [tuple(int(x) for x in b[2 * u]) for u in range(2)]
    got_c = [tuple(int(x) for x in c[u]) for u in range(4)]
    assert got_a == [(0, 0), (8, 0), (0, 8), (8, 8)], got_a
    assert got_b == [(0, 0), (0, 8)], got_b
    assert got_c == [(0, 0), (0, 1), (8, 0), (8, 1)], got_c


class KawaiiMoe:
    """One compiled kernel per (shape, routing, norm, topk-weight dtype, SM count)."""

    def __init__(
        self,
        *,
        hidden: int,
        inter: int,
        top_k: int,
        experts: int,
        has_expert_map: bool,
        norm_mode: int,  # 0 none, 1 rms_norm(weight * y), 2 rms_norm(y) * weight
        grid: int,
        routed_scaling_factor: float = 1.0,  # scales the fp32 top-k sum, as moe_sum_reduce
    ):
        self.hidden, self.inter, self.top_k, self.experts = (
            hidden,
            inter,
            top_k,
            experts,
        )
        self.has_expert_map, self.norm_mode, self.grid = has_expert_map, norm_mode, grid
        self.routed_scaling_factor = routed_scaling_factor
        self.max_entries = MAX_ROWS * top_k
        self.max_jobs = self.max_entries // _JOB_TOKENS + experts
        self.fc1_groups = inter // 64
        self.fc2_groups = hidden // 64
        # Combine chunks of 8 columns per consumer thread; the last may be partial.
        self.combine_chunks = (hidden + 8 * _CONSUMERS - 1) // (8 * _CONSUMERS)
        self.fc1_records = hidden // _FC1_RECORD_K
        self.fc2_records = inter // _FC2_RECORD_K
        self.act_slot_bytes = _JOB_TOKENS * _FC2_ROW_BYTES
        self.smem_bytes = (
            _STAGES * (_RECORD_BYTES + self.act_slot_bytes)
            + 4 * _CONSUMERS * 16 * _JOB_TILES  # unit-end partials
            + 4 * _CONSUMER_WARPS * _JOB_TOKENS
            + 5 * self.max_entries  # ids u8, pos u16, order u16
            + 4 * self.max_jobs
            + 4 * (3 * _MAX_EXPERTS + 2)
            + 16 * _STAGES
            + 1024  # alignment slack
        )

    def _weight_tmas(self, w13, s13, w2, s2):
        """TMA atoms over W13, W2 [E * K/16, words] and their scales, one record box each."""
        hidden, inter, experts = self.hidden, self.inter, self.experts
        out = []
        for ptr, rows, cols, box in (
            (w13, hidden // 16, 4 * inter, (_RECORD_TILES // 2, _TILE_BYTES // 4)),
            (s13, hidden // 16, 2 * inter, (_RECORD_TILES // 2, _SCALE_BYTES)),
            (w2, inter // 16, 2 * hidden, (_RECORD_TILES, _TILE_BYTES // 4)),
            (s2, inter // 16, hidden, (_RECORD_TILES, _SCALE_BYTES)),
        ):
            t = cute.make_tensor(
                ptr, cute.make_layout((experts * rows, cols), stride=(cols, 1))
            )
            layout = cute.make_layout(box, stride=(box[1], 1))
            atom, tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), t, layout, box
            )
            out += [atom, tma]
        return out

    def counter_index(self, kind, parity, job):
        """kind 0 = FC1 units done (ready), 1 = FC2 units done; epoch and ticket follow."""
        return (2 * kind + parity) * self.max_jobs + job

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        topk_weights: cute.Pointer,
        topk_ids: cute.Pointer,
        expert_map: cute.Pointer,
        w13: cute.Pointer,
        s13: cute.Pointer,
        g13: cute.Pointer,
        w2: cute.Pointer,
        s2: cute.Pointer,
        g2: cute.Pointer,
        norm_weight: cute.Pointer,
        out: cute.Pointer,
        act: cute.Pointer,
        y: cute.Pointer,
        ss: cute.Pointer,
        counters: cute.Pointer,
        rows: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        tmas = self._weight_tmas(w13, s13, w2, s2)
        self.kernel(
            x,
            topk_weights,
            topk_ids,
            expert_map,
            *tmas[:4],
            g13,
            *tmas[4:],
            g2,
            norm_weight,
            out,
            act,
            y,
            ss,
            counters,
            rows,
            eps,
        ).launch(
            grid=(self.grid, 1, 1),
            block=(_THREADS, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
            cooperative=True,
            use_pdl=True,
            # One CTA per SM by smem; without this ptxas caps registers for two and spills.
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Pointer,
        topk_weights: cute.Pointer,
        topk_ids: cute.Pointer,
        expert_map: cute.Pointer,
        w13_atom: cute.CopyAtom,
        w13: cute.Tensor,
        s13_atom: cute.CopyAtom,
        s13: cute.Tensor,
        g13: cute.Pointer,
        w2_atom: cute.CopyAtom,
        w2: cute.Tensor,
        s2_atom: cute.CopyAtom,
        s2: cute.Tensor,
        g2: cute.Pointer,
        norm_weight: cute.Pointer,
        out: cute.Pointer,
        act: cute.Pointer,
        y: cute.Pointer,
        ss: cute.Pointer,
        counters: cute.Pointer,
        rows: Int32,
        eps: Float32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32

        smem = SmemAllocator()
        ring = smem.allocate_array(
            Uint32, _STAGES * _RECORD_BYTES // 4, byte_alignment=128
        )
        acts = smem.allocate_array(
            Uint32, _STAGES * self.act_slot_bytes // 4, byte_alignment=128
        )
        partials = smem.allocate_array(
            Float32, _CONSUMERS * 16 * _JOB_TILES, byte_alignment=16
        )
        ss_partials = smem.allocate_array(
            Float32, _CONSUMER_WARPS * _JOB_TOKENS, byte_alignment=16
        )
        # CuTe DSL 4.8 sign-extends Uint8/Uint16 smem reads (ld.shared.s8);
        # reads that can see the 255/0xFFFF sentinels mask them back.
        ids = smem.allocate_array(Uint8, self.max_entries, byte_alignment=16)
        pos = smem.allocate_array(Uint16, self.max_entries, byte_alignment=16)
        order = smem.allocate_array(Uint16, self.max_entries, byte_alignment=16)
        jobs = smem.allocate_array(Uint32, self.max_jobs, byte_alignment=16)
        tables = smem.allocate_array(Int32, 3 * _MAX_EXPERTS + 2, byte_alignment=16)
        bars = smem.allocate_array(Int64, 2 * _STAGES, byte_alignment=8)
        full, empty = bars, bars + _STAGES

        tables_t = cute.make_tensor(tables, cute.make_layout(3 * _MAX_EXPERTS + 2))
        if tid == 0:
            for s in cutlass.range_constexpr(_STAGES):
                # Arrivals: the weight producer and the activation loader.
                cute.arch.mbarrier_init(full + s, 2)
                # One arrival per consumer warp, after the warp's reads of the slot.
                cute.arch.mbarrier_init(empty + s, _CONSUMER_WARPS)
            cute.arch.mbarrier_init_fence()
        # The previous launch has finished here: every kernel in between reads
        # its predecessor's output, so each one waits for the one before it.
        cute.arch.griddepcontrol_wait()
        epoch_ptr = counters + self.counter_index(2, 0, 0)
        if tid == 0:
            tables_t[_EPOCH] = cute.arch.load(
                epoch_ptr, Int32, sem="relaxed", scope="gpu"
            )
        cute.arch.sync_threads()
        parity = tables_t[_EPOCH] & 1  # stays right when the int32 epoch wraps
        self._clear_next_counters(cta, tid, parity, counters)

        if warp >= 2:
            self._route(
                tid - 64,
                warp - 2,
                lane,
                rows * self.top_k,
                topk_ids,
                expert_map,
                ids,
                pos,
                order,
                jobs,
                tables_t,
            )
        cute.arch.sync_threads()
        n_jobs = tables_t[_N_JOBS]
        jobs_t = cute.make_tensor(jobs, cute.make_layout(self.max_jobs))

        if warp == 0:
            with cute.arch.elect_one():
                for atom in (w13_atom, s13_atom, w2_atom, s2_atom):
                    cpasync.prefetch_descriptor(atom)
            self._produce(
                cta,
                lane,
                n_jobs,
                jobs_t,
                (w13_atom, w13, s13_atom, s13),
                (w2_atom, w2, s2_atom, s2),
                ring,
                full,
                empty,
            )
        elif warp == 1:
            self._load_activations(
                cta,
                lane,
                n_jobs,
                jobs_t,
                order,
                x,
                act,
                counters,
                parity,
                acts,
                full,
                empty,
            )
        else:
            self._consume(
                cta,
                tid - 64,
                warp - 2,
                lane,
                n_jobs,
                jobs_t,
                g13,
                g2,
                norm_weight,
                act,
                y,
                ss,
                counters,
                parity,
                ring,
                acts,
                partials,
                ss_partials,
                full,
                empty,
            )
        # PDL triggers at exit: after an earlier trigger, a PDL kernel in between that never
        # waits would let the next launch read this launch's epoch.
        if warp >= 2:
            self._combine(
                cta,
                tid - 64,
                lane,
                rows,
                topk_weights,
                norm_weight,
                out,
                y,
                ss,
                counters,
                parity,
                ids,
                pos,
                tables_t,
                eps,
            )
        cute.arch.sync_threads()
        if tid == 0:
            # acq_rel orders every CTA's epoch read before the last CTA's bump.
            ticket = cute.arch.atomic_add(
                counters + self.counter_index(2, 0, 1),
                Int32(1),
                sem="acq_rel",
                scope="gpu",
            )
            if ticket == self.grid - 1:
                cute.arch.store(counters + self.counter_index(2, 0, 1), Int32(0))
                cute.arch.store(epoch_ptr, tables_t[_EPOCH] + 1)

    @cute.jit
    def _clear_next_counters(
        self, cta: Int32, tid: Int32, parity: Int32, counters: cute.Pointer
    ):
        # The other parity was last used by the previous launch, which has finished.
        for i in cutlass.range(
            cta * _THREADS + tid, 2 * self.max_jobs, self.grid * _THREADS
        ):
            kind = i // self.max_jobs
            idx = self.counter_index(kind, 1 - parity, i % self.max_jobs)
            cute.arch.store(counters + idx, Int32(0))

    @cute.jit
    def _route(
        self,
        thread: Int32,
        cw: Int32,
        lane: Int32,
        entries: Int32,
        topk_ids: cute.Pointer,
        expert_map: cute.Pointer,
        ids: cute.Pointer,
        pos: cute.Pointer,
        order: cute.Pointer,
        jobs: cute.Pointer,
        tables: cute.Tensor,
    ):
        """Stable sort of entries t * top_k + s by local expert; jobs of <= _JOB_TOKENS entries."""
        ids_t = cute.make_tensor(ids, cute.make_layout(self.max_entries))
        pos_t = cute.make_tensor(pos, cute.make_layout(self.max_entries))
        order_t = cute.make_tensor(order, cute.make_layout(self.max_entries))
        jobs_t = cute.make_tensor(jobs, cute.make_layout(self.max_jobs))
        for f in cutlass.range(thread, entries, _CONSUMERS):
            local = cute.arch.load(topk_ids + f, Int32)
            if cutlass.const_expr(self.has_expert_map):
                if local >= 0:
                    local = cute.arch.load(expert_map + local, Int32)
            v = Int32(_NO_EXPERT)
            if local >= 0 and local < self.experts:
                v = local
            ids_t[f] = Uint8(v)
        cute.arch.barrier(barrier_id=_CONSUMER_BARRIER, number_of_threads=_CONSUMERS)

        # Consumer warp cw ranks experts cw + 16 p by ballot, in entry order.
        passes = (self.experts + _CONSUMER_WARPS - 1) // _CONSUMER_WARPS
        below = cute.arch.lanemask_lt()
        counts = cute.make_rmem_tensor((passes,), Int32)
        counts.fill(0)
        for base in cutlass.range(0, entries, 32):
            f = base + lane
            v = Int32(_NO_EXPERT)
            if f < entries:
                v = Int32(ids_t[f]) & 0xFF
            for p in cutlass.range_constexpr(passes):
                expert = cw + _CONSUMER_WARPS * p
                mask = cute.arch.vote_ballot_sync(v == expert)
                if v == expert:
                    pos_t[f] = Uint16(counts[p] + cute.arch.popc(mask & below))
                counts[p] = counts[p] + cute.arch.popc(mask)
        if lane == 0:
            for p in cutlass.range_constexpr(passes):
                tables[_COUNT + _CONSUMER_WARPS * p + cw] = counts[p]
        cute.arch.barrier(barrier_id=_CONSUMER_BARRIER, number_of_threads=_CONSUMERS)

        if cw == 0:
            # Experts 32 g + lane; each pass carries the running entry and job totals.
            first_entry = Int32(0)
            first_job = Int32(0)
            for g in cutlass.range_constexpr((self.experts + 31) // 32):
                expert = 32 * g + lane
                count = Int32(0)
                if expert < self.experts:
                    count = tables[_COUNT + expert]
                n_jobs = (count + _JOB_TOKENS - 1) // _JOB_TOKENS
                start = first_entry + _warp_exclusive_scan(count, lane)
                job_base = first_job + _warp_exclusive_scan(n_jobs, lane)
                if expert < _MAX_EXPERTS:
                    tables[_START + expert] = start
                    tables[_JOB_BASE + expert] = job_base
                for k in cutlass.range(0, n_jobs, 1):
                    size = cutlass.min(count - _JOB_TOKENS * k, _JOB_TOKENS)
                    jobs_t[job_base + k] = Uint32(
                        pack_job(start + _JOB_TOKENS * k, size, expert)
                    )
                first_entry = cute.arch.shuffle_sync(start + count, 31)
                first_job = cute.arch.shuffle_sync(job_base + n_jobs, 31)
            if lane == 0:
                tables[_N_JOBS] = first_job
        cute.arch.barrier(barrier_id=_CONSUMER_BARRIER, number_of_threads=_CONSUMERS)

        for f in cutlass.range(thread, entries, _CONSUMERS):
            v = Int32(ids_t[f]) & 0xFF
            p = Int32(_NO_ENTRY)
            if v != _NO_EXPERT:
                p = tables[_START + v] + Int32(pos_t[f])
                order_t[p] = Uint16(f)
            pos_t[f] = Uint16(p)

    @cute.jit
    def _produce(
        self,
        cta: Int32,
        lane: Int32,
        n_jobs: Int32,
        jobs: cute.Tensor,
        fc1_tmas: tuple,
        fc2_tmas: tuple,
        ring: cute.Pointer,
        full: cute.Pointer,
        empty: cute.Pointer,
    ):
        """Warp 0: each record's weight tiles and scales as TMA boxes."""
        fc1_units = self.fc1_groups * n_jobs
        record = Int32(0)
        for unit in cutlass.range(
            cta, (self.fc1_groups + self.fc2_groups) * n_jobs, self.grid, unroll=1
        ):
            if unit < fc1_units:
                job = unit // self.fc1_groups
                rows = Int32(unpack_job(jobs[job])[2]) * (self.hidden // 16)
                for r in cutlass.range(0, self.fc1_records, 1, unroll=1):
                    self._issue_record(
                        record,
                        lane,
                        True,
                        rows,
                        unit % self.fc1_groups,
                        r,
                        fc1_tmas,
                        ring,
                        full,
                        empty,
                    )
                    record = record + 1
            else:
                v = unit - fc1_units
                job = v // self.fc2_groups
                rows = Int32(unpack_job(jobs[job])[2]) * (self.inter // 16)
                for r in cutlass.range(0, self.fc2_records, 1, unroll=1):
                    self._issue_record(
                        record,
                        lane,
                        False,
                        rows,
                        v % self.fc2_groups,
                        r,
                        fc2_tmas,
                        ring,
                        full,
                        empty,
                    )
                    record = record + 1

    @cute.jit
    def _issue_record(
        self,
        record: Int32,
        lane: Int32,
        fc1: cutlass.Constexpr,
        expert_row: Int32,
        unit_group: Int32,
        r: Int32,
        tmas: tuple,
        ring: cute.Pointer,
        full: cute.Pointer,
        empty: cute.Pointer,
    ):
        """FC1: a gate box then an up box of 16 k tiles; FC2: one box of 32."""
        w_atom, w_tma, s_atom, s_tma = tmas
        slot = record % _STAGES
        _wait_barrier(empty + slot, ((record // _STAGES) % 2) ^ 1)
        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(full + slot, _RECORD_BYTES)
        cute.arch.sync_warp()
        base = ring + slot * (_RECORD_BYTES // 4)
        tiles = _RECORD_TILES // 2 if fc1 else _RECORD_TILES
        for b in cutlass.range_constexpr(_RECORD_TILES // tiles):
            k_tile, group = (
                fc1_tile_origin(tiles * b, unit_group, r, self.fc1_groups)
                if fc1
                else fc2_tile_origin(tiles * b, unit_group, r)
            )
            row = expert_row + k_tile
            w_dst = cute.recast_ptr(base + tiles * b * _TILE_BYTES // 4, dtype=Int32)
            s_off = (_RECORD_WEIGHT_BYTES + tiles * b * _SCALE_BYTES) // 4
            s_dst = cute.recast_ptr(base + s_off, dtype=Uint8)
            _tma_box(
                w_atom, w_tma, row, group, w_dst, tiles, _TILE_BYTES // 4, full + slot
            )
            _tma_box(s_atom, s_tma, row, group, s_dst, tiles, _SCALE_BYTES, full + slot)

    @cute.jit
    def _load_activations(
        self,
        cta: Int32,
        lane: Int32,
        n_jobs: Int32,
        jobs: cute.Tensor,
        order: cute.Pointer,
        x: cute.Pointer,
        act: cute.Pointer,
        counters: cute.Pointer,
        parity: Int32,
        acts: cute.Pointer,
        full: cute.Pointer,
        empty: cute.Pointer,
    ):
        """Warp 1: lane n streams token n's K chunk of each record; FC2 waits for its act."""
        hidden, inter = self.hidden, self.inter
        order_t = cute.make_tensor(order, cute.make_layout(self.max_entries))
        fc1_units = self.fc1_groups * n_jobs
        record = Int32(0)
        for unit in cutlass.range(
            cta, (self.fc1_groups + self.fc2_groups) * n_jobs, self.grid, unroll=1
        ):
            if unit < fc1_units:
                start, size, _ = unpack_job(jobs[unit // self.fc1_groups])
                token = Int64(0)
                if lane < size:
                    token = Int64(Int32(order_t[Int32(start) + lane]) // self.top_k)
                for r in cutlass.range(0, self.fc1_records, 1, unroll=1):
                    src = (x + (token * hidden + r * _FC1_RECORD_K)).toint()
                    self._issue_chunk(
                        record,
                        lane,
                        Int32(size),
                        src,
                        2 * _FC1_RECORD_K,
                        _FC1_ROW_BYTES,
                        acts,
                        full,
                        empty,
                    )
                    record = record + 1
            else:
                job = (unit - fc1_units) // self.fc2_groups
                start, size, _ = unpack_job(jobs[job])
                _wait_counter(
                    counters + self.counter_index(0, parity, job), self.fc1_groups
                )
                # act rows were written through the generic proxy by other CTAs.
                cute.arch.fence_proxy("async.global")
                entry = Int64(Int32(start) + lane)
                for r in cutlass.range(0, self.fc2_records, 1, unroll=1):
                    src = (act + (entry * inter + r * _FC2_RECORD_K)).toint()
                    self._issue_chunk(
                        record,
                        lane,
                        Int32(size),
                        src,
                        2 * _FC2_RECORD_K,
                        _FC2_ROW_BYTES,
                        acts,
                        full,
                        empty,
                    )
                    record = record + 1

    @cute.jit
    def _issue_chunk(
        self,
        record: Int32,
        lane: Int32,
        size: Int32,
        src: Int64,
        row_bytes: cutlass.Constexpr,
        row_stride: cutlass.Constexpr,
        acts: cute.Pointer,
        full: cute.Pointer,
        empty: cute.Pointer,
    ):
        slot = record % _STAGES
        _wait_barrier(empty + slot, ((record // _STAGES) % 2) ^ 1)
        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(full + slot, size * row_bytes)
        cute.arch.sync_warp()
        if lane < size:
            dst = acts.toint() + slot * self.act_slot_bytes + lane * row_stride
            _bulk_load(dst, src, row_bytes, (full + slot).toint())

    @cute.jit
    def _consume(
        self,
        cta: Int32,
        thread: Int32,
        cw: Int32,
        lane: Int32,
        n_jobs: Int32,
        jobs: cute.Tensor,
        g13: cute.Pointer,
        g2: cute.Pointer,
        norm_weight: cute.Pointer,
        act: cute.Pointer,
        y: cute.Pointer,
        ss: cute.Pointer,
        counters: cute.Pointer,
        parity: Int32,
        ring: cute.Pointer,
        acts: cute.Pointer,
        partials: cute.Pointer,
        ss_partials: cute.Pointer,
        full: cute.Pointer,
        empty: cute.Pointer,
    ):
        mma = cute.make_tiled_mma(
            warp_ops.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (1, 1, 1)
        )
        check_fragment_order(mma)
        thr = mma.get_slice(lane)
        fa = cute.make_rmem_tensor(thr.partition_shape_A((16, 16)), BFloat16)
        frags = (cute.recast_tensor(fa, Uint32), fa, mma, thr)

        fc1_units = self.fc1_groups * n_jobs
        record = Int32(0)
        for unit in cutlass.range(
            cta, (self.fc1_groups + self.fc2_groups) * n_jobs, self.grid, unroll=1
        ):
            if unit < fc1_units:
                job = unit // self.fc1_groups
                # Loaded before the records so the epilogue does not wait on L2 after its barrier.
                scale = _load_f32(g13 + Int32(unpack_job(jobs[job])[2]))
                self._mma_unit(
                    record,
                    Int32(unpack_job(jobs[job])[1]),
                    cw,
                    lane,
                    True,
                    ring,
                    acts,
                    partials,
                    full,
                    empty,
                    frags,
                )
                record = record + self.fc1_records
                self._fc1_epilogue(
                    thread,
                    job,
                    unit % self.fc1_groups,
                    jobs,
                    scale,
                    act,
                    counters,
                    parity,
                    partials,
                )
            else:
                v = unit - fc1_units
                job = v // self.fc2_groups
                scale = _load_f32(g2 + Int32(unpack_job(jobs[job])[2]))
                norm_scale = Float32(1.0)
                if cutlass.const_expr(self.norm_mode == 1):
                    column = (v % self.fc2_groups) * 64 + epilogue_coords(thread)[1]
                    norm_scale = _load_f32(norm_weight + column)
                self._mma_unit(
                    record,
                    Int32(unpack_job(jobs[job])[1]),
                    cw,
                    lane,
                    False,
                    ring,
                    acts,
                    partials,
                    full,
                    empty,
                    frags,
                )
                record = record + self.fc2_records
                self._fc2_epilogue(
                    thread,
                    lane,
                    cw,
                    job,
                    v % self.fc2_groups,
                    jobs,
                    scale,
                    norm_scale,
                    y,
                    ss,
                    counters,
                    parity,
                    partials,
                    ss_partials,
                )

    @cute.jit
    def _mma_unit(
        self,
        record: Int32,
        size: Int32,
        cw: Int32,
        lane: Int32,
        fc1: cutlass.Constexpr,
        ring: cute.Pointer,
        acts: cute.Pointer,
        partials: cute.Pointer,
        full: cute.Pointer,
        empty: cute.Pointer,
        frags: tuple,
    ):
        """A unit's records and partials, with one token tile per _TILE_TOKENS job entries."""
        # The job is the same on every thread of the CTA, so this branch never splits a warp.
        if size > _TILE_TOKENS:
            self._mma_tiles(
                record,
                cw,
                lane,
                fc1,
                _JOB_TILES,
                ring,
                acts,
                partials,
                full,
                empty,
                frags,
            )
        else:
            self._mma_tiles(
                record, cw, lane, fc1, 1, ring, acts, partials, full, empty, frags
            )

    @cute.jit
    def _mma_tiles(
        self,
        record: Int32,
        cw: Int32,
        lane: Int32,
        fc1: cutlass.Constexpr,
        tiles: cutlass.Constexpr,
        ring: cute.Pointer,
        acts: cute.Pointer,
        partials: cute.Pointer,
        full: cute.Pointer,
        empty: cute.Pointer,
        frags: tuple,
    ):
        fa32, fa, mma, thr = frags
        fbs = [
            cute.make_rmem_tensor(thr.partition_shape_B((8, 16)), BFloat16)
            for _ in range(tiles)
        ]
        # accs[4 * tile + j]: subblock j of a token tile.
        accs = [
            cute.make_rmem_tensor(thr.partition_shape_C((16, 8)), Float32)
            for _ in range(4 * tiles)
        ]
        for acc in accs:
            acc.fill(0.0)
        tile_frags = (
            fa32,
            [cute.recast_tensor(fb, Uint32) for fb in fbs],
            fa,
            fbs,
            mma,
        )
        rec = record
        for r in cutlass.range(
            0, self.fc1_records if fc1 else self.fc2_records, 1, unroll=1
        ):
            self._mma_record(
                rec, cw, lane, fc1, tiles, ring, acts, full, empty, accs, tile_frags
            )
            rec = rec + 1
        self._write_partials(cw, lane, tiles, accs, partials)

    @cute.jit
    def _mma_record(
        self,
        record: Int32,
        cw: Int32,
        lane: Int32,
        fc1: cutlass.Constexpr,
        tiles: cutlass.Constexpr,
        ring: cute.Pointer,
        acts: cute.Pointer,
        full: cute.Pointer,
        empty: cute.Pointer,
        accs: list,
        frags: tuple,
    ):
        """Two k tiles x four 16-row subblocks of one record: 8 decodes and 8 m16n8k16 per token tile."""
        fa32, fbs32, fa, fbs, mma = frags
        slot = record % _STAGES
        _wait_barrier(full + slot, (record // _STAGES) % 2)
        base = ring.toint() + slot * _RECORD_BYTES
        row_stride = _FC1_ROW_BYTES if fc1 else _FC2_ROW_BYTES
        x_row = (
            acts.toint()
            + slot * self.act_slot_bytes
            + (lane // 4) * row_stride
            + 4 * (lane % 4)
        )
        for kk in cutlass.range_constexpr(2):
            tile = consumer_tile(cw, kk, fc1)
            words = _lds4(base + tile * _TILE_BYTES + lane * 16)
            scale_words = _lds2(
                base + _RECORD_WEIGHT_BYTES + tile * _SCALE_BYTES + (lane // 4) * 8
            )
            x_k = x_row + activation_k_tile(tile, fc1) * 32
            for n in cutlass.range_constexpr(tiles):
                fbs32[n][0] = _lds1(x_k + n * _TILE_TOKENS * row_stride)
                fbs32[n][1] = _lds1(x_k + n * _TILE_TOKENS * row_stride + 16)
            scales = fp8_scale_bits(scale_words[0]) + fp8_scale_bits(scale_words[1])
            for j in cutlass.range_constexpr(4):
                a = fp4_fragment_bits(words[j])
                for u in cutlass.range_constexpr(4):
                    fa32[u] = mul_bf16x2_by_half(a[u], scales[j], u % 2 == 1)
                for n in cutlass.range_constexpr(tiles):
                    cute.gemm(mma, accs[4 * n + j], fa, fbs[n], accs[4 * n + j])
        cute.arch.sync_warp()
        if lane == 0:
            cute.arch.mbarrier_arrive(empty + slot)

    @cute.jit
    def _write_partials(
        self,
        cw: Int32,
        lane: Int32,
        tiles: cutlass.Constexpr,
        accs: list,
        partials: cute.Pointer,
    ):
        base = partials.toint()
        for n in cutlass.range_constexpr(tiles):
            for j in cutlass.range_constexpr(4):
                for h in cutlass.range_constexpr(2):
                    idx = partial_index(cw, lane, j, 2 * h, n)
                    acc = accs[4 * n + j]
                    _sts2f(base + 4 * idx, acc[2 * h], acc[2 * h + 1])
        cute.arch.barrier(barrier_id=_CONSUMER_BARRIER, number_of_threads=_CONSUMERS)

    @cute.jit
    def _fc1_epilogue(
        self,
        thread: Int32,
        job: Int32,
        act_group: Int32,
        jobs: cute.Tensor,
        scale: Float32,
        act: cute.Pointer,
        counters: cute.Pointer,
        parity: Int32,
        partials: cute.Pointer,
    ):
        """Gate and up sums in warp order, Marlin's BF16 epilogue, SwiGLU, act store."""
        start, size, _ = unpack_job(jobs[job])
        for n in cutlass.range_constexpr(_JOB_TILES):
            # Uniform across the CTA: every thread sees the same job.
            if Int32(size) > _TILE_TOKENS * n:
                self._fc1_tile(
                    thread,
                    n,
                    Int32(start),
                    Int32(size),
                    act_group,
                    scale,
                    act,
                    partials,
                )
        cute.arch.barrier(barrier_id=_CONSUMER_BARRIER, number_of_threads=_CONSUMERS)
        if thread == 0:
            ready = counters + self.counter_index(0, parity, job)
            cute.arch.red(
                ready, Int32(1), op="add", dtype=Int32, sem="release", scope="gpu"
            )

    @cute.jit
    def _fc1_tile(
        self,
        thread: Int32,
        tile: cutlass.Constexpr,
        start: Int32,
        size: Int32,
        act_group: Int32,
        scale: Float32,
        act: cute.Pointer,
        partials: cute.Pointer,
    ):
        token, col = epilogue_coords(thread, tile)
        base = partials.toint()
        gate = Float32(0.0)
        up = Float32(0.0)
        for w in cutlass.range_constexpr(0, _CONSUMER_WARPS, 2):
            gate = gate + _ldsf(base + 4 * partial_offset(w, col, token))
            up = up + _ldsf(base + 4 * partial_offset(w + 1, col, token))
        value = _silu_mul(_bf16_scale(gate, scale), _bf16_scale(up, scale))
        if token < size:
            entry = Int64(start + token)
            dst = act + (entry * self.inter + act_group * 64 + col)
            _store_bf16(dst, value)

    @cute.jit
    def _fc2_epilogue(
        self,
        thread: Int32,
        lane: Int32,
        cw: Int32,
        job: Int32,
        y_group: Int32,
        jobs: cute.Tensor,
        scale: Float32,
        norm_scale: Float32,
        y: cute.Pointer,
        ss: cute.Pointer,
        counters: cute.Pointer,
        parity: Int32,
        partials: cute.Pointer,
        ss_partials: cute.Pointer,
    ):
        """y in BF16, plus per-entry sums of squares for the norm in a fixed order."""
        start, size, _ = unpack_job(jobs[job])
        ssp = cute.make_tensor(
            ss_partials, cute.make_layout(_CONSUMER_WARPS * _JOB_TOKENS)
        )
        for n in cutlass.range_constexpr(_JOB_TILES):
            # Uniform across the CTA, so the shuffles below never run in a split warp.
            if Int32(size) > _TILE_TOKENS * n:
                self._fc2_tile(
                    thread,
                    lane,
                    cw,
                    n,
                    Int32(start),
                    Int32(size),
                    y_group,
                    scale,
                    norm_scale,
                    y,
                    partials,
                    ssp,
                )
        cute.arch.barrier(barrier_id=_CONSUMER_BARRIER, number_of_threads=_CONSUMERS)
        if cutlass.const_expr(self.norm_mode != 0):
            if thread < Int32(size):
                total = Float32(0.0)
                for w in cutlass.range_constexpr(_CONSUMER_WARPS):
                    total = total + ssp[w * _JOB_TOKENS + thread]
                cute.arch.store(
                    ss + (Int64(Int32(start) + thread) * self.fc2_groups + y_group),
                    total,
                )
            cute.arch.sync_warp()
        if thread == 0:
            done = counters + self.counter_index(1, parity, job)
            cute.arch.red(
                done, Int32(1), op="add", dtype=Int32, sem="release", scope="gpu"
            )

    @cute.jit
    def _fc2_tile(
        self,
        thread: Int32,
        lane: Int32,
        cw: Int32,
        tile: cutlass.Constexpr,
        start: Int32,
        size: Int32,
        y_group: Int32,
        scale: Float32,
        norm_scale: Float32,
        y: cute.Pointer,
        partials: cute.Pointer,
        ssp: cute.Tensor,
    ):
        token, col = epilogue_coords(thread, tile)
        base = partials.toint()
        acc = Float32(0.0)
        for w in cutlass.range_constexpr(_CONSUMER_WARPS):
            acc = acc + _ldsf(base + 4 * partial_offset(w, col, token))
        value = _bf16_scale(acc, scale)
        column = y_group * 64 + col
        entry = Int64(start + token)
        if token < size:
            _store_bf16(y + (entry * self.hidden + column), value)
        if cutlass.const_expr(self.norm_mode != 0):
            if cutlass.const_expr(self.norm_mode == 1):
                value = value * norm_scale
            sq = value * value
            # Lanes l, l^8, l^16, l^24 share token l % 8 of the tile.
            sq = sq + cute.arch.shuffle_sync_bfly(sq, 8)
            sq = sq + cute.arch.shuffle_sync_bfly(sq, 16)
            if lane < _TILE_TOKENS:
                ssp[cw * _JOB_TOKENS + token] = sq

    @cute.jit
    def _combine(
        self,
        cta: Int32,
        thread: Int32,
        lane: Int32,
        rows: Int32,
        topk_weights: cute.Pointer,
        norm_weight: cute.Pointer,
        out: cute.Pointer,
        y: cute.Pointer,
        ss: cute.Pointer,
        counters: cute.Pointer,
        parity: Int32,
        ids: cute.Pointer,
        pos: cute.Pointer,
        tables: cute.Tensor,
        eps: Float32,
    ):
        """out[t] = scale * sum in slot order of topk_weight * norm(y); 0 without local slots."""
        hidden = self.hidden
        ids_t = cute.make_tensor(ids, cute.make_layout(self.max_entries))
        pos_t = cute.make_tensor(pos, cute.make_layout(self.max_entries))
        chunks = self.combine_chunks
        norm = cute.make_rmem_tensor((8 * chunks,), Float32)
        if cutlass.const_expr(self.norm_mode != 0):
            for c in cutlass.range_constexpr(chunks):
                words = _ldg4((norm_weight + self._combine_column(c, thread)).toint())
                for i in cutlass.range_constexpr(8):
                    norm[8 * c + i] = _bf16x2_to_f32(words[i // 2], i % 2 == 1)
        acc = cute.make_rmem_tensor((8 * chunks,), Float32)
        for t in cutlass.range(cta, rows, self.grid, unroll=1):
            acc.fill(0.0)
            for s in cutlass.range_constexpr(self.top_k):
                self._combine_slot(
                    t * self.top_k + s,
                    thread,
                    lane,
                    topk_weights,
                    y,
                    ss,
                    counters,
                    parity,
                    ids_t,
                    pos_t,
                    tables,
                    eps,
                    norm,
                    acc,
                )
            if cutlass.const_expr(self.routed_scaling_factor != 1.0):
                for i in cutlass.range_constexpr(8 * chunks):
                    acc[i] = acc[i] * Float32(self.routed_scaling_factor)
            for c in cutlass.range_constexpr(chunks):
                packed = [
                    _pack_bf16x2(acc[8 * c + 2 * i], acc[8 * c + 2 * i + 1])
                    for i in range(4)
                ]
                col = c * 8 * _CONSUMERS + 8 * thread
                offset = Int64(t) * hidden + col
                if cutlass.const_expr((c + 1) * 8 * _CONSUMERS <= hidden):
                    _stg4((out + offset).toint(), *packed)
                elif col < hidden:
                    _stg4((out + offset).toint(), *packed)

    @cute.jit
    def _combine_column(self, c, thread: Int32) -> Int32:
        """First of the 8 columns thread combines in chunk c of 8 * _CONSUMERS columns.
        Past H (the last chunk when H % 4096 != 0) it repeats H - 8, and the store skips it."""
        col = c * 8 * _CONSUMERS + 8 * thread
        if cutlass.const_expr((c + 1) * 8 * _CONSUMERS > self.hidden):
            col = cutlass.min(col, self.hidden - 8)
        return col

    @cute.jit
    def _combine_slot(
        self,
        f: Int32,
        thread: Int32,
        lane: Int32,
        topk_weights: cute.Pointer,
        y: cute.Pointer,
        ss: cute.Pointer,
        counters: cute.Pointer,
        parity: Int32,
        ids: cute.Tensor,
        pos: cute.Tensor,
        tables: cute.Tensor,
        eps: Float32,
        norm: cute.Tensor,
        acc: cute.Tensor,
    ):
        """acc += topk_weight * norm(y) of entry f when its expert is local."""
        hidden = self.hidden
        entry = Int32(pos[f]) & 0xFFFF
        if entry != _NO_ENTRY:
            expert = Int32(ids[f])
            job = (
                tables[_JOB_BASE + expert]
                + (entry - tables[_START + expert]) // _JOB_TOKENS
            )
            _wait_counter(
                counters + self.counter_index(1, parity, job), self.fc2_groups
            )
            rms = Float32(1.0)
            if cutlass.const_expr(self.norm_mode != 0):
                total = Float32(0.0)
                for q in cutlass.range_constexpr(0, self.fc2_groups, 32):
                    src = (ss + (Int64(entry) * self.fc2_groups + q + lane)).toint()
                    if cutlass.const_expr(q + 32 <= self.fc2_groups):
                        total = total + _ldgf(src)
                    elif q + lane < self.fc2_groups:
                        total = total + _ldgf(src)
                rms = cute.math.rsqrt(_warp_sum(total) * (1.0 / hidden) + eps)
            weight = _load_f32(topk_weights + f)
            for c in cutlass.range_constexpr(self.combine_chunks):
                offset = Int64(entry) * hidden + self._combine_column(c, thread)
                words = _ldg4((y + offset).toint())
                for i in cutlass.range_constexpr(8):
                    v = _bf16x2_to_f32(words[i // 2], i % 2 == 1)
                    if cutlass.const_expr(self.norm_mode == 1):
                        v = v * norm[8 * c + i] * rms
                    elif cutlass.const_expr(self.norm_mode == 2):
                        v = v * rms * norm[8 * c + i]
                    acc[8 * c + i] = acc[8 * c + i] + v * weight


# Host side.

_COMPILED = {}
_WORKSPACES = {}
_CUTE_DTYPES = {torch.bfloat16: BFloat16, torch.float32: Float32}


def _weight_tensors(weights):
    return (
        weights.w13,
        weights.w13_scale,
        weights.w13_global_scale,
        weights.w2,
        weights.w2_scale,
        weights.w2_global_scale,
    )


def _pointer_dtypes(topk_dtype):
    """Element types of KawaiiMoe.__call__'s pointers, in argument order."""
    return [
        BFloat16,
        _CUTE_DTYPES[topk_dtype],
        Int32,
        Int32,
        Int32,
        Uint8,
        BFloat16,
        Int32,
        Uint8,
        BFloat16,
        BFloat16,
        BFloat16,
        BFloat16,
        BFloat16,
        Float32,
        Int32,
    ]


def _shape_supported(
    hidden_states,
    topk_weights,
    topk_ids,
    weights,
    expert_map,
    norm,
):
    if hidden_states.dim() != 2 or weights.w13.dim() != 3 or topk_ids.dim() != 2:
        return False
    rows, hidden = hidden_states.shape
    experts, k_tiles, w13_words = weights.w13.shape
    inter = w13_words // 4
    top_k = topk_ids.shape[1]
    # H % 256: FC1 streams K in records of _FC1_RECORD_K.
    return (
        1 <= rows <= MAX_ROWS
        and hidden % _FC1_RECORD_K == 0
        and _FC1_RECORD_K <= hidden <= _MAX_HIDDEN
        and k_tiles * 16 == hidden
        and inter % _FC2_RECORD_K == 0
        and _FC2_RECORD_K <= inter <= _MAX_INTER
        and w13_words == 4 * inter
        and 1 <= experts <= _MAX_EXPERTS
        and weights.w13_scale.shape == (experts, hidden // 16, 2 * inter)
        and weights.w2.shape == (experts, inter // 16, 2 * hidden)
        and weights.w2_scale.shape == (experts, inter // 16, hidden)
        and weights.w13_global_scale.shape == (experts,)
        and weights.w2_global_scale.shape == (experts,)
        and 1 <= top_k <= _MAX_TOP_K
        and topk_ids.shape == (rows, top_k)
        and topk_weights.shape == (rows, top_k)
        and (expert_map is None or expert_map.dim() == 1)
        and (norm is None or tuple(norm.weight.shape) == (hidden,))
        and _smem_fits(top_k, experts)
    )


@functools.cache
def _smem_fits(top_k, experts):
    # The plan depends only on the table sizes; the other arguments are placeholders.
    plan = KawaiiMoe(
        hidden=_FC1_RECORD_K,
        inter=_FC2_RECORD_K,
        top_k=top_k,
        experts=experts,
        has_expert_map=False,
        norm_mode=0,
        grid=1,
    )
    return plan.smem_bytes <= _SMEM_LIMIT


def _dtypes_supported(
    hidden_states,
    topk_weights,
    topk_ids,
    weights,
    expert_map,
    norm,
):
    return (
        hidden_states.dtype == torch.bfloat16
        and topk_weights.dtype in _CUTE_DTYPES
        and topk_ids.dtype == torch.int32
        and weights.w13.dtype == torch.int32
        and weights.w2.dtype == torch.int32
        and weights.w13_scale.dtype == torch.float8_e4m3fn
        and weights.w2_scale.dtype == torch.float8_e4m3fn
        and weights.w13_global_scale.dtype == torch.bfloat16
        and weights.w2_global_scale.dtype == torch.bfloat16
        and (expert_map is None or expert_map.dtype == torch.int32)
        and (norm is None or norm.weight.dtype == torch.bfloat16)
    )


def kawaii_moe_supported(
    hidden_states,
    topk_weights,
    topk_ids,
    weights,
    expert_map=None,
    norm=None,
    routed_scaling_factor=1.0,
) -> bool:
    """Whether the narrow kernel takes this call; checks metadata only, no device sync."""
    inputs = (
        hidden_states,
        topk_weights,
        topk_ids,
        weights,
        expert_map,
        norm,
    )
    if not (_shape_supported(*inputs) and _dtypes_supported(*inputs)):
        return False
    tensors = [hidden_states, topk_weights, topk_ids, *_weight_tensors(weights)]
    tensors += [
        t
        for t in (
            expert_map,
            None if norm is None else norm.weight,
        )
        if t is not None
    ]
    device = hidden_states.device
    # cp.async.bulk and the 16 B vector loads need 16 B aligned bases; the
    # launch uses the current device's stream.
    if not (
        device.type == "cuda"
        and device.index == torch.cuda.current_device()
        and all(
            t.device == device and t.is_contiguous() and t.data_ptr() % 16 == 0
            for t in tensors
        )
    ):
        return False
    # A workspace is warmed only after eager eligibility on this same stream.
    # Capture must not query the driver, compile, or allocate.
    if torch.cuda.is_current_stream_capturing():
        return _warmed(*inputs, routed_scaling_factor=routed_scaling_factor)
    # The cooperative grid uses every SM; green contexts may own only a subset.
    status, green_ctx = cuda.cuStreamGetGreenCtx(
        cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )
    return status == cuda.CUresult.CUDA_SUCCESS and not int(green_ctx)


def _config(
    hidden_states,
    topk_weights,
    topk_ids,
    weights,
    expert_map,
    norm,
    routed_scaling_factor=1.0,
):
    experts, _, w13_words = weights.w13.shape
    return dict(
        hidden=hidden_states.shape[1],
        inter=w13_words // 4,
        top_k=topk_ids.shape[1],
        experts=experts,
        has_expert_map=expert_map is not None,
        norm_mode=0 if norm is None else (1 if norm.weight_first else 2),
        grid=torch.cuda.get_device_properties(
            hidden_states.device
        ).multi_processor_count,
        routed_scaling_factor=float(routed_scaling_factor),
    )


def _compile_key(config, topk_dtype):
    return tuple(config.items()), topk_dtype


def _workspace_key(kernel, device):
    # The counter protocol orders launches only by stream order and PDL, so
    # concurrent streams must not share counters or scratch.
    stream = torch.cuda.current_stream(device).cuda_stream
    return (device, stream, kernel.hidden, kernel.inter, kernel.top_k, kernel.max_jobs)


def _warmed(hidden_states, topk_weights, *rest, routed_scaling_factor=1.0):
    config = _config(
        hidden_states, topk_weights, *rest, routed_scaling_factor=routed_scaling_factor
    )
    compiled = _COMPILED.get(_compile_key(config, topk_weights.dtype))
    return (
        compiled is not None
        and _workspace_key(compiled[0], hidden_states.device) in _WORKSPACES
    )


def _workspace(kernel, device):
    """act, y, sums of squares and counters per stream; allocated outside capture."""
    key = _workspace_key(kernel, device)
    if key not in _WORKSPACES:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("kawaii_moe: run once eagerly before CUDA graph capture")
        entries = kernel.max_entries
        _WORKSPACES[key] = (
            torch.empty(entries, kernel.inter, dtype=torch.bfloat16, device=device),
            torch.empty(entries, kernel.hidden, dtype=torch.bfloat16, device=device),
            torch.empty(entries, kernel.fc2_groups, dtype=torch.float32, device=device),
            # FC1/FC2 counters for both parities, then epoch and the exit ticket.
            torch.zeros(4 * kernel.max_jobs + 2, dtype=torch.int32, device=device),
        )
    return _WORKSPACES[key]


def _compiled(config, topk_dtype, stream):
    key = _compile_key(config, topk_dtype)
    if key not in _COMPILED:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("kawaii_moe: run once eagerly before CUDA graph capture")
        kernel = KawaiiMoe(**config)
        ptrs = [
            make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=16)
            for t in _pointer_dtypes(topk_dtype)
        ]
        fn = cute.compile(
            kernel, *ptrs, Int32(1), Float32(0.0), stream, options="--enable-tvm-ffi"
        )
        _COMPILED[key] = (kernel, fn)
    return _COMPILED[key]


def kawaii_moe(
    hidden_states,
    topk_weights,
    topk_ids,
    weights,
    expert_map=None,
    norm=None,
    routed_scaling_factor=1.0,
) -> torch.Tensor:
    """Routed experts, post-expert norm and scaled top-k sum; caller checks kawaii_moe_supported."""
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    config = _config(
        hidden_states,
        topk_weights,
        topk_ids,
        weights,
        expert_map,
        norm,
        routed_scaling_factor=routed_scaling_factor,
    )
    kernel, fn = _compiled(config, topk_weights.dtype, stream)
    act, y, ss, counters = _workspace(kernel, hidden_states.device)
    out = torch.empty_like(hidden_states)
    # Unused pointers (no expert map or norm) alias a valid tensor.
    expert_map = topk_ids if expert_map is None else expert_map
    norm_weight = hidden_states if norm is None else norm.weight
    tensors = [
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map,
        *_weight_tensors(weights),
        norm_weight,
        out,
        act,
        y,
        ss,
        counters,
    ]
    dtypes = _pointer_dtypes(topk_weights.dtype)
    ptrs = [make_ptr(d, t.data_ptr()) for d, t in zip(dtypes, tensors)]
    eps = 0.0 if norm is None else norm.eps
    fn(*ptrs, Int32(hidden_states.shape[0]), Float32(eps), stream)
    return out
