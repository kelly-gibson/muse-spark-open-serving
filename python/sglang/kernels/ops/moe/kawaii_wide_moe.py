"""Kawaii-MoE wide path: NVFP4 experts over 128-row tiles on Hopper WGMMA, in CuTe DSL.

``kawaii_moe`` decodes an expert's weights once per 16 routed rows, which suits decode
batches. In prefill one expert takes thousands of rows, so here the same Marlin words feed
``wgmma.m64n128k16``: the weights are the 64-row M operand, decoded in registers with
Marlin's bits, and 128 routed rows are the N operand in shared memory. No weight is
dequantized to global memory, and stock Marlin stays a zero-copy fallback.

One launch per projection, one CTA per (row tile, 128 or 256 output columns):

- The row tiles are the stock aligner's blocks at block size 128: tile ``t`` belongs to
  expert ``tile_experts[t]`` and holds entries ``ids[128t .. 128t+127]``, an entry being
  ``token * top_k + slot``. Pads carry an id of at least ``entries`` and store nothing.
- Warpgroup 0 fills a ring of stages, 64 K each: eight lanes copy one routed row's 128 B,
  and warp 0 also fetches the stage's Marlin words and scales as TMA boxes. The whole ring
  is in flight at once.
- Warpgroups 1 and 2 each decode two 64-column weight groups per k tile and multiply both
  by the same stage of rows: gate and up of the same activation columns in FC1, two
  neighbouring output groups in FC2. So SwiGLU needs no exchange between threads.
- The epilogue applies Marlin's BF16 rounding and global scale and stores per entry: FC1
  writes ``act[entry]``, FC2 writes ``y[entry]`` for the stock post-norm weighted sum.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, warpgroup
from cutlass.cute.runtime import make_ptr
from cutlass.memory import SmemAllocator
from cutlass.tensor_utils import LayoutEnum

from sglang.kernels.ops.moe.kawaii_moe import (
    _FP4_BF16X2_MASK,
    _bf16_scale,
    _lds1,
    _load_f32,
    _silu_mul,
    _store_bf16,
    _tma_box,
    _wait_barrier,
    mul_bf16x2_by_half,
    scale_pair_bits,
)

# Hardware constraint: one fp32 accumulator of m64nN is N/2 registers per thread, and a
# warpgroup holds two, so 128 rows is the widest tile inside the 255-register file.
_TOKENS = 128
_STAGE_K = 64  # one 128 B swizzled row per token and stage
_K_TILES = _STAGE_K // 16
_STAGES = 4  # Each slot holds one K=64 stage until both consumer warpgroups release it.
_WG = 128
_THREADS = 3 * _WG
# Hardware constraint: 65,536 registers per SM; 128 * 40 + 256 * 232 = 64,512.
_PRODUCER_REGS = 40
_CONSUMER_REGS = 232
# A CTA's weights for one stage, as int32: two TMA boxes of Marlin words, each k tiles by
# two column groups (128 words per group), then two boxes of scale bytes (64 per group).
# Hardware constraint: a TMA box is at most 256 elements wide.
_BOX_WORDS = 2 * 128
_BOX_SCALE_WORDS = 2 * 16
_STAGE_WORDS = 2 * _K_TILES * _BOX_WORDS
_STAGE_WEIGHT_WORDS = _STAGE_WORDS + 2 * _K_TILES * _BOX_SCALE_WORDS
# Barriers per ring slot: rows landed, weights landed, slot free again.
_ROWS_FULL, _WEIGHTS_FULL, _EMPTY = 0, _STAGES, 2 * _STAGES


def a_fragment_coords(warp, lane, u):
    """A register ``u`` of a WGMMA thread -> (weight row, first K) in its 64 x 16 tile."""
    return 16 * warp + lane // 4 + 8 * (u % 2), 2 * (lane % 4) + 8 * (u // 2)


def c_fragment_coords(warp, lane, i):
    """Accumulator element ``i`` of a WGMMA thread -> (weight row, routed row)."""
    return (
        16 * warp + lane // 4 + 8 * ((i // 2) % 2),
        8 * (i // 4) + 2 * (lane % 4) + i % 2,
    )


def stage_weight_words(fc1, group, warp, lane):
    """A consumer thread's int32 offsets inside one stage of weights, at k tile 0.

    Returns (word, scale, second word, second scale) for its two column groups; a k tile
    adds ``_BOX_WORDS`` and ``_BOX_SCALE_WORDS``. FC1's boxes are the gate pair and the up
    pair, and multiply warpgroup ``group`` takes column ``group`` of each. FC2's boxes are
    two neighbouring pairs, and warpgroup ``group`` takes both columns of box ``group``.
    """
    lane_word = lane * 4 + warp
    lane_scale = _STAGE_WORDS + (lane // 4) * 2 + warp // 2
    box, scale_box = _K_TILES * _BOX_WORDS, _K_TILES * _BOX_SCALE_WORDS
    if fc1:
        word, scale = group * 128 + lane_word, group * 16 + lane_scale
        return word, scale, word + box, scale + scale_box
    word, scale = group * box + lane_word, group * scale_box + lane_scale
    return word, scale, word + 128, scale + 16


def box_column(fc1, bx, b, out_cols):
    """Stage box ``b`` of CTA column ``bx`` -> its pair of weight column groups.

    FC1's up pair lies ``out_cols / 64`` groups after its gate pair; FC2's second pair is
    the next one.
    """
    return bx + b * (out_cols // 128) if fc1 else 2 * bx + b


def check_fragment_order(mma):
    """The two fragment maps against the DSL's own partitions, at trace time."""
    for thread in (0, 5, 37, 70, 127):
        warp, lane = thread // 32, thread % 32
        thr = mma.get_slice(thread)
        a = thr.partition_A(cute.make_identity_tensor((64, 16)))
        c = thr.partition_C(cute.make_identity_tensor((64, _TOKENS)))
        for u in range(4):
            got = tuple(int(x) for x in a[2 * u])
            assert got == a_fragment_coords(warp, lane, u), (thread, u, got)
        for i in range(_TOKENS // 2):
            got = tuple(int(x) for x in c[i])
            assert got == c_fragment_coords(warp, lane, i), (thread, i, got)


def _restore_signs(magnitude, mask, signs):
    """Select sign bits from signs and all other bits from magnitude."""
    if isinstance(magnitude, Uint32):
        # LOP3 truth table 0xB8 computes (magnitude & ~mask) | (signs & mask).
        return cute.arch.lop3(magnitude, Uint32(mask), signs, 0xB8)
    return magnitude ^ ((magnitude ^ signs) & mask)


def fp4_fragment_bits(q):
    """One Marlin FP4 word -> A registers a0..a3 as BF16x2 bits of e2m1 * 2**-126.

    Bit-identical to Marlin's ``dequant<nv_bfloat162, kFE2M1f>``. The sign selection also
    supports numpy uint32 arrays for the CPU numerical reference.
    """
    # Each 16-bit half holds four E2M1 nibbles. Move sign bits 3 and 7 to
    # bits 9 and 13 in low, and keep sign bits 11 and 15 in high.
    low = _restore_signs(q, 0x22002200, q << 6)
    high = _restore_signs(q >> 6, 0x88008800, q)
    # In Marlin A-register order the nibbles start at bits 0, 8, 4, 12.
    # After each shift, 0x81C0 keeps the BF16 sign at bit 15 and magnitude at bits 8..6.
    return (
        (low << 6) & _FP4_BF16X2_MASK,
        (high << 4) & _FP4_BF16X2_MASK,
        (low << 2) & _FP4_BF16X2_MASK,
        high & _FP4_BF16X2_MASK,
    )


def _decode(word, scale_word, shift, fragment):
    """One Marlin word and its scale pair -> the four A registers of one k tile."""
    a = fp4_fragment_bits(word)
    scale = scale_pair_bits(scale_word, shift)
    for u in range(4):
        fragment[u] = mul_bf16x2_by_half(a[u], scale, u % 2 == 1)


class WideFc:
    """One projection of the wide path: FC1 with SwiGLU, or FC2.

    ``size_k`` and ``size_n`` are the Marlin matrix's K and N: (hidden, 2 * inter) for FC1
    with the gate columns first, (inter, hidden) for FC2.
    """

    def __init__(self, fc1, size_k, size_n, experts):
        self.fc1, self.k, self.n, self.experts = fc1, size_k, size_n, experts
        self.out_cols = size_n // 2 if fc1 else size_n
        self.grid_x = self.out_cols // 128 if fc1 else self.out_cols // 256

    def _weight_tmas(self, q, s):
        """TMA atoms over the Marlin words and scales [E * K/16, 2N or N], boxed per stage."""
        rows = self.experts * (self.k // 16)
        out = []
        for ptr, cols, box_cols in (
            (q, 2 * self.n, _BOX_WORDS),
            (s, self.n, 4 * _BOX_SCALE_WORDS),
        ):
            t = cute.make_tensor(ptr, cute.make_layout((rows, cols), stride=(cols, 1)))
            box = (_K_TILES, box_cols)
            atom, tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                t,
                cute.make_layout(box, stride=(box_cols, 1)),
                box,
            )
            out += [atom, tma]
        return out

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer,
        s: cute.Pointer,
        global_scale: cute.Pointer,
        x: cute.Pointer,
        ids: cute.Pointer,
        tile_experts: cute.Pointer,
        counts: cute.Pointer,
        out: cute.Pointer,
        entries: Int32,
        row_div: Int32,
        tiles: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            *self._weight_tmas(q, s),
            global_scale,
            x,
            ids,
            tile_experts,
            counts,
            out,
            entries,
            row_div,
        ).launch(
            grid=(self.grid_x, tiles, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
            # One CTA per SM by registers; without this ptxas drops setmaxnreg.
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q_atom: cute.CopyAtom,
        q: cute.Tensor,
        s_atom: cute.CopyAtom,
        s: cute.Tensor,
        global_scale: cute.Pointer,
        x: cute.Pointer,
        ids: cute.Pointer,
        tile_experts: cute.Pointer,
        counts: cute.Pointer,
        out: cute.Pointer,
        entries: Int32,
        row_div: Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        bx, tile, _ = cute.arch.block_idx()
        expert = cute.arch.load(tile_experts + tile, Int32)
        padded = cute.arch.load(counts, Int32)
        # Uniform across the CTA. Inactive and unused tiles carry a negative expert.
        if tile * _TOKENS < padded and expert >= 0 and expert < self.experts:
            smem = SmemAllocator()
            atom = warpgroup.make_smem_layout_atom(
                sm90.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, _STAGE_K),
                BFloat16,
            )
            layout = cute.tile_to_shape(
                atom, (_TOKENS, _STAGE_K, _STAGES), order=(0, 1, 2)
            )
            rows = smem.allocate_tensor(
                BFloat16, layout.outer, byte_alignment=1024, swizzle=layout.inner
            )
            weights = smem.allocate_array(
                Uint32, _STAGES * _STAGE_WEIGHT_WORDS, byte_alignment=128
            )
            entry_ids = smem.allocate_array(Int32, _TOKENS, byte_alignment=16)
            row_ids = smem.allocate_array(Int32, _TOKENS, byte_alignment=16)
            bars = smem.allocate_array(Int64, 3 * _STAGES, byte_alignment=8)
            if thread == 0:
                for stage in cutlass.range_constexpr(_STAGES):
                    # One arrival per producer thread, when its row copies land.
                    cute.arch.mbarrier_init(bars + (_ROWS_FULL + stage), _WG)
                    # Warp 0's arrival plus its expected TMA bytes.
                    cute.arch.mbarrier_init(bars + (_WEIGHTS_FULL + stage), 1)
                    # One arrival per consumer thread after its last read of the slot.
                    cute.arch.mbarrier_init(bars + (_EMPTY + stage), 2 * _WG)
                cute.arch.mbarrier_init_fence()
            if thread < _TOKENS:
                entry = cute.arch.load(ids + (tile * _TOKENS + thread), Int32)
                cute.arch.store(entry_ids + thread, entry)
                # Padding reads row 0 to keep the load in bounds. The epilogue skips its store.
                row = Int32(0)
                if entry < entries:
                    row = entry // row_div
                cute.arch.store(row_ids + thread, row)
            cute.arch.sync_threads()
            # The one role branch; setmaxnreg must be its first statement on both sides.
            if cute.arch.make_warp_uniform(cute.arch.warp_idx()) < _WG // 32:
                cute.arch.setmaxregister_decrease(_PRODUCER_REGS)
                self._produce(
                    thread,
                    bx,
                    expert,
                    (q_atom, q, s_atom, s),
                    x,
                    rows,
                    row_ids,
                    weights,
                    bars,
                )
            else:
                cute.arch.setmaxregister_increase(_CONSUMER_REGS)
                self._consume(
                    thread - _WG,
                    bx,
                    expert,
                    global_scale,
                    out,
                    entries,
                    rows,
                    weights,
                    entry_ids,
                    bars,
                )

    @cute.jit
    def _produce(
        self,
        thread: Int32,
        bx: Int32,
        expert: Int32,
        tmas: tuple,
        x: cute.Pointer,
        rows: cute.Tensor,
        row_ids: cute.Pointer,
        weights: cute.Pointer,
        bars: cute.Pointer,
    ):
        """Eight lanes stream one routed row per copy; warp 0 also fetches the weight boxes."""
        q_atom, q, s_atom, s = tmas
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(), BFloat16, num_bits_per_copy=128)
        copy = cute.make_tiled_copy_tv(atom, cute.make_layout(8), cute.make_layout(8))
        # A warp's copy then covers four rows of 128 contiguous bytes. Measured on H200:
        # with one row per thread it touched 32 lines and cost 100 ns of every k tile.
        team = thread // 8
        part = copy.get_slice(thread % 8)
        starts = [
            Int64(cute.arch.load(row_ids + (16 * i + team), Int32)) * self.k
            for i in range(_TOKENS // 16)
        ]
        for stage in cutlass.range(0, self.k // _STAGE_K, 1, unroll=1):
            slot = stage % _STAGES
            _wait_barrier(bars + (_EMPTY + slot), ((stage // _STAGES) % 2) ^ 1)
            # Measured on H200: per-thread 4 B loads of these words cost 150 ns of every
            # k tile however early they were issued, so they come as TMA boxes instead.
            if thread < 32:
                landed = bars + (_WEIGHTS_FULL + slot)
                if thread == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        landed, 4 * _STAGE_WEIGHT_WORDS
                    )
                cute.arch.sync_warp()
                k_row = expert * (self.k // 16) + stage * _K_TILES
                base = weights + slot * _STAGE_WEIGHT_WORDS
                for b in cutlass.range_constexpr(2):
                    column = box_column(self.fc1, bx, b, self.out_cols)
                    words = cute.recast_ptr(
                        base + b * _K_TILES * _BOX_WORDS, dtype=Int32
                    )
                    scales = cute.recast_ptr(
                        base + (_STAGE_WORDS + b * _K_TILES * _BOX_SCALE_WORDS),
                        dtype=Uint8,
                    )
                    _tma_box(
                        q_atom, q, k_row, column, words, _K_TILES, _BOX_WORDS, landed
                    )
                    _tma_box(
                        s_atom,
                        s,
                        k_row,
                        column,
                        scales,
                        _K_TILES,
                        4 * _BOX_SCALE_WORDS,
                        landed,
                    )
            for i in cutlass.range_constexpr(_TOKENS // 16):
                # Rows are K BF16 long and K is a multiple of 64, so a stage keeps x's 16 B.
                src = cute.make_tensor(
                    (x + (starts[i] + stage * _STAGE_K)).align(16),
                    cute.make_layout(_STAGE_K),
                )
                cute.copy(
                    copy,
                    part.partition_S(src),
                    part.partition_D(rows[(16 * i + team, None, slot)]),
                )
            # Arrives when this thread's copies land, so the whole ring is in flight:
            # measured on H200, waiting per stage cost 100 ns of every k tile.
            cute.arch.cp_async_mbarrier_arrive_noinc(bars + (_ROWS_FULL + slot))

    @cute.jit
    def _consume(
        self,
        thread: Int32,
        bx: Int32,
        expert: Int32,
        global_scale: cute.Pointer,
        out: cute.Pointer,
        entries: Int32,
        rows: cute.Tensor,
        weights: cute.Pointer,
        entry_ids: cute.Pointer,
        bars: cute.Pointer,
    ):
        group = thread // _WG
        warp = (thread % _WG) // 32
        lane = thread % 32
        mma = cute.make_tiled_mma(
            warpgroup.MmaF16BF16Op(
                BFloat16,
                Float32,
                (64, _TOKENS, 16),
                warpgroup.OperandSource.RMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            ),
            (1, 1, 1),
        )
        check_fragment_order(mma)
        mma.set(warpgroup.Field.ACCUMULATE, True)
        thr = mma.get_slice(thread % _WG)
        shared = mma.get_slice(0)
        stage_b = shared.make_fragment_B(shared.partition_B(rows))
        # Two register sets of the two groups' A fragments: k tile kb decodes into set kb % 2
        # while the WGMMAs of k tile kb - 1 still read the other set.
        a0 = [
            cute.make_rmem_tensor(thr.partition_shape_A((64, 16)), BFloat16)
            for _ in range(2)
        ]
        a1 = [
            cute.make_rmem_tensor(thr.partition_shape_A((64, 16)), BFloat16)
            for _ in range(2)
        ]
        a0_bits = [cute.recast_tensor(t, Uint32) for t in a0]
        a1_bits = [cute.recast_tensor(t, Uint32) for t in a1]
        acc0 = cute.make_rmem_tensor(thr.partition_shape_C((64, _TOKENS)), Float32)
        acc1 = cute.make_rmem_tensor(thr.partition_shape_C((64, _TOKENS)), Float32)
        acc0.fill(0.0)
        acc1.fill(0.0)

        # First of this warpgroup's two output column groups.
        g0 = 4 * bx + 2 * group
        if cutlass.const_expr(self.fc1):
            g0 = 2 * bx + group
        word, scale, second_word, second_scale = stage_weight_words(
            self.fc1, group, warp, lane
        )
        shift = Uint32(8 * (1 - warp % 2))
        for stage in cutlass.range(0, self.k // _STAGE_K, 1, unroll=1):
            slot = stage % _STAGES
            _wait_barrier(bars + (_ROWS_FULL + slot), (stage // _STAGES) % 2)
            _wait_barrier(bars + (_WEIGHTS_FULL + slot), (stage // _STAGES) % 2)
            # The row copies landed through the generic proxy; WGMMA reads through the async one.
            cute.arch.fence_view_async_shared()
            base = weights.toint() + 4 * (slot * _STAGE_WEIGHT_WORDS)
            for kb in cutlass.range_constexpr(_K_TILES):
                tile = base + 4 * kb * _BOX_WORDS
                scales = base + 4 * kb * _BOX_SCALE_WORDS
                _decode(
                    _lds1(tile + 4 * word),
                    _lds1(scales + 4 * scale),
                    shift,
                    a0_bits[kb % 2],
                )
                _decode(
                    _lds1(tile + 4 * second_word),
                    _lds1(scales + 4 * second_scale),
                    shift,
                    a1_bits[kb % 2],
                )
                warpgroup.fence()
                cute.gemm(
                    mma,
                    acc0,
                    a0[kb % 2][(None, None, 0)],
                    stage_b[(None, None, kb, slot)],
                    acc0,
                )
                cute.gemm(
                    mma,
                    acc1,
                    a1[kb % 2][(None, None, 0)],
                    stage_b[(None, None, kb, slot)],
                    acc1,
                )
                warpgroup.commit_group()
                # A set is read until its group retires: keep k tile kb in flight while kb + 1
                # decodes into the other set, and drain at the stage's end before the slot's release.
                warpgroup.wait_group(0 if kb == _K_TILES - 1 else 1)
            cute.arch.mbarrier_arrive(bars + (_EMPTY + slot))

        global_bits = _load_f32(global_scale + expert)
        for i in cutlass.range_constexpr(_TOKENS // 2):
            row, token = c_fragment_coords(warp, lane, i)
            entry = cute.arch.load(entry_ids + token, Int32)
            first = _bf16_scale(acc0[i], global_bits)
            second = _bf16_scale(acc1[i], global_bits)
            if entry < entries:
                dst = out + (Int64(entry) * self.out_cols + (g0 * 64 + row))
                if cutlass.const_expr(self.fc1):
                    _store_bf16(dst, _silu_mul(first, second))
                else:
                    _store_bf16(dst, first)
                    _store_bf16(dst + 64, second)


# Element types of WideFc.__call__'s pointers, in argument order.
_POINTER_DTYPES = (Int32, Uint8, BFloat16, BFloat16, Int32, Int32, Int32, BFloat16)


@functools.cache
def _compiled(fc1, size_k, size_n, experts):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "kawaii_wide_moe: run once eagerly before CUDA graph capture"
        )
    kernel = WideFc(fc1, size_k, size_n, experts)
    ptrs = [
        make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=16)
        for t in _POINTER_DTYPES
    ]
    return cute.compile(
        kernel,
        *ptrs,
        Int32(1),
        Int32(1),
        Int32(1),
        cuda.CUstream(0),
        options="--enable-tvm-ffi",
    )


def wide_fc_supported(
    fc1,
    words,
    scales,
    global_scale,
    x,
    ids,
    tile_experts,
    counts,
    out,
    top_k,
):
    """Whether ``wide_fc`` takes these tensors: Hopper, Marlin NVFP4, whole CTA columns."""
    tensors = (words, scales, global_scale, x, ids, tile_experts, counts, out)
    if any(t.device != words.device or not t.is_contiguous() for t in tensors):
        return False
    if (
        words.device.type != "cuda"
        or torch.cuda.get_device_capability(words.device) != (9, 0)
        or words.dtype != torch.int32
        or scales.dtype != torch.float8_e4m3fn
        or ids.dtype != torch.int32
        or tile_experts.dtype != torch.int32
        or counts.dtype != torch.int32
        or any(t.ndim != 1 for t in (ids, tile_experts, counts))
        or any(t.dtype != torch.bfloat16 for t in (global_scale, x, out))
        or words.ndim != 3
        or x.ndim != 2
        or out.ndim != 2
    ):
        return False
    experts, k_tiles, twice_n = words.shape
    size_k, size_n = 16 * k_tiles, twice_n // 2
    out_cols = size_n // 2 if fc1 else size_n
    return (
        experts > 0
        and size_k > 0
        and out_cols > 0
        and size_k % _STAGE_K == 0
        and twice_n == 2 * size_n
        and size_n == out_cols * (2 if fc1 else 1)
        and out_cols % (128 if fc1 else 256) == 0
        and scales.shape == (experts, k_tiles, size_n)
        and global_scale.numel() == experts
        and x.shape[1] == size_k
        and out.shape[1] == out_cols
        and isinstance(top_k, int)
        and 1 <= top_k < 2**31
        and 0 < out.shape[0] < 2**31
        and x.shape[0] * (top_k if fc1 else 1) >= out.shape[0]
        and ids.numel() >= _TOKENS
        and 1 <= tile_experts.numel() < 2**31
        and counts.numel() >= 1
        and all(t.data_ptr() % 16 == 0 for t in tensors)
    )


def wide_fc(
    fc1,
    words,
    scales,
    global_scale,
    x,
    ids,
    tile_experts,
    counts,
    out,
    top_k,
):
    """One projection over the aligner's 128-row tiles; unsupported inputs raise ValueError.

    ``words`` int32 ``[E, K/16, 2N]`` and ``scales`` e4m3 ``[E, K/16, N]`` are the Marlin
    NVFP4 layout, ``global_scale`` BF16 ``[E]`` carries Marlin's exponent bias. ``ids``
    int32 ``[capacity]`` and ``tile_experts`` int32 ``[ceil(capacity / 128)]`` are the aligner's
    sorted entries and block experts, ``counts[0]`` its padded entry count, which must
    not exceed ``ids.numel()`` and must be a multiple of 128. The launch covers the whole
    table; tiles past ``counts[0]`` exit. Entry IDs must be nonnegative (padding is ``entries``).
    FC1 reads ``x[entry // top_k]`` and writes ``out[entry]`` = SwiGLU, ``[entries, N/2]``;
    FC2 reads ``x[entry]`` and writes ``out[entry]``, ``[entries, N]``. Entries of other
    tiles keep their old contents. No allocation, and no host read of the routes.
    """
    if not wide_fc_supported(
        fc1,
        words,
        scales,
        global_scale,
        x,
        ids,
        tile_experts,
        counts,
        out,
        top_k,
    ):
        raise ValueError(
            "wide_fc requires SM90, aligned Marlin NVFP4 tensors and compatible row/route shapes"
        )
    experts, k_tiles, twice_n = words.shape
    fn = _compiled(bool(fc1), 16 * k_tiles, twice_n // 2, experts)
    tensors = (words, scales, global_scale, x, ids, tile_experts, counts, out)
    ptrs = [make_ptr(d, t.data_ptr()) for d, t in zip(_POINTER_DTYPES, tensors)]
    with torch.cuda.device(words.device):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        fn(
            *ptrs,
            Int32(out.shape[0]),
            # FC1 reads x by token, FC2 reads act by entry.
            Int32(top_k if fc1 else 1),
            Int32(tile_experts.numel()),
            stream,
        )
    return out


def wide_fc_warm(fc1, words, scales, global_scale):
    """Compile this projection and launch it once on an empty route table.

    Call outside capture and before serving. Returns False for unsupported weights.
    """
    if (
        words.device.type != "cuda"
        or torch.cuda.get_device_capability(words.device) != (9, 0)
        or words.ndim != 3
    ):
        return False
    _, k_tiles, twice_n = words.shape
    kwargs = dict(dtype=torch.bfloat16, device=words.device)
    x = torch.zeros(1, 16 * k_tiles, **kwargs)
    out = torch.zeros(1, twice_n // 4 if fc1 else twice_n // 2, **kwargs)
    ids = torch.zeros(_TOKENS, dtype=torch.int32, device=words.device)
    tile_experts = torch.zeros(1, dtype=torch.int32, device=words.device)
    # A padded count of zero: the launch's one row of CTAs exits at its guard.
    counts = torch.zeros(1, dtype=torch.int32, device=words.device)
    if not wide_fc_supported(
        fc1, words, scales, global_scale, x, ids, tile_experts, counts, out, 1
    ):
        return False
    wide_fc(fc1, words, scales, global_scale, x, ids, tile_experts, counts, out, 1)
    return True
