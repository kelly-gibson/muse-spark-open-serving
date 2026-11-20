"""Transient whole-tile prefill borrowing; no change to expert arithmetic."""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint16, Uint64
from cutlass.cute.runtime import make_ptr
from cutlass.memory import SmemAllocator

from .kawaii_wide_moe import _TOKENS as BORROW_TILE

COPY_TILES = 4  # One expert copy beside the wide kernel costs about four tiles of work.
MIN_TILES = 2
THREADS = 256
# One CTA: its warps times PLAN_LOADS are the route loads in flight.
PLAN_WARPS = 32
PLAN_LOADS = 8
MAX_ENTRIES = 1 << 21


def borrow_capacity(entries, ranks):
    return ((entries + ranks * BORROW_TILE - 1) // (ranks * BORROW_TILE)) // 2


def plan_counts_native(counts, owner_pos, ranks, *, cap):
    """Rank-independent integer plan: (expert, borrower, owner-kept routes)."""
    counts = [int(x) for x in counts]
    positions = [int(x) for x in owner_pos]
    local = len(counts) // ranks
    if (
        not local
        or len(counts) % ranks
        or sorted(positions) != list(range(len(counts)))
    ):
        raise ValueError("expected a capacity-preserving expert permutation")
    if min(counts) < 0 or sum(counts) > MAX_ENTRIES or cap < 0:
        raise ValueError("unsupported count/cost domain")
    owner = [p // local for p in positions]
    load = [
        sum(
            (n + BORROW_TILE - 1) // BORROW_TILE
            for n, r in zip(counts, owner)
            if r == q
        )
        for q in range(ranks)
    ]
    lent, borrowed, plan = set(), set(), []
    for _ in range(ranks):
        o = max(range(ranks), key=lambda r: (load[r], -r))
        free = [r for r in range(ranks) if r != o and r not in borrowed]
        mine = [e for e in range(len(counts)) if owner[e] == o and e not in lent]
        if not free or not mine:
            break
        q = min(free, key=lambda r: (load[r], r))
        e = max(mine, key=lambda e: (counts[e], -e))
        m = min((load[o] - load[q] - COPY_TILES) // 2, counts[e] // BORROW_TILE, cap)
        if m < MIN_TILES:
            break
        cut = counts[e] - BORROW_TILE * m
        plan.append((e, q, cut))
        lent.add(e)
        borrowed.add(q)
        load[o] -= m
        load[q] += m + COPY_TILES
    return plan


def borrow_plan_native(topk_ids, owner_pos, ranks):
    ids = topk_ids.flatten().long()
    counts = torch.bincount(
        ids[(ids >= 0) & (ids < len(owner_pos))], minlength=len(owner_pos)
    )
    return plan_counts_native(
        counts, owner_pos, ranks, cap=borrow_capacity(ids.numel(), ranks)
    )


def split_routes_native(topk_ids, owner_pos, ranks, rank, plan):
    """Stable route-entry ordinals; supports duplicates and negative padding IDs."""
    flat = topk_ids.flatten().long()
    local = len(owner_pos) // ranks
    valid = (flat >= 0) & (flat < len(owner_pos))
    pos = owner_pos[flat.clamp(0, len(owner_pos) - 1)]
    main = torch.where(valid & (pos // local == rank), pos % local, -1).int()
    final = main.clone()
    borrowed = torch.empty(0, dtype=torch.int32, device=flat.device)
    for e, q, cut in plan:
        entries = (flat == e).nonzero().flatten()[cut:]
        main[entries] = -1
        final[entries] = local if q == rank else -1
        if q == rank:
            borrowed = entries.int()
    return main.view_as(topk_ids), final.view_as(topk_ids), borrowed


class BorrowPlan:
    def __init__(self, experts, ranks):
        self.experts, self.ranks = experts, ranks
        self.local = experts // ranks
        self.smem_bytes = 4 * (5 * experts + 2 * ranks + ranks * PLAN_WARPS) + 128

    @cute.jit
    def __call__(
        self,
        ids: cute.Pointer,
        owners: cute.Pointer,
        enabled: cute.Pointer,
        main: cute.Pointer,
        final: cute.Pointer,
        borrowed: cute.Pointer,
        tiles: cute.Pointer,
        count: cute.Pointer,
        plan: cute.Pointer,
        entries: Int32,
        cap: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            ids,
            owners,
            enabled,
            main,
            final,
            borrowed,
            tiles,
            count,
            plan,
            entries,
            cap,
            rank,
        ).launch(
            grid=(1, 1, 1),
            block=(PLAN_WARPS * 32, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        ids: cute.Pointer,
        owners: cute.Pointer,
        enabled: cute.Pointer,
        main: cute.Pointer,
        final: cute.Pointer,
        borrowed: cute.Pointer,
        tiles: cute.Pointer,
        count: cute.Pointer,
        plan: cute.Pointer,
        entries: Int32,
        cap: Int32,
        rank: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = tid // 32, tid % 32
        G, R, T = self.experts, self.ranks, PLAN_WARPS * 32
        smem = SmemAllocator()
        hist = smem.allocate_array(Int32, G, byte_alignment=16)
        lent = smem.allocate_array(Int32, G, byte_alignment=16)
        cuts = smem.allocate_array(Int32, G, byte_alignment=16)
        round_id = smem.allocate_array(Int32, G, byte_alignment=16)
        pos = smem.allocate_array(Int32, G, byte_alignment=16)
        loads = smem.allocate_array(Int32, R, byte_alignment=16)
        used = smem.allocate_array(Int32, R, byte_alignment=16)
        scan = smem.allocate_array(Int32, R * PLAN_WARPS, byte_alignment=16)
        for e in cutlass.range(tid, G, T):
            cute.arch.store(hist + e, Int32(0))
            cute.arch.store(lent + e, Int32(-1))
            cute.arch.store(cuts + e, Int32(0))
            cute.arch.store(round_id + e, Int32(-1))
            cute.arch.store(pos + e, cute.arch.load(owners + e, Int32))
        for i in cutlass.range(tid, R * PLAN_WARPS, T):
            cute.arch.store(scan + i, Int32(0))
        for q in cutlass.range(tid, R, T):
            cute.arch.store(used + q, Int32(0))
            for j in cutlass.range_constexpr(3):
                cute.arch.store(plan + q * 3 + j, Int32(-1))
        if tid == 0:
            cute.arch.store(count, Int32(0))
        for i in cutlass.range(tid, cap * BORROW_TILE, T):
            cute.arch.store(borrowed + i, entries)
        for i in cutlass.range(tid, cap, T):
            cute.arch.store(tiles + i, Int32(-1))
        cute.arch.sync_threads()
        if cute.arch.load(enabled, Int32) != 0:
            for base in cutlass.range(tid, entries, T * PLAN_LOADS):
                got = []
                for k in cutlass.range_constexpr(PLAN_LOADS):
                    e = Int32(-1)
                    if base + k * T < entries:
                        e = cute.arch.load(ids + base + k * T, Int32)
                    got.append(e)
                for k in cutlass.range_constexpr(PLAN_LOADS):
                    if got[k] >= 0 and got[k] < G:
                        cute.arch.atomic_add(
                            hist + got[k], Int32(1), sem="relaxed", scope="cta"
                        )
        cute.arch.sync_threads()
        if tid < R:
            total = Int32(0)
            for e in cutlass.range(0, G, 1):
                if cute.arch.load(pos + e, Int32) // self.local == tid:
                    total += (
                        cute.arch.load(hist + e, Int32) + BORROW_TILE - 1
                    ) // BORROW_TILE
            cute.arch.store(loads + tid, total)
        cute.arch.sync_threads()
        if tid == 0:
            active = Int32(1)
            for step in cutlass.range(0, R, 1):
                if active != 0:
                    o = Int32(0)
                    for r in cutlass.range(1, R, 1):
                        if cute.arch.load(loads + r, Int32) > cute.arch.load(
                            loads + o, Int32
                        ):
                            o = r
                    q = Int32(-1)
                    best = Int32(2147483647)
                    for r in cutlass.range(0, R, 1):
                        v = cute.arch.load(loads + r, Int32)
                        if r != o and cute.arch.load(used + r, Int32) == 0 and v < best:
                            q = r
                            best = v
                    e = Int32(-1)
                    most = Int32(-1)
                    for j in cutlass.range(0, G, 1):
                        n = cute.arch.load(hist + j, Int32)
                        if (
                            cute.arch.load(pos + j, Int32) // self.local == o
                            and cute.arch.load(lent + j, Int32) < 0
                            and n > most
                        ):
                            e = j
                            most = n
                    m = Int32(0)
                    if q >= 0 and e >= 0:
                        m = cutlass.min(
                            cap,
                            cutlass.min(
                                most // BORROW_TILE,
                                (cute.arch.load(loads + o, Int32) - best - COPY_TILES)
                                // 2,
                            ),
                        )
                    if m < MIN_TILES:
                        active = Int32(0)
                    else:
                        cute.arch.store(lent + e, q)
                        cute.arch.store(cuts + e, most - m * BORROW_TILE)
                        cute.arch.store(round_id + e, step)
                        cute.arch.store(used + q, Int32(1))
                        cute.arch.store(loads + o, cute.arch.load(loads + o, Int32) - m)
                        cute.arch.store(loads + q, best + m + COPY_TILES)
                        cute.arch.store(plan + q * 3, e)
                        cute.arch.store(
                            plan + q * 3 + 1, cute.arch.load(pos + e, Int32)
                        )
                        cute.arch.store(plan + q * 3 + 2, m * BORROW_TILE)
        cute.arch.sync_threads()
        planned = Int32(0)
        for r in cutlass.range_constexpr(R):
            planned += cute.arch.load(used + r, Int32)
        # Each warp owns a contiguous interval, lanes reading neighbours; prefix the
        # per-warp counts, then rank lanes by match, for stable entry ordinals.
        start = warp * ((entries + T - 1) // T * 32)
        stop = cutlass.min(entries, start + (entries + T - 1) // T * 32)
        if planned != 0:
            for base in cutlass.range(start + lane, stop, 32 * PLAN_LOADS):
                got = []
                for k in cutlass.range_constexpr(PLAN_LOADS):
                    e = Int32(-1)
                    if base + 32 * k < stop:
                        e = cute.arch.load(ids + base + 32 * k, Int32)
                    got.append(e)
                for k in cutlass.range_constexpr(PLAN_LOADS):
                    if got[k] >= 0 and got[k] < G:
                        step = cute.arch.load(round_id + got[k], Int32)
                        if step >= 0:
                            cute.arch.atomic_add(
                                scan + step * PLAN_WARPS + warp,
                                Int32(1),
                                sem="relaxed",
                                scope="cta",
                            )
            cute.arch.sync_threads()
            if tid < R:
                total = Int32(0)
                for w in cutlass.range(0, PLAN_WARPS, 1):
                    p = scan + tid * PLAN_WARPS + w
                    n = cute.arch.load(p, Int32)
                    cute.arch.store(p, total)
                    total += n
            cute.arch.sync_threads()
        for base in cutlass.range(start, stop, 32 * PLAN_LOADS):
            got = []
            for k in cutlass.range_constexpr(PLAN_LOADS):
                e = Int32(-1)
                if base + 32 * k + lane < stop:
                    e = cute.arch.load(ids + base + 32 * k + lane, Int32)
                got.append(e)
            for k in cutlass.range_constexpr(PLAN_LOADS):
                i = base + 32 * k + lane
                e = got[k]
                slot = Int32(-1)
                result = Int32(-1)
                step = Int32(-1)
                if e >= 0 and e < G:
                    p = cute.arch.load(pos + e, Int32)
                    if p // self.local == rank:
                        slot = p % self.local
                    result = slot
                    step = cute.arch.load(round_id + e, Int32)
                if planned != 0:
                    peers = cute.arch.match_sync(0xFFFFFFFF, step)
                    ordinal = Int32(0)
                    if step >= 0:
                        ordinal = cute.arch.load(
                            scan + step * PLAN_WARPS + warp, Int32
                        ) + Int32(cute.arch.popc(peers & cute.arch.lanemask_lt()))
                    cute.arch.sync_warp()
                    if step >= 0:
                        if (peers & cute.arch.lanemask_gt()) == 0:
                            cute.arch.store(
                                scan + step * PLAN_WARPS + warp, ordinal + 1
                            )
                        cut = cute.arch.load(cuts + e, Int32)
                        if ordinal >= cut:
                            slot = Int32(-1)
                            result = Int32(-1)
                            if cute.arch.load(lent + e, Int32) == rank:
                                result = Int32(self.local)
                                cute.arch.store(borrowed + ordinal - cut, i)
                    cute.arch.sync_warp()
                if i < stop:
                    cute.arch.store(main + i, slot)
                    cute.arch.store(final + i, result)
        n = cutlass.max(Int32(0), cute.arch.load(plan + rank * 3 + 2, Int32))
        if tid == 0:
            cute.arch.store(count, n)
        for i in cutlass.range(tid, cap, T):
            if i * BORROW_TILE < n:
                cute.arch.store(tiles + i, Int32(0))


class BorrowCopy:
    def __init__(self, local):
        self.local = local

    @cute.jit
    def __call__(
        self,
        peers: cute.Pointer,
        destinations: cute.Pointer,
        sizes: cute.Pointer,
        plan: cute.Pointer,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(peers, destinations, sizes, plan, rank).launch(
            grid=(132, 6, 1), block=(THREADS, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        peers: cute.Pointer,
        destinations: cute.Pointer,
        sizes: cute.Pointer,
        plan: cute.Pointer,
        rank: Int32,
    ):
        block, field, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        pos = cute.arch.load(plan + rank * 3 + 1, Int32)
        if pos >= 0:
            n = cute.arch.load(sizes + field, Int64)
            source = (
                cute.arch.load(peers + (pos // self.local) * 6 + field, Int64)
                + Int64(pos % self.local) * n
            )
            dest = cute.arch.load(destinations + field, Int64)
            self.copy_field(source, dest, n, block, tid)

    @cute.jit
    def copy_field(
        self, source: Int64, dest: Int64, n: Int64, block: Int32, tid: Int32
    ):
        a16 = cutlass.inttoptr(source, cute.AddressSpace.gmem, Uint16)
        b16 = cutlass.inttoptr(dest, cute.AddressSpace.gmem, Uint16)
        a = cutlass.inttoptr(source, cute.AddressSpace.gmem, Uint64)
        b = cutlass.inttoptr(dest, cute.AddressSpace.gmem, Uint64)
        if n == 2:
            if block == 0 and tid == 0:
                cute.arch.store(b16, cute.arch.load(a16, Uint16))
        else:
            for i in cutlass.range(
                block * THREADS + tid, Int32(n) // 16, 132 * THREADS
            ):
                cute.arch.store(b + 2 * i, cute.arch.load(a + 2 * i, Uint64))
                cute.arch.store(b + 2 * i + 1, cute.arch.load(a + 2 * i + 1, Uint64))


@functools.cache
def compiled_plan(experts, ranks):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm borrowing before capture")
    ptrs = [
        make_ptr(Int32, 16, cute.AddressSpace.gmem, assumed_align=16) for _ in range(9)
    ]
    return cute.compile(
        BorrowPlan(experts, ranks),
        *ptrs,
        Int32(1),
        Int32(1),
        Int32(0),
        cuda.CUstream(0),
        options="--enable-tvm-ffi",
    )


@functools.cache
def compiled_copy(local):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm borrowing before capture")
    ptrs = [
        make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=16)
        for t in (Int64, Int64, Int64, Int32)
    ]
    return cute.compile(
        BorrowCopy(local), *ptrs, Int32(0), cuda.CUstream(0), options="--enable-tvm-ffi"
    )
