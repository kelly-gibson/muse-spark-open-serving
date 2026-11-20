# Muse Spark Open serving on Hopper

*Dedicated to the open source community.*

This repository holds our final serving-track submission to the
[Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/): a patch to
[SGLang](https://github.com/sgl-project/sglang) that serves Meta's Muse Spark Open
(`Muse-Spark-Open-816B-A42B-NVFP4`) faster on an 8×H200 node.

In the official scored run the patch reached **1.8509×** the throughput of stock
SGLang v0.5.23 at the same per-user speed, as a geometric mean over four load
levels. That figure comes from one judge run. The organizers rerun and review
top submissions after the challenge, so it is provisional and it is not a
leaderboard rank.

## Result

Each row is one load level on the same node. "Credited" is the patched server's
throughput at the per-user speed stock SGLang delivers at that load, which is
what the challenge scores.

| Concurrent users | Stock tok/s/GPU | Credited tok/s/GPU | Ratio |
|---:|---:|---:|---:|
| 1 | 6,418.9 | 11,657.4 | 1.8161 |
| 4 | 9,926.2 | 17,728.2 | 1.7860 |
| 16 | 12,244.0 | 23,692.2 | 1.9350 |
| 32 | 12,695.2 | 23,740.0 | 1.8700 |
| **Geometric mean** | | | **1.8509** |

The run passed the judge's correctness gates against 176 stock references:

| Gate | Measured | Bound |
|---|---:|---:|
| Prefill KL (mean) | 0.049 | ≤ 0.11 |
| Prefill top-1 agreement | 0.82 | ≥ 0.62 |
| Decode KL (mean) | 0.011 | ≤ 0.035 |
| Later-turn agreement, 42,194 positions | 0.97 | ≥ 0.92 |

GPU memory stayed at or below stock (the limit was +2 GB per GPU), and all four
loads met the limit of 2× stock time to first token.

## What is in this repository

| Path | Contents |
|---|---|
| [`patch.diff`](patch.diff) | The submitted patch, byte for byte. SHA-256 `35f1fced3ac110da3e15dca021543b8bd1c475cd7dc52145b5486797a1594b81`. |
| [`flags`](flags) | The launch flags the submission ran with. |
| [`python/sglang/`](python/sglang) | The 43 files the patch touches, as they look after applying it, for reading on GitHub. 9 are new and 34 are modified SGLang files. |
| [`LICENSE`](LICENSE) | Apache License 2.0, copied from SGLang. |

The patch adds 5,413 lines and removes 103. It is not a complete SGLang tree:
apply it to SGLang to run it.

## Apply it

```bash
git clone https://github.com/sgl-project/sglang && cd sglang
git checkout 297e54fd105e974cdf401e5b4d20253a22c16f90   # source of v0.5.23
git apply /path/to/patch.diff
```

The challenge ran it in the `lmsysorg/sglang:v0.5.23` image with tensor and
expert parallelism of 8, using the flags in [`flags`](flags) plus the fixed
flags listed in that file's comments. The patch needs nothing outside that image
and compiles its GPU code from source at startup.

## What the patch changes

Everything new is enabled only for Muse Spark on Hopper (SM90) and for the
shapes and dtypes each piece supports. Other models, other GPUs and unsupported
shapes keep stock behaviour, and stock Marlin remains a zero-copy fallback for
the experts. Each feature has an environment switch in
[`environ.py`](python/sglang/srt/environ.py), and
`SGLANG_FORCE_MUSE_SPARK_MARLIN_MOE=1` puts the experts back on Marlin.

**Routed experts.** A new MoE backend,
[`nvfp4_w4a16_moe.py`](python/sglang/kernels/ops/moe/nvfp4_w4a16_moe.py), sits
beside Marlin and has a pure-PyTorch `forward_native` reference. It has two CuTe
DSL kernels that read the NVFP4 Marlin weight layout in place and reproduce
Marlin's BF16 dequantization bits:

- [`kawaii_moe.py`](python/sglang/kernels/ops/moe/kawaii_moe.py) serves decode
  batches. One persistent launch per layer routes entries by expert, fetches
  weights with TMA and runs `m16n8k16` MMA with the weights as the M operand.
- [`kawaii_wide_moe.py`](python/sglang/kernels/ops/moe/kawaii_wide_moe.py)
  serves prefill, where one expert takes thousands of rows. It feeds
  `wgmma.m64n128k16` over 128-row tiles and decodes the weights in registers, so
  no weight is dequantized to global memory.

**Prefill expert borrowing.**
[`kawaii_borrow.py`](python/sglang/kernels/ops/moe/kawaii_borrow.py) and
[`muse_spark_expert_borrow.py`](python/sglang/srt/layers/muse_spark/muse_spark_expert_borrow.py)
let wide prefill use a peer rank's expert weights for whole tiles. The expert
arithmetic is unchanged. All ranks admit the feature together at startup, and a
failure on any rank turns it off on every rank.

**Dense GEMMs.**
[`cublaslt_fixed.py`](python/sglang/kernels/ops/gemm/cublaslt_fixed.py) runs
BF16 GEMMs on the image's cuBLASLt with fixed algorithms chosen offline for the
H200. An unknown device, library version or shape keeps the stock GEMM.

**Prefill communication.** Large prefill projections are sharded across the
tensor-parallel ranks, and the captured prefill path uses SGLang's custom
all-reduce with a reduction that returns only the rows each rank owns
([`custom_all_reduce_v2.py`](python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py)).

**Attention and speculative decoding.** FlashAttention-3 target verification
plans its KV splits once per step and window type, and fuses the Q/K/V casts
with the FP8 KV write. The released DFlash drafter and its proposals are
unchanged.

**Hierarchical cache.** The staged write-back copies a run of consecutive pages
in one copy
([`staged_write_back.cuh`](python/sglang/kernels/jit/csrc/kvcacheio/staged_write_back.cuh)).

**Startup.** The selected kernel specializations compile and warm up before the
server reports healthy, so the first request does not pay for compilation.

## Limits

- The optimized paths are built and measured for Hopper only. They were tuned on
  the H200, and the cuBLASLt table has entries only for that device.
- The score above is a single run on one node. The patch it grew from scored
  1.8319 in an earlier run of the same kind.
- The parity tests and benchmarks from development are not in this repository.
  The challenge accepted only files under `python/sglang`.
- The model keeps eight routed experts per token, the 2,048-token sliding
  window, every norm, BF16 activations and the FP8 KV cache. Weight repacking is
  lossless.

## License

Apache License 2.0, the same license as SGLang. The files under `python/sglang`
that are not new are modified copies of SGLang files from commit `297e54fd`;
`patch.diff` shows every change. SGLang is the work of its contributors at
<https://github.com/sgl-project/sglang>.
