<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# LayerNorm + Affine + Cast (f32 -> bf16)

This design implements a fused row-wise LayerNorm, per-column affine (real
`gamma`/`beta`, not fixed to 1/0), and `float32 -> bfloat16` narrowing cast,
in one kernel / one dispatch, across an 8-core sequence. NPU2-only (the
underlying kernel lives under `aie_kernels/aie2p/`).

Per row of `embedding_dim`:

```
out = ((x - mean(x)) / sqrt(var(x) + eps)) * gamma + beta -> bfloat16
```

`gamma`/`beta` are the same for every row and are packed into one
`[2 * embedding_dim]` buffer (gamma then beta); every core loads it once,
before its row loop, rather than re-loading it per row.

This is a seam op: a dataflow stage that needs f32 precision for the
LayerNorm reduction (the mean/variance cancel badly in bf16 for
near-zero-mean rows) but whose consumer is a bf16 matmul can produce its
final, affine-applied, already-narrowed output in one on-chip dispatch,
instead of leaving the device to requantize on the host or fusing gamma/beta
into a two-kernel LN -> cast chain that pays for the reduction twice.

## Source Files Overview

1. `ln_affine_cast.py`: IRON design. Structurally mirrors
   [`ml/norm`](../norm)'s 8-core row-split `@iron.jit` design, extended two
   ways: input and output tiles have different dtypes (f32 in, bf16 out --
   as in [`ml/cast_f32_bf16`](../cast_f32_bf16)), and there is a third,
   per-core-constant parameter tensor (gamma/beta) alongside the per-row
   data tensor. Neither fits `transform_parallel`/`transform_parallel_binary`
   (uniform dtype, two tensors only), hence the explicit
   `ObjectFifo`/`Worker`/`Runtime` wiring.

1. `ln_affine_cast.cc`: AIE2P kernel, pulled from
   [`aie_kernels/aie2p/`](../../../aie_kernels/aie2p/).

1. `test.cpp`: C++ testbench. Loads the compiled XCLBIN + `insts.bin` via
   `setup_and_run_aie`, computes the f32 reference on the host, and checks
   the output against it with a tolerance (the on-chip two-pass f32
   reduction is not bit-exact against an exact f32 host reduction).

## Usage

### Standalone JIT verification

```shell
python3 ln_affine_cast.py --dev npu2
```

### C++ Testbench

```shell
make
make run
```
