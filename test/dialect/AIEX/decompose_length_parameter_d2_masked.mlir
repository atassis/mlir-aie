//===- decompose_length_parameter_d2_masked.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// d2 size 1 with a real (non-contiguous) stride: isContiguousTransfer misreads
// this as truly contiguous (see BdLowering.h's hasLengthParameter). A
// length_parameter BD must never reach decomposeRecursive on that misreading
// -- it treats a size-1 dimension as free capacity and silently discards its
// stride when redistributing overflow. aie-decompose-large-dma-bd checks
// hasLengthParameter before any contiguity/decomposition logic instead: pass
// through untouched if the BD already verifies as-is, otherwise fail loudly.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file --verify-diagnostics %s | FileCheck %s

// This BD does not verify as-is (d0=4124 exceeds the 10-bit wrap limit), so
// it must fail loudly rather than silently reach decomposeRecursive.
module {
  aiex.scratchpad_parameter @len : i32
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @length_param_d2_masked(%in: memref<16384xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        // expected-error@+1 {{splitting a length_parameter buffer descriptor into multiple descriptors is not implemented}}
        aie.dma_bd(%in : memref<16384xi32> offset = 0 len = 4124 sizes = [1, 1, 1, 4124] strides = [0, 8248, 0, 1])
          {length_parameter = @len, length_granule = 4 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// A FITTING length_parameter BD (verifies as-is) must pass through the
// decompose pass byte-identically: the probe shape plus a real operator tap
// (bf16, a 512-wide row, 64 rows per block, jump 4 blocks' worth for d2).

// CHECK-LABEL: aie.runtime_sequence @length_param_fits
// CHECK: aie.dma_bd
// CHECK-SAME: sizes = [1, 1, 64, 512]
// CHECK-SAME: strides = [0, 262144, 512, 1]
module {
  aiex.scratchpad_parameter @len2 : i32
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @length_param_fits(%in: memref<1048576xbf16>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<1048576xbf16> offset = 0 len = 32768 sizes = [1, 1, 64, 512] strides = [0, 262144, 512, 1])
          {length_parameter = @len2, length_granule = 512 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}
