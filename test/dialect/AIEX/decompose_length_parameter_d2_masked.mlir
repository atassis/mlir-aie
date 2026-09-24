//===- decompose_length_parameter_d2_masked.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// d2 size 1 with a real (non-contiguous) stride: isContiguousTransfer misreads
// this as truly contiguous (see BdLowering.h's hasLengthParameter), so
// aie-decompose-large-dma-bd used to bail via its own early-return before ever
// trying to decompose -- a silent no-op, not the "not implemented" diagnostic
// (verifyStridesWraps never bounds d2's hardware size, so a d0/d1-contiguous,
// d2-masked BD always has a legal single-BD refactor; this shape cannot reach
// that diagnostic through this pass, only the silent no-op the guard fixes).

// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' %s | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence @length_param_d2_masked
// The pass must attempt this BD instead of silently leaving it untouched.
// CHECK-NOT: sizes = [1, 1, 1, 4124]
module {
  aiex.scratchpad_parameter @len : i32
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @length_param_d2_masked(%in: memref<16384xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<16384xi32> offset = 0 len = 4124 sizes = [1, 1, 1, 4124] strides = [0, 8248, 0, 1])
          {length_parameter = @len, length_granule = 4 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}
