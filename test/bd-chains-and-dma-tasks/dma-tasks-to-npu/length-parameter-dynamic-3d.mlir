//===- length-parameter-dynamic-3d.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The static-path counterpart is scratchpad_length_parameter_3d.mlir (in
// test/dialect/AIEX); this is the same length_parameter BD shape, but with a
// runtime bd_id (the dynamic free-list pool), forcing buildShimBdWords' SSA
// path (AIEDMATasksToNPU.cpp's rewriteSingleBDDynamic).
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --pass-pipeline='any(aie-lower-scratchpad-parameters,aie.device(aie-dma-tasks-to-npu))' %s | aie-opt --canonicalize | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence @length_param_3d_dynamic
// CHECK: aiex.npu.blockwrite_values
// -- word[3]: d0_size = 64 << 20 -- not linear.
// CHECK-SAME: 67108864
// -- word[5]: axcache (2, bits [27:24]) | d2_stride. d2_stride survives as
// -- 2047, the biased-by-1 encoding of stride 2048 (see
// -- AIENormalizeDmaBdDimsPass).
// CHECK-SAME: 33556479
module {
  aiex.scratchpad_parameter @extra_blocks : i32
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @length_param_3d_dynamic(%in: memref<16384xi32>) {
      %bd = aiex.dma_bd_pool_pop(0, 0) : i32
      %tk = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%in : memref<16384xi32> offset = 512 len = 512 sizes = [1, 1, 8, 64] strides = [0, 2048, 64, 1]) bd_id_val %bd : i32
          {length_granule = 512 : i32, length_parameter = @extra_blocks}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
      aiex.dma_bd_pool_push(0, 0) bd_id %bd : i32
    }
  }
}
