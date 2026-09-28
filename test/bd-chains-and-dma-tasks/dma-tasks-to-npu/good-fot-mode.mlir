//===- good-fot-mode.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// fot_mode lowers to the DMA_S2MM_0_Ctrl.FoT_Mode maskwrite32 (bits 17:16) a
// design used to write by hand at the register's raw address.

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4xi32>) {
      // Shim S2MM channel 0 CTRL register is at local offset 0x1D200
      // (= 119296); FoT_Mode is bits 17:16 (mask 0x30000 = 196608),
      // counts_with_task_tokens = 2 (value 0x20000 = 131072).
      // CHECK-DAG: %[[ADDR:.*]] = arith.constant 119296 : i32
      // CHECK-DAG: %[[VAL:.*]] = arith.constant 131072 : i32
      // CHECK-DAG: %[[MASK:.*]] = arith.constant 196608 : i32
      // CHECK: aiex.npu.maskwrite32(%[[ADDR]], %[[VAL]], %[[MASK]])
      %t = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4) {bd_id = 0 : i32}
        aie.end
      } {fot_mode = 2 : i32}
      aiex.dma_start_task(%t)
    }
  }
}
