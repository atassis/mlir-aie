//===- length-parameter-memtile.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// length_parameter on a MemTile BD used to be rejected outright ("only
// supported on shim NOC tile BDs") for a field-width reason, not a device
// one (rf-memtile-bd-runtime-length). It now lowers the same way the shim
// path does, at the MemTile BD's own address.

// RUN: aie-opt --pass-pipeline='any(aie-lower-scratchpad-parameters,aie.device(aie-dma-tasks-to-npu))' %s | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence @length_param_memtile
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = {{[0-9]+}} : ui32, func_arg = 4 : ui32, state_table_idx = 0 : ui8}
module {
  aiex.scratchpad_parameter @len : i32
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %buf = aie.buffer(%t) {address = 0 : i32} : memref<64xi32>
    aie.runtime_sequence @length_param_memtile(%unused: memref<4xi32>) {
      %tk = aiex.dma_configure_task(%t, S2MM, 0) {
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 16) {bd_id = 0 : i32, length_granule = 4 : i32, length_parameter = @len}
        aie.end
      }
      aiex.dma_start_task(%tk)
    }
  }
}
