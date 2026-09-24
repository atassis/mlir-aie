//===- scratchpad_length_parameter.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A length_parameter lowers to an aiex.npu.update_from_scratchpad targeting
// the BD's word-0 (buffer_length) register, func=mul, func_arg = length_granule
// * element size / 4 (words). Emitted after the BD blockwrite and address
// patch, before the queue push -- mirrors offset_parameter's address-register
// update.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --pass-pipeline='any(aie-lower-scratchpad-parameters,aie.device(aie-dma-to-npu))' %s | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence @length_param
// CHECK: aiex.npu.blockwrite{{.*}}{address = [[BASE:[0-9]+]] : ui32}
// CHECK: aiex.npu.address_patch
// CHECK: aiex.npu.update_from_scratchpad<mul> {address = [[BASE]] : ui32, func_arg = 4 : ui32, state_table_idx = 0 : ui8}
module {
  aiex.scratchpad_parameter @len : i32
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @length_param(%in: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,16][0,0,0,1])
        {id = 0 : i64, metadata = @a, length_parameter = @len, length_granule = 4 : i32} : memref<64xi32>
    }
  }
}
