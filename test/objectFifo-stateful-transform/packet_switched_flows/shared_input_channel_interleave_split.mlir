//===- shared_input_channel_interleave_split.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Split renames every symbol a shared_input_channel join's fifos are known
// by, including inside a runtime-sequence dma_configure_task_for's
// `interleave` list (a generic property of aie.symbolTable::replaceAllSymbolUses,
// not new plumbing): @payload -> @payload_prod_dma, @scale -> @scale_prod_dma.

// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem  = aie.tile(0, 1)
    %core = aie.tile(0, 2)

    aie.objectfifo @payload (%shim, {%mem}, 2 : i32)
      {packet, packet_id = 0 : i8} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @scale (%shim, {%mem}, 2 : i32)
      {packet, packet_id = 1 : i8} : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @joined (%mem, {%core}, 4 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo.link [@payload, @scale] -> [@joined] ([0, 16][]) {shared_input_channel}

    aie.core(%core) { aie.end }

    aie.runtime_sequence(%w : memref<1024xi32>) {
      // CHECK: %{{.*}} = aiex.dma_configure_task_for @payload_prod_dma
      // CHECK: } {interleave = [@scale_prod_dma], issue_token = true, repeat_count = 7 : i32}
      %ti = aiex.dma_configure_task_for @payload {
          aie.dma_bd(%w : memref<1024xi32> offset = 0 len = 16
                     sizes = [8, 1, 1, 16] strides = [16, 0, 0, 1])
          aie.next_bd ^s
        ^s:
          aie.dma_bd(%w : memref<1024xi32> offset = 512 len = 4
                     sizes = [8, 1, 1, 4] strides = [4, 0, 0, 1])
          aie.end
      } {repeat_count = 7 : i32, issue_token = true, interleave = [@scale]}
      aiex.dma_start_task(%ti)
      aiex.dma_await_task(%ti)
    }
  }
}
