//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-materialize-bd-chains --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// A MemTile channel only reaches its own parity's BD bank: even channels
// 0-23, odd 24-47. A task on channel 1 (odd) must get an id from 24-47, not
// from the pool nextBdId(0) would hand a shim/core task (K045).

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)

    aie.bd_chain @c(%arg0: memref<8xi16>) {
      aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8)
      aie.end
    }

    aie.runtime_sequence(%arg0: memref<8xi16>) {
      %t = aiex.dma_start_bd_chain @c(%arg0) : (memref<8xi16>)
                                    on (%tile_0_1, S2MM, 1)
      // CHECK: aiex.dma_configure_task(%{{.*}}tile_0_1, S2MM, 1) {
      // CHECK:   aie.dma_bd(%arg0 : memref<8xi16> offset = {{.*}} len = {{.*}}) {bd_id = 24 : i32}
      aiex.dma_await_task(%t)
    }
  }
}
