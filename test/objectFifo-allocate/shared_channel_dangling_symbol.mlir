// RUN: not aie-opt --aie-objectfifo-allocate %s 2>&1 | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Hand-written IR (ObjectFifo split always names a real leader) with a
// sharesChannel pointing at a symbol that never gets a channel.

module @shared_channel_dangling_symbol {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %memtile = aie.tile(0, 1)

    aie.route_endpoint @scales_shim(%shim) DMA {fifoName = "scales", sharesChannel = @no_such_leader}

    aie.objectfifo.pool @scales_pool(%memtile) {
      depth = 2 : i32
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @scales_dma(%memtile) fills @scales_pool {segments = [@s0]}

    aie.route from @scales_shim to [@scales_dma]
  }
}

// CHECK: error: 'aie.route_endpoint' op sharesChannel names '@no_such_leader', which has no assigned channel
