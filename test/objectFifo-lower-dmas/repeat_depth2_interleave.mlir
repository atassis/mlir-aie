// RUN: aie-opt --aie-objectfifo-lower-dmas %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A depth-2 pool replayed with repeat_count must ALTERNATE its buffers, not drain one of them
// `repeat` times and then the other. Draining block-wise leaves the two buffers never in flight
// together, so the depth buys no ping-pong and the consumer stalls on every object.
// At depth 1 both orders are identical, which is why every other repeat_count test passes either
// way -- this one is depth 2 on purpose.

module @repeat_depth2 {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)

    %b0 = aie.buffer(%tile12) {sym_name = "b0"} : memref<16xi32>
    %b1 = aie.buffer(%tile12) {sym_name = "b1"} : memref<16xi32>
    %free = aie.lock(%tile12) {init = 2 : i32, sym_name = "free"}
    %full = aie.lock(%tile12) {init = 0 : i32, sym_name = "full"}

    aie.objectfifo.pool @prod_pool(%tile12) {
      depth = 2 : i32, buffers = [@b0, @b1], repeat_count = 3 : i32
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {consumeLock = @full, offset = 0 : i32, produceLock = @free, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @prod_dma(%tile12) drains @prod_pool {
      channelIndex = 0 : i32
    }
  }
}

// Six descriptors: repeat 3 over a depth-2 pool. The buffers must interleave b0,b1,b0,b1,b0,b1.
// CHECK-LABEL: @repeat_depth2
// CHECK:     aie.dma_bd(%b0
// CHECK:     aie.dma_bd(%b1
// CHECK:     aie.dma_bd(%b0
// CHECK:     aie.dma_bd(%b1
// CHECK:     aie.dma_bd(%b0
// CHECK:     aie.dma_bd(%b1
