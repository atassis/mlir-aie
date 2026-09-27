//===- join_4x22_memtile.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// A 4-way join of depth 22 on a MemTile: one buffer holds the ring, each input
// channel runs one BD iterating over the 22 slots, and the output channel
// chains one iterating BD per segment.

// CHECK: %[[BUF:.*]] = aie.buffer(%{{.*}}tile_0_1) {sym_name = "cat_buff"} : memref<253440xi8>
// CHECK-NOT: aie.buffer(%{{.*}}tile_0_1)
// CHECK-LABEL: aie.memtile_dma(%{{.*}}tile_0_1) {
// CHECK: aie.dma_start(S2MM, 0, ^[[IN0:bb[0-9]+]], ^[[S2MM_NEXT0:bb[0-9]+]])
// CHECK-NEXT: ^[[IN0]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P0:[a-z0-9_]+]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<253440xi8> offset = 0 len = 2304) {iteration = #aie.bd_iteration<size = 22, stride = 11520, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C0:[a-z0-9_]+]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[IN0]]
// CHECK-NEXT: ^[[S2MM_NEXT0]]:
// CHECK-NEXT: aie.dma_start(S2MM, 1, ^[[IN1:bb[0-9]+]], ^[[S2MM_NEXT1:bb[0-9]+]])
// CHECK-NEXT: ^[[IN1]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P1:[a-z0-9_]+]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<253440xi8> offset = 2304 len = 2304) {iteration = #aie.bd_iteration<size = 22, stride = 11520, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C1:[a-z0-9_]+]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[IN1]]
// CHECK-NEXT: ^[[S2MM_NEXT1]]:
// CHECK-NEXT: aie.dma_start(S2MM, 2, ^[[IN2:bb[0-9]+]], ^[[S2MM_NEXT2:bb[0-9]+]])
// CHECK-NEXT: ^[[IN2]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P2:[a-z0-9_]+]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<253440xi8> offset = 4608 len = 2304) {iteration = #aie.bd_iteration<size = 22, stride = 11520, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C2:[a-z0-9_]+]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[IN2]]
// CHECK-NEXT: ^[[S2MM_NEXT2]]:
// CHECK-NEXT: aie.dma_start(S2MM, 3, ^[[IN3:bb[0-9]+]], ^[[S2MM_NEXT3:bb[0-9]+]])
// CHECK-NEXT: ^[[IN3]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P3:[a-z0-9_]+]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<253440xi8> offset = 6912 len = 4608) {iteration = #aie.bd_iteration<size = 22, stride = 11520, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C3:[a-z0-9_]+]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[IN3]]
// CHECK-NEXT: ^[[S2MM_NEXT3]]:
// CHECK-NEXT: aie.dma_start(MM2S, 0, ^[[OUT0:bb[0-9]+]], ^[[END:bb[0-9]+]])
// CHECK-NEXT: ^[[OUT0]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C0]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<253440xi8> offset = 0 len = 2304) {iteration = #aie.bd_iteration<size = 22, stride = 11520, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P0]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[OUT1:bb[0-9]+]]
// CHECK-NEXT: ^[[OUT1]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C1]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<253440xi8> offset = 2304 len = 2304) {iteration = #aie.bd_iteration<size = 22, stride = 11520, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P1]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[OUT2:bb[0-9]+]]
// CHECK-NEXT: ^[[OUT2]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C2]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<253440xi8> offset = 4608 len = 2304) {iteration = #aie.bd_iteration<size = 22, stride = 11520, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P2]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[OUT3:bb[0-9]+]]
// CHECK-NEXT: ^[[OUT3]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C3]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<253440xi8> offset = 6912 len = 4608) {iteration = #aie.bd_iteration<size = 22, stride = 11520, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P3]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[OUT0]]
// CHECK-NEXT: ^[[END]]:
// CHECK-NEXT: aie.end

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %c0 = aie.tile(0, 2)
    %c1 = aie.tile(0, 3)
    %c2 = aie.tile(0, 4)
    %c3 = aie.tile(0, 5)
    %cons = aie.tile(1, 2)
    aie.objectfifo @in0 (%c0, {%mem}, 2 : i32) : !aie.objectfifo<memref<2304xi8>>
    aie.objectfifo @in1 (%c1, {%mem}, 2 : i32) : !aie.objectfifo<memref<2304xi8>>
    aie.objectfifo @in2 (%c2, {%mem}, 2 : i32) : !aie.objectfifo<memref<2304xi8>>
    aie.objectfifo @in3 (%c3, {%mem}, 2 : i32) : !aie.objectfifo<memref<4608xi8>>
    aie.objectfifo @cat (%mem, {%cons}, [22, 2]) {iterate_bds} : !aie.objectfifo<memref<11520xi8>>
    aie.objectfifo.link [@in0, @in1, @in2, @in3] -> [@cat] ([0, 2304, 4608, 6912] [])
    aie.core(%c0) {
      %e = aie.objectfifo.acquire @in0 (Produce, 1) : memref<2304xi8>
      aie.objectfifo.release @in0 (Produce, 1)
      aie.end
    }
    aie.core(%c1) {
      %e = aie.objectfifo.acquire @in1 (Produce, 1) : memref<2304xi8>
      aie.objectfifo.release @in1 (Produce, 1)
      aie.end
    }
    aie.core(%c2) {
      %e = aie.objectfifo.acquire @in2 (Produce, 1) : memref<2304xi8>
      aie.objectfifo.release @in2 (Produce, 1)
      aie.end
    }
    aie.core(%c3) {
      %e = aie.objectfifo.acquire @in3 (Produce, 1) : memref<4608xi8>
      aie.objectfifo.release @in3 (Produce, 1)
      aie.end
    }
    aie.core(%cons) {
      %e = aie.objectfifo.acquire @cat (Consume, 1) : memref<11520xi8>
      aie.objectfifo.release @cat (Consume, 1)
      aie.end
    }
  }
}
