//===- split_iterate.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// A 3-way split of depth 6: the input channel chains one iterating BD per
// segment, and each output channel runs one iterating BD over its segment.

// CHECK: %[[BUF:.*]] = aie.buffer(%{{.*}}tile_0_1) {sym_name = "in_cons_buff"} : memref<2304xi32>
// CHECK-NOT: aie.buffer(%{{.*}}tile_0_1)
// CHECK-LABEL: aie.memtile_dma(%{{.*}}tile_0_1) {
// CHECK: aie.dma_start(S2MM, 0, ^[[IN0:bb[0-9]+]], ^[[NEXT:bb[0-9]+]])
// CHECK-NEXT: ^[[IN0]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P0:[a-z0-9_]+]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2304xi32> offset = 0 len = 128) {iteration = #aie.bd_iteration<size = 6, stride = 384, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C0:[a-z0-9_]+]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[IN1:bb[0-9]+]]
// CHECK-NEXT: ^[[IN1]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P1:[a-z0-9_]+]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2304xi32> offset = 128 len = 128) {iteration = #aie.bd_iteration<size = 6, stride = 384, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C1:[a-z0-9_]+]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[IN2:bb[0-9]+]]
// CHECK-NEXT: ^[[IN2]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P2:[a-z0-9_]+]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2304xi32> offset = 256 len = 128) {iteration = #aie.bd_iteration<size = 6, stride = 384, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C2:[a-z0-9_]+]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[IN0]]
// CHECK-NEXT: ^[[NEXT]]:
// CHECK-NEXT: aie.dma_start(MM2S, 0, ^[[OUT0:bb[0-9]+]], ^[[NEXT0:bb[0-9]+]])
// CHECK-NEXT: ^[[OUT0]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C0]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2304xi32> offset = 0 len = 128) {iteration = #aie.bd_iteration<size = 6, stride = 384, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P0]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[OUT0]]
// CHECK-NEXT: ^[[NEXT0]]:
// CHECK-NEXT: aie.dma_start(MM2S, 1, ^[[OUT1:bb[0-9]+]], ^[[NEXT1:bb[0-9]+]])
// CHECK-NEXT: ^[[OUT1]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C1]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2304xi32> offset = 128 len = 128) {iteration = #aie.bd_iteration<size = 6, stride = 384, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P1]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[OUT1]]
// CHECK-NEXT: ^[[NEXT1]]:
// CHECK-NEXT: aie.dma_start(MM2S, 2, ^[[OUT2:bb[0-9]+]], ^[[END:bb[0-9]+]])
// CHECK-NEXT: ^[[OUT2]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C2]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2304xi32> offset = 256 len = 128) {iteration = #aie.bd_iteration<size = 6, stride = 384, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P2]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[OUT2]]
// CHECK-NEXT: ^[[END]]:
// CHECK-NEXT: aie.end

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %c0 = aie.tile(0, 2)
    %c1 = aie.tile(0, 3)
    %c2 = aie.tile(0, 4)
    aie.objectfifo @in (%shim, {%mem}, 6 : i32) {iterate_bds} : !aie.objectfifo<memref<384xi32>>
    aie.objectfifo @o0 (%mem, {%c0}, [6, 2]) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @o1 (%mem, {%c1}, [6, 2]) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @o2 (%mem, {%c2}, [6, 2]) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo.link [@in] -> [@o0, @o1, @o2] ([] [0, 128, 256])
    aie.core(%c0) {
      %e = aie.objectfifo.acquire @o0 (Consume, 1) : memref<128xi32>
      aie.objectfifo.release @o0 (Consume, 1)
      aie.end
    }
    aie.core(%c1) {
      %e = aie.objectfifo.acquire @o1 (Consume, 1) : memref<128xi32>
      aie.objectfifo.release @o1 (Consume, 1)
      aie.end
    }
    aie.core(%c2) {
      %e = aie.objectfifo.acquire @o2 (Consume, 1) : memref<128xi32>
      aie.objectfifo.release @o2 (Consume, 1)
      aie.end
    }
  }
}
