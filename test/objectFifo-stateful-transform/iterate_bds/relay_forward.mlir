//===- relay_forward.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// A MemTile forward of depth 8: one BD per side, each iterating over the ring.
// The flag on either fifo of the link selects the link's pool.

// CHECK: %[[BUF:.*]] = aie.buffer(%{{.*}}tile_0_1) {sym_name = "in_cons_buff"} : memref<2048xi32>
// CHECK-NOT: aie.buffer(%{{.*}}tile_0_1)
// CHECK-DAG: %[[P:.*]] = aie.lock(%{{.*}}tile_0_1) {init = 8 : i32, sym_name = "in_cons_prod_lock_0"}
// CHECK-DAG: %[[C:.*]] = aie.lock(%{{.*}}tile_0_1) {init = 0 : i32, sym_name = "in_cons_cons_lock_0"}
// CHECK-LABEL: aie.memtile_dma(%{{.*}}tile_0_1) {
// CHECK: aie.dma_start(S2MM, 0, ^[[IN:bb[0-9]+]], ^[[NEXT:bb[0-9]+]])
// CHECK-NEXT: ^[[IN]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2048xi32> offset = 0 len = 256) {iteration = #aie.bd_iteration<size = 8, stride = 256, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[IN]]
// CHECK-NEXT: ^[[NEXT]]:
// CHECK-NEXT: aie.dma_start(MM2S, 0, ^[[OUT:bb[0-9]+]], ^[[END:bb[0-9]+]])
// CHECK-NEXT: ^[[OUT]]:
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[C]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[BUF]] : memref<2048xi32> offset = 0 len = 256) {iteration = #aie.bd_iteration<size = 8, stride = 256, current = 0>}
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[P]], Release, %{{.*}})
// CHECK-NEXT: aie.next_bd ^[[OUT]]
// CHECK-NEXT: ^[[END]]:
// CHECK-NEXT: aie.end

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @in (%shim, {%mem}, 8 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @out (%mem, {%core}, [8, 2]) {iterate_bds} : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%core) {
      %e = aie.objectfifo.acquire @out (Consume, 1) : memref<256xi32>
      aie.objectfifo.release @out (Consume, 1)
      aie.end
    }
  }
}
