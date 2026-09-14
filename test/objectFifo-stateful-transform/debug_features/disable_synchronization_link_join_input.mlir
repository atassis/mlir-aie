//===- disable_synchronization_link_join_input.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// A join's shared pool is always owned by outs[0] (AIEObjectFifoSplit.cpp
// createLinkPools), so every input is a non-owner participant by
// construction. disable_synchronization set on either input must still reach
// the shared pool: before the OR-across-participants fix, createPool read
// only the owner's value, so BOTH of these settings were silently dropped
// and the pool got locks anyway.

// CHECK: module @disable_sync_join {
// CHECK:   aie.device(xcve2302) {
// CHECK-DAG:     %{{.*}}tile_1_1 = aie.tile(1, 1)
// The shared pool's buffer at the link tile carries no lock -- neither a
// producer nor a consumer lock -- because disable_synchronization reached it
// from a non-owner input.
// CHECK-DAG:     %[[POOL_BUF:.*]] = aie.buffer(%{{.*}}tile_1_1) {sym_name = "of2_buff_0"} : memref<8xi32>
// CHECK-NOT:     aie.lock(%{{.*}}tile_1_1)
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%{{.*}}tile_1_1) {
// The shared pool's own DMA never acquires or releases a lock either.
// CHECK-NOT:       aie.use_lock
// CHECK:     }
// CHECK:   }
// CHECK: }

module @disable_sync_join {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)
    %tile11 = aie.tile(1, 1)
    %tile10 = aie.tile(1, 0)

    aie.objectfifo @of0 (%tile12, {%tile11}, 2 : i32) : !aie.objectfifo<memref<2x2xi32>>
    // disable_synchronization is set on @of1, an INPUT -- never the join
    // owner (always outs[0]) -- to isolate the non-owner-participant case.
    aie.objectfifo @of1 (%tile23, {%tile11}, 2 : i32) { disable_synchronization = true } : !aie.objectfifo<memref<2x2xi32>>
    aie.objectfifo @of2 (%tile11, {%tile10}, 2 : i32) : !aie.objectfifo<memref<8xi32>>

    aie.objectfifo.link [@of0, @of1] -> [@of2] ([0, 4] [])
 }
}
