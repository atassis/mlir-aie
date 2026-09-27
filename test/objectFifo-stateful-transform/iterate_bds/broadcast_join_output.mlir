//===- broadcast_join_output.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// The SPAN shape: a join's output objectfifo is ALSO broadcast to two core
// consumers. iterate_bds must lower the MemTile side to one ring buffer and
// iterating BDs, and must NOT leak onto the broadcast consumers' per-tile
// pools -- each core still gets its own chain of ordinary per-object buffers.

// CHECK: %[[BUF:.*]] = aie.buffer(%{{.*}}tile_0_1) {sym_name = "cat_buff"} : memref<2048xi8>
// CHECK-NOT: aie.buffer(%{{.*}}tile_0_1)
// CHECK-LABEL: aie.memtile_dma(%{{.*}}tile_0_1) {
// CHECK: aie.dma_bd(%[[BUF]] : memref<2048xi8> offset = 0 len = 256) {{.*}}iteration = #aie.bd_iteration<size = 4, stride = 512, current = 0>
// CHECK: aie.dma_bd(%[[BUF]] : memref<2048xi8> offset = 256 len = 256) {{.*}}iteration = #aie.bd_iteration<size = 4, stride = 512, current = 0>

// CHECK-LABEL: aie.mem(%{{.*}}tile_1_2) {
// CHECK-NOT: iteration
// CHECK: aie.dma_bd(%{{.*}}cat_0_cons_buff_0 : memref<512xi8> offset = 0 len = 512)
// CHECK-NOT: iteration
// CHECK: aie.dma_bd(%{{.*}}cat_0_cons_buff_1 : memref<512xi8> offset = 0 len = 512)
// CHECK-NOT: iteration
// CHECK: aie.dma_bd(%{{.*}}cat_0_cons_buff_2 : memref<512xi8> offset = 0 len = 512)
// CHECK-NOT: iteration
// CHECK: aie.dma_bd(%{{.*}}cat_0_cons_buff_3 : memref<512xi8> offset = 0 len = 512)

// CHECK-LABEL: aie.mem(%{{.*}}tile_1_3) {
// CHECK-NOT: iteration
// CHECK: aie.dma_bd(%{{.*}}cat_1_cons_buff_0 : memref<512xi8> offset = 0 len = 512)
// CHECK-NOT: iteration
// CHECK: aie.dma_bd(%{{.*}}cat_1_cons_buff_1 : memref<512xi8> offset = 0 len = 512)
// CHECK-NOT: iteration
// CHECK: aie.dma_bd(%{{.*}}cat_1_cons_buff_2 : memref<512xi8> offset = 0 len = 512)
// CHECK-NOT: iteration
// CHECK: aie.dma_bd(%{{.*}}cat_1_cons_buff_3 : memref<512xi8> offset = 0 len = 512)

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %p0 = aie.tile(0, 2)
    %p1 = aie.tile(0, 3)
    %cons0 = aie.tile(1, 2)
    %cons1 = aie.tile(1, 3)
    aie.objectfifo @in0 (%p0, {%mem}, 2 : i32) : !aie.objectfifo<memref<256xi8>>
    aie.objectfifo @in1 (%p1, {%mem}, 2 : i32) : !aie.objectfifo<memref<256xi8>>
    aie.objectfifo @cat (%mem, {%cons0, %cons1}, [4, 4, 4]) {iterate_bds} : !aie.objectfifo<memref<512xi8>>
    aie.objectfifo.link [@in0, @in1] -> [@cat] ([0, 256] [])
    aie.core(%p0) {
      %e = aie.objectfifo.acquire @in0 (Produce, 1) : memref<256xi8>
      aie.objectfifo.release @in0 (Produce, 1)
      aie.end
    }
    aie.core(%p1) {
      %e = aie.objectfifo.acquire @in1 (Produce, 1) : memref<256xi8>
      aie.objectfifo.release @in1 (Produce, 1)
      aie.end
    }
    aie.core(%cons0) {
      %e = aie.objectfifo.acquire @cat (Consume, 1) : memref<512xi8>
      aie.objectfifo.release @cat (Consume, 1)
      aie.end
    }
    aie.core(%cons1) {
      %e = aie.objectfifo.acquire @cat (Consume, 1) : memref<512xi8>
      aie.objectfifo.release @cat (Consume, 1)
      aie.end
    }
  }
}
