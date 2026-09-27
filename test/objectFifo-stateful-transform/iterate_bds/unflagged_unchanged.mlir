//===- unflagged_unchanged.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file -verify-diagnostics --aie-objectFifo-stateful-transform %s | FileCheck %s

// Without iterate_bds a join keeps one BD per (object, segment): depth 3 gives
// 3 BDs per input channel and 12 on the output channel, and depth 22 overflows
// the MemTile's 48 BDs.

// CHECK-DAG: aie.buffer(%{{.*}}tile_0_1) {sym_name = "cat_buff_0"} : memref<11520xi8>
// CHECK-DAG: aie.buffer(%{{.*}}tile_0_1) {sym_name = "cat_buff_1"} : memref<11520xi8>
// CHECK-DAG: aie.buffer(%{{.*}}tile_0_1) {sym_name = "cat_buff_2"} : memref<11520xi8>
// CHECK-LABEL: aie.memtile_dma(%{{.*}}tile_0_1) {
// CHECK-COUNT-24: aie.dma_bd(%cat_buff_{{[0-2]}} : memref<11520xi8> offset = {{[0-9]+}} len = {{[0-9]+}}){{$}}
// CHECK-NOT: aie.dma_bd
// CHECK: aie.end

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
    aie.objectfifo @cat (%mem, {%cons}, [3, 2]) : !aie.objectfifo<memref<11520xi8>>
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

// -----

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %c0 = aie.tile(0, 2)
    %c1 = aie.tile(0, 3)
    %c2 = aie.tile(0, 4)
    %c3 = aie.tile(0, 5)
    %cons = aie.tile(1, 2)
    // expected-error@+1 {{'aie.memtile_dma' op has more than 48 blocks}}
    aie.objectfifo @in0 (%c0, {%mem}, 2 : i32) : !aie.objectfifo<memref<2304xi8>>
    aie.objectfifo @in1 (%c1, {%mem}, 2 : i32) : !aie.objectfifo<memref<2304xi8>>
    // expected-note@+1 {{no space for this BD}}
    aie.objectfifo @in2 (%c2, {%mem}, 2 : i32) : !aie.objectfifo<memref<2304xi8>>
    aie.objectfifo @in3 (%c3, {%mem}, 2 : i32) : !aie.objectfifo<memref<4608xi8>>
    aie.objectfifo @cat (%mem, {%cons}, [22, 2]) : !aie.objectfifo<memref<11520xi8>>
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
