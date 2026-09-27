//===- bad_core_access.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file -verify-diagnostics --aie-objectFifo-stateful-transform %s

// A core-to-core fifo has no MemTile pool for iterate_bds to apply to.
module {
  aie.device(npu2) {
    %c0 = aie.tile(0, 2)
    %c1 = aie.tile(0, 4)
    // expected-error@+1 {{iterate_bds applies only to a link through a MemTile}}
    aie.objectfifo @of (%c0, {%c1}, 4 : i32) {iterate_bds} : !aie.objectfifo<memref<64xi32>>
    aie.core(%c0) {
      %e = aie.objectfifo.acquire @of (Produce, 1) : memref<64xi32>
      aie.objectfifo.release @of (Produce, 1)
      aie.end
    }
    aie.core(%c1) {
      %e = aie.objectfifo.acquire @of (Consume, 1) : memref<64xi32>
      aie.objectfifo.release @of (Consume, 1)
      aie.end
    }
  }
}

// -----

// A link through a compute tile keeps its pool where that tile's core could
// reach it.
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %c0 = aie.tile(0, 2)
    %c1 = aie.tile(0, 3)
    aie.objectfifo @in (%shim, {%c0}, 4 : i32) : !aie.objectfifo<memref<64xi32>>
    // expected-error@+1 {{iterate_bds applies only to a link through a MemTile}}
    aie.objectfifo @out (%c0, {%c1}, 4 : i32) {iterate_bds} : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%c1) {
      %e = aie.objectfifo.acquire @out (Consume, 1) : memref<64xi32>
      aie.objectfifo.release @out (Consume, 1)
      aie.end
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo.pool @p(%mem) {depth = 4 : i32, iterateBds} : memref<64xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 64 : i32}
    }
    // expected-error@+1 {{cannot access pool 'p': its iterate_bds objects are addressed only by a DMA}}
    aie.objectfifo.core_endpoint @reader(%core) drains @p
  }
}
