//===- bad_repeat_count.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file -verify-diagnostics --aie-objectFifo-stateful-transform %s

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @in (%shim, {%mem}, 4 : i32) : !aie.objectfifo<memref<64xi32>>
    // expected-error@+1 {{`repeat_count` cannot be combined with `iterate_bds`}}
    aie.objectfifo @out (%mem, {%core}, [4, 2]) {iterate_bds, repeat_count = 2 : i32} : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
  }
}

// -----

// The link's pool takes its repeat count from any participant.
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @in (%shim, {%mem}, 4 : i32) {iterate_bds} : !aie.objectfifo<memref<64xi32>>
    // expected-error@+1 {{`repeat_count` cannot be combined with a link using `iterate_bds`}}
    aie.objectfifo @out (%mem, {%core}, [4, 2]) {repeat_count = 2 : i32} : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%core) {
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
    // expected-error@+1 {{iterate_bds cannot be combined with 'repeatCount'}}
    aie.objectfifo.pool @p(%mem) {depth = 4 : i32, iterateBds, repeatCount = 2 : i32} : memref<64xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 64 : i32}
    }
  }
}
