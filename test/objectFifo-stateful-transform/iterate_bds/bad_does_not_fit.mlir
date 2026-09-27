//===- bad_does_not_fit.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file -verify-diagnostics --aie-objectFifo-stateful-transform %s

// 40 objects of 16 KiB exceed a 512 KiB MemTile. Without iterate_bds they
// would spill to a neighbouring MemTile; one ring buffer cannot.
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(1, 1)
    %core = aie.tile(1, 2)
    aie.objectfifo @in (%shim, {%mem}, 40 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @out (%mem, {%core}, [40, 2]) {iterate_bds} : !aie.objectfifo<memref<4096xi32>>
    // expected-error@+1 {{'aie.objectfifo.pool' op iterate_bds needs one 655360-byte buffer on its MemTile, which has 524288 bytes free}}
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%core) {
      %e = aie.objectfifo.acquire @out (Consume, 1) : memref<4096xi32>
      aie.objectfifo.release @out (Consume, 1)
      aie.end
    }
  }
}

// -----

// The BD iteration counter wraps at 64.
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @in (%shim, {%mem}, 65 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out (%mem, {%core}, [65, 2]) {iterate_bds} : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{'aie.objectfifo.pool' op iterate_bds depth 65 exceeds the 64 steps of a BD's iteration}}
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%core) {
      %e = aie.objectfifo.acquire @out (Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out (Consume, 1)
      aie.end
    }
  }
}
