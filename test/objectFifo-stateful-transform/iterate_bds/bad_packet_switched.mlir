//===- bad_packet_switched.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -verify-diagnostics --aie-objectFifo-stateful-transform="packet-sw-objFifos=true" %s

// Packet-switching every fifo puts a packet header on the MemTile's BDs.
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @in (%shim, {%mem}, 4 : i32) : !aie.objectfifo<memref<64xi32>>
    // expected-error@+1 {{packet header cannot be combined with a pool using iterate_bds}}
    aie.objectfifo @out (%mem, {%core}, [4, 2]) {iterate_bds} : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%core) {
      %e = aie.objectfifo.acquire @out (Consume, 1) : memref<64xi32>
      aie.objectfifo.release @out (Consume, 1)
      aie.end
    }
  }
}
