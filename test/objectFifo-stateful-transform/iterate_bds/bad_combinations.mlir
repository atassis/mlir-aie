//===- bad_combinations.mlir -------------------------------------*- MLIR -*-===//
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
    // expected-error@+1 {{`disable_synchronization` cannot be combined with `iterate_bds`}}
    aie.objectfifo @out (%mem, {%core}, [4, 2]) {iterate_bds, disable_synchronization = true} : !aie.objectfifo<memref<64xi32>>
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
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @in (%shim, {%mem}, 4 : i32) : !aie.objectfifo<memref<64xi32>>
    // expected-error@+1 {{`packet` cannot be combined with `iterate_bds`}}
    aie.objectfifo @out (%mem, {%core}, [4, 2]) {iterate_bds, packet} : !aie.objectfifo<memref<64xi32>>
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
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @in (%shim, {%mem}, 4 : i32) : !aie.objectfifo<memref<64xi32>>
    // expected-error@+1 {{`iter_count` cannot be combined with `iterate_bds`}}
    aie.objectfifo @out (%mem, {%core}, [4, 2]) {iterate_bds, iter_count = 2 : i32} : !aie.objectfifo<memref<64xi32>>
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
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @in (%shim, {%mem}, 4 : i32) : !aie.objectfifo<memref<64xi32>>
    // expected-error@+1 {{`padDimensions` cannot be combined with `iterate_bds`}}
    aie.objectfifo @out (%mem dimensionsToStream [<size = 16, stride = 4>, <size = 4, stride = 1>], {%core}, [4, 2]) {iterate_bds, padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 4>]>} : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
  }
}

// -----

module {
  aie.device(npu2) {
    %c0 = aie.tile(0, 2)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 3)
    // expected-error@+1 {{`init_values` cannot be combined with `iterate_bds`}}
    aie.objectfifo @out (%mem, {%core}, 2 : i32) {iterate_bds} : !aie.objectfifo<memref<4xi32>> = [dense<[0, 1, 2, 3]> : memref<4xi32>, dense<[4, 5, 6, 7]> : memref<4xi32>]
  }
}

// -----

// The link's pool also serves the unflagged side, so its restrictions apply
// there too.
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @in (%shim, {%mem}, 4 : i32) {iterate_bds} : !aie.objectfifo<memref<64xi32>>
    // expected-error@+1 {{`padDimensions` cannot be combined with a link using `iterate_bds`}}
    aie.objectfifo @out (%mem dimensionsToStream [<size = 16, stride = 4>, <size = 4, stride = 1>], {%core}, [4, 2]) {padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 4>]>} : !aie.objectfifo<memref<64xi32>>
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
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @in (%shim, {%mem}, 4 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out (%mem, {%core}, [4, 2]) {iterate_bds} : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    // expected-error@+1 {{cannot be combined with a link using `iterate_bds`}}
    aie.objectfifo.allocate @out (%mem)
    aie.core(%core) {
      %e = aie.objectfifo.acquire @out (Consume, 1) : memref<64xi32>
      aie.objectfifo.release @out (Consume, 1)
      aie.end
    }
  }
}

// -----

// A channel reset restores the locks but not the BDs' iteration state.
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @in (%shim, {%mem}, 4 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out (%mem, {%core}, [4, 2]) {iterate_bds} : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%core) {
      %e = aie.objectfifo.acquire @out (Consume, 1) : memref<64xi32>
      aie.objectfifo.release @out (Consume, 1)
      aie.end
    }
    aie.runtime_sequence() {
      // expected-error@+1 {{cannot re-arm a fifo whose pool uses iterate_bds}}
      aiex.dma_channel_reset_for(@in)
    }
  }
}
