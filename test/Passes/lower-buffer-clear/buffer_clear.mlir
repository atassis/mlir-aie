//===- buffer_clear.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file --aie-lower-buffer-clear %s | FileCheck %s --implicit-check-not=aiex.buffer_clear

// Each aiex.buffer_clear lowers to a single aiex.npu.blockwrite of `length`
// literal zero words to `address`, targeting the tile via the column/row
// attributes (address is a tile-local data-memory offset, so no absolute-address
// math is needed). Two calls with the same `length` (8 words) share one
// "blockwrite_data_<n>" zero-data global -- only ONE memref.global is emitted for
// both, even though each call site gets its own memref.get_global +
// npu.blockwrite (the dedup only collapses the IR-level constant, not the
// per-call instruction: each executed npu.blockwrite still carries its own
// copy of the 8 words in the runtime instruction stream). A third call with a
// different length (4 words) gets a second, distinct global.
module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_1 = aie.tile(0, 1)

    // CHECK: memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<0>
    // CHECK: memref.global "private" constant @blockwrite_data_1 : memref<4xi32> = dense<0>
    aie.runtime_sequence() {
      // Core tile (0, 2), 8 words at offset 0.
      // CHECK: %[[G0:.*]] = memref.get_global @blockwrite_data_0 : memref<8xi32>
      // CHECK: aiex.npu.blockwrite(%[[G0]]) {address = 0 : ui32, column = 0 : i32, row = 2 : i32} : memref<8xi32>
      aiex.buffer_clear(%tile_0_2, 0, 8)

      // Same core tile, same length (8 words), different offset: reuses
      // @blockwrite_data_0 rather than emitting a second identical global.
      // CHECK: %[[G1:.*]] = memref.get_global @blockwrite_data_0 : memref<8xi32>
      // CHECK: aiex.npu.blockwrite(%[[G1]]) {address = 32 : ui32, column = 0 : i32, row = 2 : i32} : memref<8xi32>
      aiex.buffer_clear(%tile_0_2, 32, 8)

      // Mem tile (0, 1), 4 words: a different length gets its own global.
      // CHECK: %[[G2:.*]] = memref.get_global @blockwrite_data_1 : memref<4xi32>
      // CHECK: aiex.npu.blockwrite(%[[G2]]) {address = 4 : ui32, column = 0 : i32, row = 1 : i32} : memref<4xi32>
      aiex.buffer_clear(%tile_0_1, 4, 4)
    }
  }
}

// -----

// The local data memory sizes used by the verifier (not exercised by this
// lowering test directly, see buffer_clear_invalid.mlir) come from
// AIETargetModel::getLocalMemorySize/getMemTileSize, which npu1 and npu2 both
// inherit unspecialized from AIE2TargetModel (0x10000 / 0x80000). The
// lowering itself -- a tile-local blockwrite -- is likewise device-independent
// across the AIE2 family, mirroring core_reset.mlir's npu1 case.
module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    aie.runtime_sequence() {
      // CHECK: memref.global "private" constant @blockwrite_data_0 : memref<2xi32> = dense<0>
      // CHECK: %[[G:.*]] = memref.get_global @blockwrite_data_0 : memref<2xi32>
      // CHECK: aiex.npu.blockwrite(%[[G]]) {address = 0 : ui32, column = 0 : i32, row = 2 : i32} : memref<2xi32>
      aiex.buffer_clear(%tile_0_2, 0, 2)
    }
  }
}
