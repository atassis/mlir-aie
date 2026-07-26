//===- buffer_clear_invalid.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -split-input-file -verify-diagnostics --aie-lower-buffer-clear %s

// A shim NOC tile is rejected: it has no local data memory module at all. This
// matches aie-rt's XAie_DataMemBlockWrite, which only accepts AIETILE and
// MEMTILE tile types (driver/src/memory/xaie_mem.c).
module {
  aie.device(npu2) {
    %shim_tile = aie.tile(0, 0)
    aie.runtime_sequence() {
      // expected-error @+1 {{tile (0, 0) has no local data memory to clear (only core and mem tiles do)}}
      aiex.buffer_clear(%shim_tile, 0, 4)
    }
  }
}

// -----

// A zero length is rejected: there is no region to clear, and silently
// accepting it would let a degenerate call site (e.g. a mis-specified size
// parameter) become a silent no-op instead of a caught mistake.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.runtime_sequence() {
      // expected-error @+1 {{length must be nonzero}}
      aiex.buffer_clear(%tile, 0, 0)
    }
  }
}

// -----

// A non-word-aligned address is rejected. The op lowers to a single
// npu.blockwrite, which writes whole 32-bit words; there is no
// read-modify-write path here for a partial edge word the way aie-rt's
// byte-granular XAie_DataMemBlockWrite has, so a misaligned start is a hard
// error rather than a silently-rounded one.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.runtime_sequence() {
      // expected-error @+1 {{address 2 is not 4-byte (word) aligned}}
      aiex.buffer_clear(%tile, 2, 1)
    }
  }
}

// -----

// A region that runs past the end of the tile's local data memory is
// rejected. Core tile data memory on the AIE2 family (npu1 and npu2 alike) is
// 0x10000 = 65536 bytes (AIETargetModel::getLocalMemorySize, matching
// aie-rt's Aie2PTileMemMod.Size in xaie2pgbl_reginit.c): address 65532 (the
// last word) plus a 2-word (8-byte) region overruns by 4 bytes.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.runtime_sequence() {
      // expected-error @+1 {{region [65532, 65540) exceeds tile (0, 2)'s local data memory size (65536 bytes)}}
      aiex.buffer_clear(%tile, 65532, 2)
    }
  }
}

// -----

// A mem tile region that runs past 0x80000 = 524288 bytes
// (AIETargetModel::getMemTileSize, matching aie-rt's Aie2PMemTileMemMod.Size)
// is rejected the same way, with the mem tile's larger bound.
module {
  aie.device(npu2) {
    %mem_tile = aie.tile(0, 1)
    aie.runtime_sequence() {
      // expected-error @+1 {{region [524284, 524292) exceeds tile (0, 1)'s local data memory size (524288 bytes)}}
      aiex.buffer_clear(%mem_tile, 524284, 2)
    }
  }
}

// -----

// AIE1 is rejected: the op lowers to an NPU instruction and the runtime
// sequence has no meaning on AIE1, matching CoreResetOp's and SetLockOp's AIE1
// rejection.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(0, 3)
    aie.runtime_sequence() {
      // expected-error @+1 {{not supported on AIE1}}
      aiex.buffer_clear(%tile, 0, 4)
    }
  }
}
