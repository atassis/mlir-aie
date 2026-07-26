//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Clears a core tile's resident accumulator via the aiex.buffer_clear op. See
// README.md.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    %prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "prod_lock"}
    %cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "cons_lock"}

    // Pinned at local address 0x400 (1024) so the runtime sequence can target
    // it directly with aiex.buffer_clear, which addresses tile-local data
    // memory by offset (like aiex.npu.blockwrite's column/row form), not by
    // symbol. Address 0 is NOT safe to use here: aie.core below takes the ODS
    // default stack_size of 0x400, and --alloc-scheme=basic-sequential (what
    // run.lit passes to aiecc) rejects any manually-pinned buffer whose
    // address falls inside [0, stack_size) as overlapping the stack. 0x400 is
    // the first word-aligned address past the default stack.
    %acc = aie.buffer(%tile_0_2) {sym_name = "acc", address = 1024 : i32} : memref<8xi32>

    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c1_i32 = arith.constant 1 : i32

      aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_i32)
      // acc[i] = i + 1: a resident "accumulator" the core has filled with a
      // distinct nonzero value per word, so a later all-zero readback can only
      // be explained by aiex.buffer_clear, not by an all-uniform coincidence.
      scf.for %i = %c0 to %c8 step %c1 {
        %i_i32 = arith.index_cast %i : index to i32
        %v = arith.addi %i_i32, %c1_i32 : i32
        memref.store %v, %acc[%i] : memref<8xi32>
      }
      aie.use_lock(%cons_lock, Release, %c1_i32)
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_lk = arith.constant 1 : i32
      aie.use_lock(%cons_lock, AcquireGreaterEqual, %c1_lk)
      aie.dma_bd(%acc : memref<8xi32>) { len = 8 : i32 }
      aie.use_lock(%prod_lock, Release, %c1_lk)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 0)

    aie.runtime_sequence @seq(%arg0: memref<16xi32>) {
      // Batch 1: the core has run once from load; collect acc = [1..8] into
      // arg0[0:8], proving the accumulator really holds nonzero data.
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out0}

      // Clear the accumulator directly from the runtime sequence: 8 words at
      // its pinned local offset 0x400. No core re-run is needed -- unlike
      // aiex.core_reset, this op clears data memory in place.
      aiex.buffer_clear(%tile_0_2, 1024, 8)

      // The mem-side DMA loop (^bb1 above) is back waiting on
      // cons_lock >= 1 after batch 1. Signal it directly from the sequence to
      // fire a second S2MM read of acc, now that aiex.buffer_clear has zeroed
      // it -- no compute core involvement, so this isolates the op itself as
      // the thing under test.
      aiex.set_lock(%cons_lock, 1)

      // Batch 2: collect acc = [0..0] into arg0[8:16].
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 8][1, 1, 1, 8][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @out0} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}
