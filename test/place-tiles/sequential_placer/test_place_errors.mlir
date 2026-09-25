//===- test_place_errors.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-place-tiles %s 2>&1 | FileCheck %s

module @three_inputs_exceeds_capacity {
  aie.device(npu1) {
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim3 = aie.logical_tile<ShimNOCTile>(?, ?)

    // CHECK: error: tile (0, 3) requires 3 input/0 output DMA channels, but only 2 input/2 output available
    // CHECK: note: placer selected this tile
    %core = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @in1 (%shim1, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2 (%shim2, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in3 (%shim3, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core) { aie.end }
  }
}

// -----

module @three_outputs_exceeds_capacity {
  aie.device(npu1) {
    // CHECK: error: tile (0, 2) requires 0 input/3 output DMA channels, but only 2 input/2 output available
    %core = aie.logical_tile<CoreTile>(0, 2)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)
    %mem3 = aie.logical_tile<MemTile>(?, ?)

    aie.objectfifo @out1 (%core, {%mem1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2 (%core, {%mem2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out3 (%core, {%mem3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core) { aie.end }
  }
}

// -----

// MemTile DMA exhaustion on single-column device
module @memtile_exhaustion {
  aie.device(npu1_1col) {
    %core1 = aie.logical_tile<CoreTile>(?, ?)
    %core2 = aie.logical_tile<CoreTile>(?, ?)
    %core3 = aie.logical_tile<CoreTile>(?, ?)
    %core4 = aie.logical_tile<CoreTile>(?, ?)

    // First 6 memtiles merge, using all 6 output channels
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)
    %mem3 = aie.logical_tile<MemTile>(?, ?)
    %mem4 = aie.logical_tile<MemTile>(?, ?)
    %mem5 = aie.logical_tile<MemTile>(?, ?)
    %mem6 = aie.logical_tile<MemTile>(?, ?)
    // Global exhaustion: npu1_1col has exactly one MemTile, and it is
    // already fully booked, so no column pin could ever help.
    // CHECK: error: no MemTile on the device has {{[0-9]+ input/[0-9]+ output}} DMA channel(s) free: all 1 MemTile(s) are at {{[0-9]+/[0-9]+}} input, {{[0-9]+/[0-9]+}} output channels used
    // CHECK: note: this is the device's total MemTile DMA budget, not a placement choice
    // CHECK: note: short 1 output channel(s); fan traffic through a MemTile via aie.objectfifo.link
    %mem7 = aie.logical_tile<MemTile>(?, ?)

    aie.objectfifo @of1 (%mem1, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%mem2, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of3 (%mem3, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of4 (%mem4, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of5 (%mem5, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of6 (%mem6, {%core4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.objectfifo @of7 (%mem7, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core1) { aie.end }
    aie.core(%core2) { aie.end }
    aie.core(%core3) { aie.end }
    aie.core(%core4) { aie.end }
  }
}

// -----

// ShimPLTile has no DMA, unconstrained placement not supported
module @shimpl_unsupported {
  aie.device(xcvc1902) {
    // CHECK: error: DMA channel-based SequentialPlacer does not support unplaced ShimPLTiles
    %shim = aie.logical_tile<ShimPLTile>(?, ?)
  }
}

// -----

// Column constraint cannot be satisfied (all tiles in column taken)
module @col_constraint_exhausted {
  aie.device(npu1_1col) {
    // npu1_1col has only column 0 with 4 core rows (2-5)
    %c1 = aie.logical_tile<CoreTile>(0, 2)
    %c2 = aie.logical_tile<CoreTile>(0, 3)
    %c3 = aie.logical_tile<CoreTile>(0, 4)
    %c4 = aie.logical_tile<CoreTile>(0, 5)
    // CHECK: error: no compute tile available matching constraint (0, ?)
    %c5 = aie.logical_tile<CoreTile>(0, ?)

    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    aie.core(%c3) { aie.end }
    aie.core(%c4) { aie.end }
    aie.core(%c5) { aie.end }
  }
}

// -----

// Row constraint cannot be satisfied (all tiles in row taken)
module @row_constraint_exhausted {
  aie.device(npu1) {
    // npu1 has 4 columns (0-3), take all row 2 tiles
    %c1 = aie.logical_tile<CoreTile>(0, 2)
    %c2 = aie.logical_tile<CoreTile>(1, 2)
    %c3 = aie.logical_tile<CoreTile>(2, 2)
    %c4 = aie.logical_tile<CoreTile>(3, 2)
    // CHECK: error: no compute tile available matching constraint (?, 2)
    %c5 = aie.logical_tile<CoreTile>(?, 2)

    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    aie.core(%c3) { aie.end }
    aie.core(%c4) { aie.end }
    aie.core(%c5) { aie.end }
  }
}

// -----

// ShimNOCTile DMA exhaustion on single-column device
module @shimnoc_exhaustion {
  aie.device(npu1_1col) {
    %core1 = aie.logical_tile<CoreTile>(?, ?)
    %core2 = aie.logical_tile<CoreTile>(?, ?)
    %core3 = aie.logical_tile<CoreTile>(?, ?)

    // First two shims merge, using 2 output channels
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    // Global exhaustion: npu1_1col has exactly one ShimNOCTile, and it is
    // already fully booked, so no column pin could ever help.
    // CHECK: error: no ShimNOCTile on the device has {{[0-9]+ input/[0-9]+ output}} DMA channel(s) free: all 1 ShimNOCTile(s) are at {{[0-9]+/[0-9]+}} input, {{[0-9]+/[0-9]+}} output channels used
    // CHECK: note: this is the device's total ShimNOCTile DMA budget, not a placement choice
    // CHECK: note: short 1 output channel(s); fan traffic through a MemTile via aie.objectfifo.link
    %shim3 = aie.logical_tile<ShimNOCTile>(?, ?)

    aie.objectfifo @of1 (%shim1, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%shim2, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.objectfifo @of3 (%shim3, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core1) { aie.end }
    aie.core(%core2) { aie.end }
    aie.core(%core3) { aie.end }
  }
}

// -----

// All compute tiles exhausted (unconstrained)
module @compute_tiles_exhausted {
  aie.device(npu1_1col) {
    // npu1_1col has 4 core tiles (rows 2-5 in column 0)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    %c4 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK: error: no available compute tiles for placement
    %c5 = aie.logical_tile<CoreTile>(?, ?)

    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    aie.core(%c3) { aie.end }
    aie.core(%c4) { aie.end }
    aie.core(%c5) { aie.end }
  }
}

// -----

// 5-tile chain on npu1_1col (4 core rows): chain exhausts the column.
module @cascade_chain_unsatisfiable {
  aie.device(npu1_1col) {
    %a = aie.logical_tile<CoreTile>(0, 5)
    %b = aie.logical_tile<CoreTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %d = aie.logical_tile<CoreTile>(?, ?)
    // CHECK: error: no available compute tiles for placement (cascade adjacency unsatisfiable)
    // CHECK: note: cascade source peer placed at (0, 2)
    %e = aie.logical_tile<CoreTile>(?, ?)
    aie.cascade_flow(%a, %b)
    aie.cascade_flow(%b, %c)
    aie.cascade_flow(%c, %d)
    aie.cascade_flow(%d, %e)
    aie.core(%a) { aie.end }
    aie.core(%b) { aie.end }
    aie.core(%c) { aie.end }
    aie.core(%d) { aie.end }
    aie.core(%e) { aie.end }
  }
}

// -----

// Verifier-legal but lowering-illegal direction: lowering requires src to be
// one row North or one column West of dst (rows increase upward).
module @cascade_pinned_wrong_direction {
  aie.device(npu1) {
    // CHECK: error: tile (0, 5) violates cascade adjacency
    // CHECK: note: cascade source peer placed at (0, 4)
    // CHECK: note: cascade adjacency requires the destination tile to be one row South or one column East of the source tile
    %dst = aie.logical_tile<CoreTile>(0, 5)
    %src = aie.logical_tile<CoreTile>(0, 4)
    aie.cascade_flow(%src, %dst)
    aie.core(%src) { aie.end }
    aie.core(%dst) { aie.end }
  }
}

// -----

// Partial constraint (col=0) is incompatible with pinned src at (1,3).
// Exercises candidate-filter rejection (col-0 tiles exist, none satisfy).
module @cascade_partial_constraint_unsatisfiable {
  aie.device(npu1) {
    %src = aie.logical_tile<CoreTile>(1, 3)
    // CHECK: error: no compute tile available matching constraint (0, ?) and cascade adjacency
    // CHECK: note: cascade source peer placed at (1, 3)
    %dst = aie.logical_tile<CoreTile>(0, ?)
    aie.cascade_flow(%src, %dst)
    aie.core(%src) { aie.end }
    aie.core(%dst) { aie.end }
  }
}

// -----

// Hybrid cascade with conflicting partial constraint on the logical peer.
module @cascade_hybrid_unsatisfiable {
  aie.device(npu1) {
    %src = aie.tile(1, 3)
    // CHECK: error: no compute tile available matching constraint (3, ?) and cascade adjacency
    // CHECK: note: cascade source peer placed at (1, 3)
    %dst = aie.logical_tile<CoreTile>(3, ?)
    aie.cascade_flow(%src, %dst)
    aie.core(%src) { aie.end }
    aie.core(%dst) { aie.end }
  }
}

// -----

// Flow-derived MM2S demand exceeds shim capacity.
module @flow_shim_output_exceeds_capacity {
  aie.device(npu1) {
    // CHECK: error: tile (0, 0) requires 0 input/3 output DMA channels, but only 2 input/2 output available
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    aie.flow(%shim, DMA : 0, %c1, DMA : 0)
    aie.flow(%shim, DMA : 1, %c2, DMA : 0)
    aie.flow(%shim, DMA : 2, %c3, DMA : 0)
  }
}

// -----

// Distinct source DMA channels are NOT deduplicated: three flows on channels
// 0, 1, 2 still consume three MM2S budgets (only same-channel broadcasts
// dedup). This guards against an over-eager dedup that would mask real
// capacity overflows.
module @flow_distinct_channels_no_dedup {
  aie.device(npu1) {
    // CHECK: error: tile (0, 0) requires 0 input/3 output DMA channels, but only 2 input/2 output available
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    aie.flow(%shim, DMA : 0, %c1, DMA : 0)
    aie.flow(%shim, DMA : 1, %c2, DMA : 0)
    aie.flow(%shim, DMA : 2, %c3, DMA : 0)
  }
}

// -----

// Mixed input/output exceeds capacity
module @mixed_channels_exceed_capacity {
  aie.device(npu1) {
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)
    %mem3 = aie.logical_tile<MemTile>(?, ?)

    // CHECK: error: tile (0, 3) requires 2 input/3 output DMA channels, but only 2 input/2 output available
    // CHECK: note: placer selected this tile
    %core = aie.logical_tile<CoreTile>(?, ?)

    // 2 inputs
    aie.objectfifo @in1 (%shim1, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2 (%shim2, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // 3 outputs - exceeds capacity
    aie.objectfifo @out1 (%core, {%mem1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2 (%core, {%mem2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out3 (%core, {%mem3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core) { aie.end }
  }
}

// -----

// Both endpoints pinned far apart: not memory-affinity neighbors.
module @buffer_adjacency_both_pinned_violation {
  aie.device(npu1) {
    // CHECK: error: tile (0, 2) violates shared-L1 buffer adjacency
    // CHECK: note: shared-L1 buffer consumer peer placed at (3, 5)
    // CHECK: note: shared-L1 buffer adjacency requires this LTO to be on a tile whose L1 is shared with the buffer owner's tile
    %owner = aie.logical_tile<CoreTile>(0, 2)
    %buf   = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(3, 5)
    aie.core(%consumer) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
  }
}

// -----

// Pinned owner + unconstrained consumer with a column constraint that has no
// memory-affinity slot relative to the owner.
module @buffer_adjacency_unsatisfiable_column {
  aie.device(npu1) {
    // CHECK: error: no compute tile available matching constraint (3, ?) and shared-L1 buffer adjacency
    // CHECK: note: shared-L1 buffer owner peer placed at (0, 2)
    %owner = aie.logical_tile<CoreTile>(0, 2)
    %buf   = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(3, ?)
    aie.core(%consumer) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
  }
}

// -----

// Star with too many consumers: an owner can host at most 3 cross-tile
// consumers (W, N, S — E is internal in AIE2). A 4th unconstrained consumer
// has no affinity slot left.
module @buffer_adjacency_star_oversubscribed {
  aie.device(npu1) {
    %owner = aie.logical_tile<CoreTile>(1, 3)
    %buf   = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK: error: no available compute tiles for placement (shared-L1 buffer adjacency unsatisfiable)
    // CHECK: note: shared-L1 buffer owner peer placed at (1, 3)
    %c4 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%c1) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    aie.core(%c2) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    aie.core(%c3) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    aie.core(%c4) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
  }
}

// -----

// Same shape as @shim_packet_join_inputs_share_channel in
// test_place_objectfifo.mlir, but none of the three ObjectFifos is
// packet-flagged: no dedup applies, so three producer-side channels are
// required against the shim's 2, and placement fails.
module @shim_three_outputs_two_packet_one_plain {
  aie.device(npu1) {
    // CHECK: error: tile (0, 0) requires 0 input/3 output DMA channels, but only 2 input/2 output available
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @payload (%shim, {%c1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @scales (%shim, {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @plain (%shim, {%c3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    aie.core(%c3) { aie.end }
  }
}

// -----

// Packet ObjectFifos pinned to DISTINCT producer channels are NOT deduped:
// each pinned channel is its own bucket, so this is still 3 output channels
// against the shim's 2. Guards the packet dedup against over-eagerness, the
// same way @flow_distinct_channels_no_dedup guards the flow dedup.
module @shim_packet_objectfifos_distinct_pinned_channels_no_dedup {
  aie.device(npu1) {
    // CHECK: error: tile (0, 0) requires 0 input/3 output DMA channels, but only 2 input/2 output available
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @payload (%shim, {%c1}, 2 : i32) {packet, packet_id = 0 : i8, prod_dma_channel = 0 : i32}
      : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @scales (%shim, {%c2}, 2 : i32) {packet, packet_id = 1 : i8, prod_dma_channel = 1 : i32}
      : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @plain (%shim, {%c3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    aie.core(%c3) { aie.end }
  }
}


// -----

// Two packet-flagged ObjectFifos from one shim, each going to a SEPARATE
// consumer with no objectfifo.link between them: sharing is declared on a
// join (v1 design note, 2026-09-25), so being packet-flagged and unpinned is
// not by itself enough to share a channel. Still 3 output channels against
// the shim's 2, and placement fails -- this is the case
// @shim_packet_join_inputs_share_channel in test_place_objectfifo.mlir
// reverses by adding the join.
module @shim_packet_objectfifos_no_join_no_dedup {
  aie.device(npu1) {
    // CHECK: error: tile (0, 0) requires 0 input/3 output DMA channels, but only 2 input/2 output available
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @payload (%shim, {%c1}, 2 : i32) {packet, packet_id = 0 : i8}
      : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @scales (%shim, {%c2}, 2 : i32) {packet, packet_id = 1 : i8}
      : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @plain (%shim, {%c3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    aie.core(%c3) { aie.end }
  }
}

// -----

// Same shape as @shim_packet_join_inputs_share_channel in
// test_place_objectfifo.mlir (two packet fifos, same shim producer, joined
// at a memtile), but the link does not set `shared_input_channel`: the
// attribute is the opt-in for sharing (v1 design note, 2026-09-25), so an
// unadorned join still bills 2 producer-side channels, 3 total against the
// shim's 2, and placement fails.
module @shim_packet_join_no_attribute_no_dedup {
  aie.device(npu1) {
    // CHECK: error: tile (0, 0) requires 0 input/3 output DMA channels, but only 2 input/2 output available
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %core = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @payload (%shim, {%mem}, 2 : i32) {packet, packet_id = 0 : i8}
      : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @scales (%shim, {%mem}, 2 : i32) {packet, packet_id = 1 : i8}
      : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @joined (%mem, {%core}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@payload, @scales] -> [@joined] ([0, 16][])

    aie.objectfifo @plain (%shim, {%c3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core) { aie.end }
    aie.core(%c3) { aie.end }
  }
}

// -----

// Consumer-side packet ObjectFifos are never merged (sharing is a
// producer-side property, v1 design note): three packet-flagged fifos from
// three different shims into one core require 3 input channels against the
// core's 2, and placement fails.
module @packet_consumer_side_not_merged {
  aie.device(npu1) {
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim3 = aie.logical_tile<ShimNOCTile>(?, ?)

    // CHECK: error: tile (0, 3) requires 3 input/0 output DMA channels, but only 2 input/2 output available
    %core = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @in1 (%shim1, {%core}, 2 : i32) {packet, packet_id = 0 : i8}
      : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2 (%shim2, {%core}, 2 : i32) {packet, packet_id = 1 : i8}
      : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in3 (%shim3, {%core}, 2 : i32) {packet, packet_id = 2 : i8}
      : !aie.objectfifo<memref<16xi32>>

    aie.core(%core) { aie.end }
  }
}
