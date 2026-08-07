//===- test_place_spread_unanchored_tiles.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Exercises the `spread-unanchored-tiles` pass option on aie-place-tiles.
// computeCentroidColumn derives a non-core LTO's target column from its
// CoreTile peers, so a design with no compute tiles has no anchor at all.
// The default keeps the historical column-0 fallback, which puts every
// candidate at the same distance and lets the load tiebreaker fire only
// once column 0 is DMA-full. With the option, an unanchored LTO is ranked
// by DMA load instead, in the directions it actually needs.

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s --check-prefix=PILE
// RUN: aie-opt --split-input-file --aie-place-tiles='spread-unanchored-tiles=false' %s | FileCheck %s --check-prefix=PILE
// RUN: aie-opt --split-input-file --aie-place-tiles='spread-unanchored-tiles=true' %s | FileCheck %s --check-prefix=SPREAD

// Four independent shim->memtile->shim passthrough chains, no compute tile
// anywhere. Piled up, all four memtile LTOs merge onto one physical memtile
// and the design runs ~1-way instead of 4-way. Spread, chain i lands entirely
// on column i -- shim and memtile together, which is what ranking by
// direction-matched load buys: a summed load would push each chain's read and
// write onto different columns.

// PILE-LABEL:     @anchorless_passthrough
// PILE-DAG:       aie.tile(0, 0)
// PILE-DAG:       aie.tile(0, 1)
// PILE-NOT:       aie.tile(2, 0)
// PILE-NOT:       aie.tile(1, 1)

// SPREAD-LABEL:   @anchorless_passthrough
// SPREAD-DAG:     aie.tile(0, 0)
// SPREAD-DAG:     aie.tile(1, 0)
// SPREAD-DAG:     aie.tile(2, 0)
// SPREAD-DAG:     aie.tile(3, 0)
// SPREAD-DAG:     aie.tile(0, 1)
// SPREAD-DAG:     aie.tile(1, 1)
// SPREAD-DAG:     aie.tile(2, 1)
// SPREAD-DAG:     aie.tile(3, 1)
module @anchorless_passthrough {
  aie.device(npu1) {
    %shim0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem0 = aie.logical_tile<MemTile>(?, ?)
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)
    %shim3 = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem3 = aie.logical_tile<MemTile>(?, ?)
    aie.flow(%shim0, DMA : 0, %mem0, DMA : 0)
    aie.flow(%mem0, DMA : 0, %shim0, DMA : 0)
    aie.flow(%shim1, DMA : 0, %mem1, DMA : 0)
    aie.flow(%mem1, DMA : 0, %shim1, DMA : 0)
    aie.flow(%shim2, DMA : 0, %mem2, DMA : 0)
    aie.flow(%mem2, DMA : 0, %shim2, DMA : 0)
    aie.flow(%shim3, DMA : 0, %mem3, DMA : 0)
    aie.flow(%mem3, DMA : 0, %shim3, DMA : 0)
  }
}

// -----

// One CoreTile is enough to anchor both non-core LTOs, so the option changes
// nothing: the centroid is real and distance still leads.

// PILE-LABEL:     @anchored_is_unaffected
// PILE-DAG:       aie.tile(0, 0)
// PILE-DAG:       aie.tile(0, 1)
// PILE-NOT:       aie.tile(1, 0)
// PILE-NOT:       aie.tile(1, 1)

// SPREAD-LABEL:   @anchored_is_unaffected
// SPREAD-DAG:     aie.tile(0, 0)
// SPREAD-DAG:     aie.tile(0, 1)
// SPREAD-NOT:     aie.tile(1, 0)
// SPREAD-NOT:     aie.tile(1, 1)
module @anchored_is_unaffected {
  aie.device(npu1) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %core = aie.tile(0, 2)
    aie.flow(%shim, DMA : 0, %mem, DMA : 0)
    aie.flow(%mem, DMA : 0, %core, DMA : 0)
  }
}
