//===- elide_identical_pm_writes.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi=elide-identical-pm-writes=true --split-input-file %s | FileCheck %s

// A run of >= 8 identical program-memory words that an earlier segment
// already wrote is dropped; the differing tail becomes its own, smaller
// blockwrite at the advanced address. This is the mechanism a device probe
// validated (two designs on one core tile chained through a real
// `@empty_N` firmware reset: a build with the identical prefix physically
// removed ran correctly; a matched build that instead removed part of the
// differing tail produced wrong output at exactly the addresses it
// touched) -- see the PR description.

// CHECK-LABEL: aie.device(npu2_1col) {
// CHECK-DAG: memref.global "private" constant @pm_elide_0 : memref<2xi32> = dense<[99, 100]>
module {
  aie.device(npu2_1col) @empty_0 { aie.end }
  aie.device(npu2_1col) @empty_1 { aie.end }
  aie.device(npu2_1col) @main {
    memref.global "private" constant @seg0_pm : memref<10xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]>
    memref.global "private" constant @seg1_pm : memref<10xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8, 99, 100]>

    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @empty_0, expand_mode = 0 : i32}
      %g0 = memref.get_global @seg0_pm : memref<10xi32>
      // The first segment has nothing to elide against (tracking starts
      // empty for every runtime sequence) and survives untouched.
      // CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 2228224 : ui32} : memref<10xi32>
      aiex.npu.blockwrite(%g0) {address = 2228224 : ui32} : memref<10xi32>

      // The `@empty_1` reset is proven not to clear program memory, so
      // tracking from segment 0 carries across it.
      aiex.npu.load_pdi {device_ref = @empty_1, expand_mode = 0 : i32}
      %g1 = memref.get_global @seg1_pm : memref<10xi32>
      // Words [0:8) of seg1_pm repeat seg0_pm exactly (a run of 8, meeting
      // the minimum) and are dropped; words [8:10) differ and survive as a
      // 2-word blockwrite at address 2228224 + 4*8 = 2228256.
      // CHECK-NOT: aiex.npu.blockwrite({{.*}}) {address = 2228224 : ui32} : memref<10xi32>
      // CHECK: %[[TAIL:.*]] = memref.get_global @pm_elide_0 : memref<2xi32>
      // CHECK: aiex.npu.blockwrite(%[[TAIL]]) {address = 2228256 : ui32} : memref<2xi32>
      aiex.npu.blockwrite(%g1) {address = 2228224 : ui32} : memref<10xi32>
    }
  }
}

// -----

// A run shorter than the minimum (7 words here) does not amortise the new
// blockwrite's own header, so it is conservatively kept whole rather than
// split.

// CHECK-LABEL: aie.device(npu2_1col) {
module {
  aie.device(npu2_1col) @empty_0 { aie.end }
  aie.device(npu2_1col) @empty_1 { aie.end }
  aie.device(npu2_1col) @main {
    memref.global "private" constant @seg0_pm : memref<10xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]>
    memref.global "private" constant @seg1_pm : memref<10xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 42, 9, 10]>

    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @empty_0, expand_mode = 0 : i32}
      %g0 = memref.get_global @seg0_pm : memref<10xi32>
      aiex.npu.blockwrite(%g0) {address = 2228224 : ui32} : memref<10xi32>
      aiex.npu.load_pdi {device_ref = @empty_1, expand_mode = 0 : i32}
      %g1 = memref.get_global @seg1_pm : memref<10xi32>
      // Only a run of 7 identical words precedes the one differing word
      // (index 7); that is below kMinElideRun (8), so this op must be left
      // exactly as authored.
      // CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 2228224 : ui32} : memref<10xi32>
      aiex.npu.blockwrite(%g1) {address = 2228224 : ui32} : memref<10xi32>
    }
  }
}

// -----

// A REAL PDI load (not one of this pass's own `@empty_N` resets) carries
// content this pass never read, so it must invalidate every tracked
// program-memory address -- the same 8-word run that was safely elided
// against an `@empty_N` reset above must NOT be elided across a real load.

// CHECK-LABEL: aie.device(npu2_1col) {
module {
  aie.device(npu2_1col) @real_dev { aie.end }
  aie.device(npu2_1col) @main {
    memref.global "private" constant @seg0_pm : memref<10xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]>
    memref.global "private" constant @seg1_pm : memref<10xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8, 99, 100]>

    aie.runtime_sequence(%arg0: memref<1xi32>) {
      %g0 = memref.get_global @seg0_pm : memref<10xi32>
      aiex.npu.blockwrite(%g0) {address = 2228224 : ui32} : memref<10xi32>
      // expand_mode = none: a real PDI load this pass does not expand and
      // whose device content it never read.
      aiex.npu.load_pdi {device_ref = @real_dev, expand_mode = 0 : i32}
      %g1 = memref.get_global @seg1_pm : memref<10xi32>
      // Must survive whole: the tracker was invalidated by the load above.
      // CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 2228224 : ui32} : memref<10xi32>
      aiex.npu.blockwrite(%g1) {address = 2228224 : ui32} : memref<10xi32>
    }
  }
}
