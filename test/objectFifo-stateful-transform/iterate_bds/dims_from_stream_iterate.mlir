//===- dims_from_stream_iterate.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" %s | FileCheck %s

// dimensionsFromStream on a join's input pools and iterate_bds on the join's
// output pool are independent hardware fields (ND dims vs the iteration
// field) and compose on one BD: the same S2MM BD carries both the data
// layout transform (sizes/strides) and the iteration attribute.

// CHECK: aie.dma_bd(%{{.*}}of2_buff : memref<512xi32> offset = 0 len = 128 sizes = [3] strides = [4]) {{.*}}iteration = #aie.bd_iteration<size = 2, stride = 256, current = 0>
// CHECK: aie.dma_bd(%{{.*}}of2_buff : memref<512xi32> offset = 128 len = 128 sizes = [2] strides = [2]) {{.*}}iteration = #aie.bd_iteration<size = 2, stride = 256, current = 0>

module {
  aie.device(xcve2302) {
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile12, {%tile11 dimensionsFromStream [<size = 3, stride = 4>]}, 2 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @of1 (%tile33, {%tile11 dimensionsFromStream [<size = 2, stride = 2>]}, 2 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @of2 (%tile11, {%tile23}, 2 : i32) {iterate_bds} : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@of0, @of1] -> [@of2] ([0, 128] [])
  }
}
