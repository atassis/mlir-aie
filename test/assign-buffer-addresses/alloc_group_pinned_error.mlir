//===- alloc_group_pinned_error.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s

// Two pinned members of ONE group are live together and must not overlap.
// CHECK: error: 'aie.buffer' op would override allocated address in its own alloc_group
module {
  aie.device(npu2) {
    %m = aie.tile(0, 1)
    %x = aie.buffer(%m) { sym_name = "x", address = 0 : i32, alloc_group = "a" } : memref<4096xi8>
    %y = aie.buffer(%m) { sym_name = "y", address = 1024 : i32, alloc_group = "a" } : memref<2048xi8>
  }
}
