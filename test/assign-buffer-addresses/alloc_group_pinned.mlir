//===- alloc_group_pinned.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-buffer-addresses %s | FileCheck %s

// Address-pinned members of different groups may share bytes; the unpinned
// buffer is placed clear of both.
// CHECK: {address = 0 : i32, alloc_group = "a", mem_bank = 0 : i32, sym_name = "x"}
// CHECK: {address = 0 : i32, alloc_group = "b", mem_bank = 0 : i32, sym_name = "y"}
// CHECK: {address = 4096 : i32, mem_bank = 0 : i32, sym_name = "free"}
module {
  aie.device(npu2) {
    %m = aie.tile(0, 1)
    %x = aie.buffer(%m) { sym_name = "x", address = 0 : i32, alloc_group = "a" } : memref<4096xi8>
    %y = aie.buffer(%m) { sym_name = "y", address = 0 : i32, alloc_group = "b" } : memref<2048xi8>
    %free = aie.buffer(%m) { sym_name = "free" } : memref<1024xi8>
  }
}
