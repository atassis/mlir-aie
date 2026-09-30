//===- reproducible_xclbin.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two builds of one design, in different directories and tmpdirs, give the
// same xclbin.

// REQUIRES: peano

// RUN: rm -rf %t && mkdir -p %t/a %t/b
// RUN: cd %t/a && %aiecc --tmpdir=tmp_a --get-xclbin --xclbin-name=aie.xclbin %S/simple_xclbin.mlir
// RUN: cd %t/b && %aiecc --tmpdir=tmp_b --get-xclbin --xclbin-name=aie.xclbin %S/simple_xclbin.mlir
// RUN: cmp %t/a/aie.xclbin %t/b/aie.xclbin

module {}
