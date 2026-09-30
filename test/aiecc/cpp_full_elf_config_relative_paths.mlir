//===- cpp_full_elf_config_relative_paths.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// full_elf_config.json's sibling-file paths (PDI_file, TXN_ctrl_code_file) must
// be relative to the JSON's own directory, not baked in as absolute paths --
// otherwise the JSON stops resolving once its directory is copied elsewhere
// (e.g. by the aiecc Python whole-run cache restoring a build under a new
// path). aiebu-asm's aie2_config target resolves a relative path in the JSON
// against the JSON's own directory.

// REQUIRES: peano
// REQUIRES: aiebu

// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && aiecc --get-full-elf --full-elf-name=full.elf --tmpdir=%t %s 2>&1
// RUN: cat %t/full_elf_config.json | FileCheck %s
// RUN: rm -rf %t.moved && cp -r %t %t.moved && rm -rf %t
// RUN: cd / && aiebu-asm -t aie2_config -j %t.moved/full_elf_config.json -o %t.moved/reassembled.elf
// RUN: cmp %t.moved/full.elf %t.moved/reassembled.elf

// A relative filename, not an absolute path baked in from the original tmpdir.
// CHECK: "TXN_ctrl_code_file": "{{[^/"]+}}"

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c1_i32 = arith.constant 1 : i32

      %elem_in = aie.objectfifo.acquire @of_in(Consume, 1) : memref<16xi32>
      %elem_out = aie.objectfifo.acquire @of_out(Produce, 1) : memref<16xi32>

      scf.for %i = %c0 to %c16 step %c1 {
        %val = memref.load %elem_in[%i] : memref<16xi32>
        %result = arith.addi %val, %c1_i32 : i32
        memref.store %result, %elem_out[%i] : memref<16xi32>
      }

      aie.objectfifo.release @of_in(Consume, 1)
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%in : memref<16xi32>, %out : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_in, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
