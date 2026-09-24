//===- scratchpad_length_parameter_3d.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A length_parameter BD whose d2 dimension has static size 1 but a nonzero
// stride: the runtime extends the d2 wrap count past 1 via buffer_length, so
// the stride is load-bearing even though the dimension "looks" contiguous
// (size-1 dims are otherwise stride-irrelevant). This must survive lowering
// as an ND BD (not be folded to linear mode, which drops d2_stride).
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --pass-pipeline='any(aie-lower-scratchpad-parameters,aie.device(aie-dma-tasks-to-npu))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=LENPARAM

// LENPARAM-LABEL: aie.device(npu2)
// LENPARAM:         aie.runtime_sequence @length_param_3d
// LENPARAM:           aiex.npu.writebd
// -- Not linear mode: d0/d1 carry the real inner block shape.
// LENPARAM-SAME:        d0_size = 64
// LENPARAM-SAME:        d1_size = 8
// -- d2 stride must survive: 2048 elements, the jump to this column's next
// -- round-robin block. Step fields are hardware-encoded as actual-1.
// LENPARAM-SAME:        d2_stride = 2047
module {
  aiex.scratchpad_parameter @extra_blocks : i32
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @length_param_3d(%in: memref<16384xi32>) {
      %tk = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%in : memref<16384xi32> offset = 512 len = 512 sizes = [1, 1, 8, 64] strides = [0, 2048, 64, 1])
          {bd_id = 0 : i32, length_granule = 512 : i32, length_parameter = @extra_blocks}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
    }
  }
}

// -----

// Same shape without length_parameter: must still fold to linear mode (no
// regression from the guard above). d2's stride is genuinely irrelevant
// here since nothing extends its wrap count at run time.

// RUN: aie-opt --pass-pipeline='any(aie.device(aie-dma-tasks-to-npu))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=NOLENPARAM

// NOLENPARAM-LABEL: aie.device(npu2)
// NOLENPARAM:         aie.runtime_sequence @no_length_param_3d
// NOLENPARAM:           aiex.npu.writebd
// NOLENPARAM-SAME:        buffer_length = 512
// NOLENPARAM-SAME:        d0_size = 0
// NOLENPARAM-SAME:        d1_size = 0
module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @no_length_param_3d(%in: memref<16384xi32>) {
      %tk = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%in : memref<16384xi32> offset = 512 len = 512 sizes = [1, 1, 8, 64] strides = [0, 512, 64, 1])
          {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
    }
  }
}
