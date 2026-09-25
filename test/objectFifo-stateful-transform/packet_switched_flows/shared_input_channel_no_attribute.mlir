//===- shared_input_channel_no_attribute.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Same shape as shared_input_channel.mlir, but the join does not set
// `shared_input_channel`: @payload and @scales draw distinct MM2S channels,
// so the shim needs 3 against npu1's 2 and allocation fails.

// RUN: not aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" %s 2>&1 | FileCheck %s

// CHECK: error: 'aie.tile' op number of output DMA channel exceeded! requires at least 3 MM2S channels, but capacity is 2

module @shim_packet_join_no_attribute {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core0 = aie.tile(0, 2)
    %core1 = aie.tile(0, 3)

    aie.objectfifo @payload (%shim, {%mem}, 2 : i32) {packet, packet_id = 0 : i8} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @scales (%shim, {%mem}, 2 : i32) {packet, packet_id = 1 : i8} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @joined (%mem, {%core0}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@payload, @scales] -> [@joined] ([0, 16][])

    aie.objectfifo @plain (%shim, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core0) { aie.end }
    aie.core(%core1) { aie.end }
  }
}
