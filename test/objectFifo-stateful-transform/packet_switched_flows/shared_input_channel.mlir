//===- shared_input_channel.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A join whose link carries `shared_input_channel` bills its packet-flagged
// source fifos against ONE MM2S channel: npu1's shim has 2, so @payload,
// @scales and @plain (3 producer-side fifos total) fit. Without the
// attribute this is 3 channels against 2 and fails (see
// shared_input_channel_no_attribute.mlir).

// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s --check-prefix=SPLIT
// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" %s | FileCheck %s

module @shim_packet_join_shares_channel {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core0 = aie.tile(0, 2)
    %core1 = aie.tile(0, 3)

    aie.objectfifo @payload (%shim, {%mem}, 2 : i32) {packet, packet_id = 0 : i8} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @scales (%shim, {%mem}, 2 : i32) {packet, packet_id = 1 : i8} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @joined (%mem, {%core0}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@payload, @scales] -> [@joined] ([0, 16][]) {shared_input_channel}

    aie.objectfifo @plain (%shim, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core0) { aie.end }
    aie.core(%core1) { aie.end }
  }
}

// The leader's endpoint draws its own channel as usual; the follower's names
// the leader instead of pinning a channel.
// SPLIT: aie.route_endpoint @payload_prod_dma({{.*}}) DMA {fifoName = "payload"}
// SPLIT: aie.route_endpoint @scales_prod_dma({{.*}}) DMA {fifoName = "scales", sharesChannel = @payload_prod_dma}

// Both packet fifos land on shim MM2S 0 with their own packet ID; @plain, not
// in the group, draws the shim's other channel, MM2S 1.
// CHECK: aie.shim_dma_allocation @payload_shim_alloc({{.*}}, MM2S, 0, <pkt_type = 0, pkt_id = 0>)
// CHECK: aie.shim_dma_allocation @scales_shim_alloc({{.*}}, MM2S, 0, <pkt_type = 0, pkt_id = 1>)
// CHECK: aie.shim_dma_allocation @plain_shim_alloc({{.*}}, MM2S, 1)

// The memtile join still demuxes the two packet IDs onto its own two S2MM
// channels -- sharing is producer-side only.
// CHECK: aie.dma_start(S2MM, 0,
// CHECK: aie.dma_start(S2MM, 1,
