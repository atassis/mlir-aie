//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// dma_configure_task_for's `interleave`: one task, one BD per shared_input_channel group member, leader
// first. Substitute keeps the leader's (tile, dir, channel) and stamps each
// BD with ITS OWN member's packet info instead of the task's single default.
// Matches the device-proven encoding: one push_queue chains
// both BDs, each keeping its own packet_id.

// RUN: aie-opt --aie-substitute-shim-dma-allocations --aie-dma-tasks-to-npu %s | FileCheck %s

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @payload_shim_alloc (%tile_0_0, MM2S, 0, <pkt_type = 0, pkt_id = 0>)
    aie.shim_dma_allocation @scale_shim_alloc (%tile_0_0, MM2S, 0, <pkt_type = 0, pkt_id = 1>)

    aie.runtime_sequence(%arg0: memref<1024xi32>) {
      // CHECK: aiex.npu.writebd {{.*}}bd_id = 1{{.*}}enable_packet = 1{{.*}}iteration_size = 7{{.*}}next_bd = 2{{.*}}packet_id = 0{{.*}}use_next_bd = 1
      // CHECK: aiex.npu.writebd {{.*}}bd_id = 2{{.*}}enable_packet = 1{{.*}}iteration_size = 7{{.*}}next_bd = 0{{.*}}packet_id = 1{{.*}}use_next_bd = 0
      %t = aiex.dma_configure_task_for @payload_shim_alloc {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 16
                   sizes = [8, 1, 1, 16] strides = [16, 0, 0, 1]) {bd_id = 1 : i32}
        aie.next_bd ^s
      ^s:
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 4
                   sizes = [8, 1, 1, 4] strides = [4, 0, 0, 1]) {bd_id = 2 : i32}
        aie.end
      } {repeat_count = 7 : i32, issue_token = true, interleave = [@scale_shim_alloc]}

      // One push onto the shared channel drives both BDs through the chain.
      // CHECK-DAG: %[[BD:.*]] = arith.constant 1 : i32
      // CHECK-DAG: %[[RC:.*]] = arith.constant 7 : i32
      // CHECK: aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %[[BD]] repeat %[[RC]] {issue_token = true} : i32, i32
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
