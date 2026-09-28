//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core/mem-tile task-completion-token push has no shim-tile precedent for a
// missing controller_id (unlike a shim push, which many isolated unit tests
// exercise without ever running --aie-assign-tile-controller-ids). Silently
// dropping the controller-id write here left the awaiting dma_await_task
// polling a token that is never stamped: every byte of the transfer arrives,
// and the runtime sequence hangs anyway.

// RUN: aie-opt --split-input-file --aie-dma-to-npu --verify-diagnostics %s | FileCheck %s

module {
  aie.device(npu1) {
    aie.runtime_sequence() {
      %rc0 = arith.constant 0 : i32
      %bd0 = arith.constant 3 : i32
      // expected-error @+3 {{issues a task-completion token on tile (0, 2), which has no controller_id attribute}}
      // expected-note @+2 {{Run the `--aie-assign-tile-controller-ids` pass before `--aie-dma-to-npu`.}}
      // expected-error @+1 {{failed to legalize operation 'aiex.npu.push_queue'}}
      aiex.npu.push_queue (0, 2, S2MM:0) bd_id %bd0 repeat %rc0 {issue_token = true} : i32, i32
    }
  }
}

// -----

// The same push on the same tile succeeds once the tile carries the
// controller_id --aie-assign-tile-controller-ids would have assigned.
// CHECK-LABEL: @with_controller_id
// CHECK: aiex.npu.maskwrite32
// CHECK: aiex.npu.write32
module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.runtime_sequence @with_controller_id() {
      %rc0 = arith.constant 0 : i32
      %bd0 = arith.constant 3 : i32
      aiex.npu.push_queue (0, 2, S2MM:0) bd_id %bd0 repeat %rc0 {issue_token = true} : i32, i32
    }
  }
}
