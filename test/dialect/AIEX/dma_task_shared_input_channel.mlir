//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// dma_configure_task_for's `interleave` attribute names a shared_input_channel group's followers, leader ($alloc) first.
// These checks only fire pre-split, while $alloc/$interleave still resolve to
// ObjectFifoCreateOps; post-split IR defers them (see
// shared_input_channel_no_attribute.mlir's sibling positive case).

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// BD count must equal the group size (leader + every interleave entry).
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @payload (%shim, {%mem}, 2 : i32) {packet, packet_id = 0 : i8} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @scale (%shim, {%mem}, 2 : i32) {packet, packet_id = 1 : i8} : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @joined (%mem, {%core}, 4 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo.link [@payload, @scale] -> [@joined] ([0, 16][]) {shared_input_channel}

    aie.runtime_sequence(%w : memref<1024xi32>) {
      // expected-error@+1 {{interleave names a group of 2 members but the body has 1 buffer descriptors}}
      %t = aiex.dma_configure_task_for @payload {
        aie.dma_bd(%w : memref<1024xi32> offset = 0 len = 16 sizes = [8, 1, 1, 16] strides = [16, 0, 0, 1])
        aie.end
      } {repeat_count = 7 : i32, interleave = [@scale]}
    }
  }
}

// -----

// Every BD's outermost size must equal repeat_count + 1 (lockstep).
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @payload (%shim, {%mem}, 2 : i32) {packet, packet_id = 0 : i8} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @scale (%shim, {%mem}, 2 : i32) {packet, packet_id = 1 : i8} : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @joined (%mem, {%core}, 4 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo.link [@payload, @scale] -> [@joined] ([0, 16][]) {shared_input_channel}

    aie.runtime_sequence(%w : memref<1024xi32>) {
      %t = aiex.dma_configure_task_for @payload {
        // expected-error@+1 {{outermost size (8) must equal repeat_count + 1 (6) to stay in lockstep}}
        aie.dma_bd(%w : memref<1024xi32> offset = 0 len = 16 sizes = [8, 1, 1, 16] strides = [16, 0, 0, 1])
        aie.next_bd ^s
      ^s:
        aie.dma_bd(%w : memref<1024xi32> offset = 512 len = 4 sizes = [8, 1, 1, 4] strides = [4, 0, 0, 1])
        aie.end
      } {repeat_count = 5 : i32, interleave = [@scale]}
    }
  }
}

// -----

// interleave must name exactly one shared_input_channel group, leader first:
// @other is a real ObjectFifo but not @payload's group.
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %core2 = aie.tile(0, 3)
    aie.objectfifo @payload (%shim, {%mem}, 2 : i32) {packet, packet_id = 0 : i8} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @scale (%shim, {%mem}, 2 : i32) {packet, packet_id = 1 : i8} : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @joined (%mem, {%core}, 4 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo.link [@payload, @scale] -> [@joined] ([0, 16][]) {shared_input_channel}
    aie.objectfifo @other (%shim, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.runtime_sequence(%w : memref<1024xi32>) {
      // expected-error@+1 {{alloc and interleave do not name exactly one shared_input_channel group, leader first}}
      %t = aiex.dma_configure_task_for @payload {
        aie.dma_bd(%w : memref<1024xi32> offset = 0 len = 16 sizes = [8, 1, 1, 16] strides = [16, 0, 0, 1])
        aie.next_bd ^s
      ^s:
        aie.dma_bd(%w : memref<1024xi32> offset = 512 len = 16 sizes = [8, 1, 1, 16] strides = [16, 0, 0, 1])
        aie.end
      } {repeat_count = 7 : i32, interleave = [@other]}
    }
  }
}

// -----

// Deadlock rule (design note §5): a BD invocation may not move more objects
// (m) than its member's own consumer depth. @scale's depth is 1; the BD below
// moves 2 of its 4-i32 objects per invocation.
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @payload (%shim, {%mem}, 2 : i32) {packet, packet_id = 0 : i8} : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @scale (%shim, {%mem}, 1 : i32) {packet, packet_id = 1 : i8} : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @joined (%mem, {%core}, 4 : i32) : !aie.objectfifo<memref<36xi32>>
    aie.objectfifo.link [@payload, @scale] -> [@joined] ([0, 32][]) {shared_input_channel}

    aie.runtime_sequence(%w : memref<1024xi32>) {
      %t = aiex.dma_configure_task_for @payload {
        aie.dma_bd(%w : memref<1024xi32> offset = 0 len = 32 sizes = [8, 1, 1, 32] strides = [32, 0, 0, 1])
        aie.next_bd ^s
      ^s:
        // expected-error@+1 {{moves 2 objects per invocation, which exceeds its shared_input_channel consumer's depth (1)}}
        aie.dma_bd(%w : memref<1024xi32> offset = 512 len = 8 sizes = [8, 1, 2, 4] strides = [4, 0, 4, 1])
        aie.end
      } {repeat_count = 7 : i32, interleave = [@scale]}
    }
  }
}
