//===- aie_negctl.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Negative control: aie.mlir with both BDs' outer stride zeroed, so every
// repeat re-reads slice 0 instead of advancing. Proves test_elf.cpp's
// byte-exact check has power to fail.

module @packet_hw_probe {
  aie.device(npu2) @test {
    %shim = aie.tile(0, 0)
    %mem  = aie.tile(0, 1)
    %core = aie.tile(0, 2)

    aie.objectfifo @payload (%shim, {%mem}, 2 : i32)
      {packet, packet_id = 0 : i8} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @scale (%shim, {%mem}, 2 : i32)
      {packet, packet_id = 1 : i8} : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @joined (%mem, {%core}, 4 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo.link [@payload, @scale] -> [@joined] ([0, 16][]) {shared_input_channel}

    aie.objectfifo @outp (%core, {%shim}, 4 : i32) : !aie.objectfifo<memref<20xi32>>

    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c20 = arith.constant 20 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %in = aie.objectfifo.acquire @joined (Consume, 1) : memref<20xi32>
        %out = aie.objectfifo.acquire @outp (Produce, 1) : memref<20xi32>
        scf.for %j = %c0 to %c20 step %c1 {
          %v = memref.load %in[%j] : memref<20xi32>
          memref.store %v, %out[%j] : memref<20xi32>
        }
        aie.objectfifo.release @joined (Consume, 1)
        aie.objectfifo.release @outp (Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sequence(%w : memref<1024xi32>, %o : memref<256xi32>) {
      aiex.npu.load_pdi { device_ref = @test }
      %to = aiex.dma_configure_task_for @outp {
        aie.dma_bd(%o : memref<256xi32> offset = 0 len = 20
                   sizes = [8, 1, 1, 20] strides = [20, 0, 0, 1])
        aie.end
      } {repeat_count = 7 : i32, issue_token = true}
      aiex.dma_start_task(%to)

      %ti = aiex.dma_configure_task_for @payload {
          aie.dma_bd(%w : memref<1024xi32> offset = 0 len = 16
                     sizes = [8, 1, 1, 16] strides = [0, 0, 0, 1])
          aie.next_bd ^s
        ^s:
          aie.dma_bd(%w : memref<1024xi32> offset = 512 len = 4
                     sizes = [8, 1, 1, 4] strides = [0, 0, 0, 1])
          aie.end
      } {repeat_count = 7 : i32, issue_token = true, interleave = [@scale]}
      aiex.dma_start_task(%ti)

      aiex.dma_await_task(%ti)
      aiex.dma_await_task(%to)
    }
  }
}
