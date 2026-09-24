// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test: DMA transfer length extension via length_parameter.
//
// A DDR -> memtile -> DDR passthrough (aie.objectfifo.link, no core). Both
// shim BDs carry the same length_parameter @len, granule 64 elements (i32):
// the transferred length is base_len + StateTable[@len] * 64 words. Base BD
// length is one 64-word object.
//
//   @len = 0  -> 64 words moved
//   @len = 5  -> 384 words moved
//   @len = 3  -> 256 words moved
//
module {
    aiex.scratchpad_parameter @len : i32

    aie.device(npu2) @empty { }

    aie.device(npu2) @test {
        %t00 = aie.tile(0, 0)
        %t01 = aie.tile(0, 1)

        aie.objectfifo @in  (%t00, {%t01}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
        aie.objectfifo @out (%t01, {%t00}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
        aie.objectfifo.link [@in] -> [@out] ([] [])

        aie.runtime_sequence @sequence(%a : memref<1024xi32>, %c : memref<1024xi32>) {
            aiex.npu.load_pdi { device_ref = @empty }
            aiex.npu.load_pdi { device_ref = @test }

            %ti = aiex.dma_configure_task_for @in {
                aie.dma_bd(%a : memref<1024xi32> offset = 0 len = 64) {length_parameter = @len, length_granule = 64 : i32}
                aie.end
            }
            %to = aiex.dma_configure_task_for @out {
                aie.dma_bd(%c : memref<1024xi32> offset = 0 len = 64) {length_parameter = @len, length_granule = 64 : i32}
                aie.end
            } {issue_token = true}

            aiex.dma_start_task(%ti)
            aiex.dma_start_task(%to)
            aiex.dma_await_task(%to)
        }
    }
}
