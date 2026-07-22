# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_

N = 256
TILE = 16
N_TILES = N // TILE


def build_design():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            vector_ty = np.ndarray[(N,), np.dtype[np.int32]]
            tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]

            shim = tile(0, 0)
            mem = tile(0, 1)
            compute = tile(0, 2)

            weights = object_fifo(
                "weights",
                mem,
                compute,
                1,
                vector_ty,
                initValues=[np.arange(1, N + 1, dtype=np.int32)],
            )
            inputs = object_fifo("inputs", shim, compute, 2, tile_ty)
            outputs = object_fifo("outputs", compute, shim, 2, tile_ty)

            @core(compute)
            def core_body():
                for _ in range_(sys.maxsize):
                    weight = weights.acquire(ObjectFifoPort.Consume, 1)
                    for tile_index in range_(N_TILES):
                        input_tile = inputs.acquire(ObjectFifoPort.Consume, 1)
                        output_tile = outputs.acquire(ObjectFifoPort.Produce, 1)
                        for element in range_(TILE):
                            weight_index = tile_index * TILE + element
                            output_tile[element] = input_tile[element] + weight[weight_index]
                        inputs.release(ObjectFifoPort.Consume, 1)
                        outputs.release(ObjectFifoPort.Produce, 1)
                    weights.release(ObjectFifoPort.Consume, 1)

            @runtime_sequence(vector_ty, vector_ty)
            def sequence(input_vector, output_vector):
                dma_channel_reset_for("weights")
                npu_dma_memcpy_nd(
                    metadata=inputs,
                    bd_id=2,
                    mem=input_vector,
                    sizes=[1, 1, 1, N],
                )
                npu_dma_memcpy_nd(
                    metadata=outputs,
                    bd_id=0,
                    mem=output_vector,
                    sizes=[1, 1, 1, N],
                )
                dma_wait(outputs)

        if not ctx.module.operation.verify():
            raise RuntimeError("generated module failed verification")
        print(ctx.module)


if __name__ == "__main__":
    build_design()
