# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# IRON design: DMA transfer-length extension via length_parameter (see test.py).
#
# Usage:
#   python3 aie_design.py > aie.mlir

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2Col1
from aie.iron.scratchpad_parameter import ScratchpadParameter
from aie.dialects.aiex import npu_load_pdi
from aie.helpers.taplib import TensorAccessPattern


def design():
    device_name = "test"

    tile_ty = np.ndarray[(64,), np.dtype[np.int32]]
    io_ty = np.ndarray[(64,), np.dtype[np.int32]]

    # Parameter: element count (in units of 16) appended to the output's
    # 16-element base transfer length.
    out_len = ScratchpadParameter("out_len", np.int32)

    of_in = ObjectFifo(tile_ty, name="objfifo_in")
    of_out = ObjectFifo(tile_ty, name="objfifo_out")

    def core_fn(of_in, of_out):
        in_elem = of_in.acquire(1)
        out_elem = of_out.acquire(1)
        for i in range(64):
            out_elem[i] = in_elem[i]
        of_in.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [of_in.cons(), of_out.prod()],
        while_true=False,
    )

    def sequence(in_tensor, out_tensor, in_h, out_h):
        npu_load_pdi(device_ref="empty")
        npu_load_pdi(device_ref=device_name)

        in_h.fill(in_tensor, wait=False)

        # length_parameter extends the 16-element base transfer by 16 *
        # StateTable[@out_len] elements out of the fixed 64-element object.
        out_tap = TensorAccessPattern(
            (64,), offset=0, sizes=[1, 1, 1, 16], strides=[0, 0, 0, 1]
        )
        out_h.drain(
            out_tensor,
            tap=out_tap,
            wait=True,
            length_parameter=out_len,
            length_granule=16,
        )

    rt = Runtime(sequence, [io_ty, io_ty, of_in.prod(), of_out.cons()])

    module = Program(NPU2Col1(), rt, workers=[worker]).resolve_program(
        device_name=device_name
    )

    mlir_text = str(module)
    empty_device = "  aie.device(npu2) @empty { }\n"
    mlir_text = mlir_text.replace("module {\n", "module {\n" + empty_device, 1)
    return mlir_text


mlir_text = design()
print(mlir_text)
