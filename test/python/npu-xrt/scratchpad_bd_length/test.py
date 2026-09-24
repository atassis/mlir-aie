# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Test for DMA transfer-length extension via length_parameter (IRON flow).
# @out_len selects how many of a fixed 64-element passthrough object reach
# the host (base 16 + 16*k, see aie_design.py).
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings, xrt_python_ctrl_scratchpad_bo
#
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: %aiecc -v --get-full-elf --dynamic-objFifos --get-scratchpad-parameters aie.mlir
# RUN: %run_on_npu2% %pytest %s

import numpy as np
import pytest
import pyxrt

import aie.iron as iron
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
    ParameterScratchpad,
)

N = 64
SENTINEL = np.int32(-1)


@pytest.fixture(scope="module")
def kernel_setup():
    runtime = XRTHostRuntime()
    device = runtime._device
    elf = pyxrt.elf("aie.elf")
    context = pyxrt.hw_context(device, elf)
    kernel = pyxrt.ext.kernel(context, "test:sequence")

    in_tensor = iron.arange(N, dtype=np.int32, device="cpu")
    out_tensor = iron.tensor((N,), dtype=np.int32, device="cpu")

    run = pyxrt.run(kernel)
    run.set_arg(0, in_tensor.buffer_object())
    run.set_arg(1, out_tensor.buffer_object())

    params = ParameterScratchpad(run, "params.txt")
    return run, params, in_tensor, out_tensor


@pytest.mark.parametrize("k,want", [(0, 16), (1, 32), (2, 48), (3, 64)])
def test_out_len(kernel_setup, k, want):
    run, params, in_tensor, out_tensor = kernel_setup

    out_tensor.data.fill(SENTINEL)
    out_tensor.to("npu")
    in_tensor.to("npu")

    params.write("out_len", np.int32(k))
    params.sync()

    run.start()
    run.wait2()

    out_tensor.to("cpu")
    result = out_tensor.numpy()
    expected_prefix = in_tensor.numpy()[:want]
    assert result[:want].tolist() == expected_prefix.tolist()
    assert np.all(result[want:] == SENTINEL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
