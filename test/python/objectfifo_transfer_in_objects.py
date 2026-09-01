# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A fill()/drain() access pattern is expressed in the runtime buffer's element
# type, while what the stream carries is objectFIFO objects.  Where the two
# element types differ - a host buffer viewing the bytes of a differently-typed
# stream, as ml/mobilenet does - every call site converts by hand.  objects=
# takes the count and does the conversion from the fifo's own geometry.

# RUN: %python %s | FileCheck %s

import aie.iron as iron
import numpy as np
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import AnyShimTile, from_name

iron.set_current_device(from_name("npu2", n_cols=1))

LINE = 256  # elements per object
N = 4 * LINE


def build(name, fifo_dtype, buf_dtype, buf_elems, **fill_kwargs):
    line_ty = np.ndarray[(LINE,), np.dtype[fifo_dtype]]
    buf_ty = np.ndarray[(buf_elems,), np.dtype[buf_dtype]]

    of_in = ObjectFifo(line_ty, name=f"in_{name}")
    of_out = of_in.cons().forward(name=f"out_{name}")

    def sequence(a, c, in_h, out_h):
        in_h.fill(a, **fill_kwargs)
        out_h.drain(c, wait=True, **fill_kwargs)

    rt = Runtime(
        sequence,
        [buf_ty, buf_ty, of_in.prod(tile=AnyShimTile), of_out.cons(tile=AnyShimTile)],
    )
    return str(Program(iron.get_current_device(), rt).resolve_program())


# Same element type: two objects of 256 i32 is 512 buffer elements at offset 512.
# CHECK-LABEL: TEST: objects_of_the_same_element_type
# CHECK: aie.dma_bd(%{{.*}} : memref<1024xi32> offset = 512 len = 512
print("TEST: objects_of_the_same_element_type")
print(build("same", np.int32, np.int32, N, objects=2, object_offset=2))

# A wider host view: the fifo's object is 256 i8 = 64 i32, so one object at
# object offset 3 is 64 elements at element offset 192 of the i32 buffer. This
# is the conversion ml/mobilenet writes as `byte_offset // 4` by hand.
# CHECK-LABEL: TEST: objects_through_a_wider_host_view
# CHECK: aie.dma_bd(%{{.*}} : memref<256xi32> offset = 192 len = 64
print("TEST: objects_through_a_wider_host_view")
print(build("view", np.int8, np.int32, N // 4, objects=1, object_offset=3))

# Running off the end of the buffer is caught here rather than at the BD.
# CHECK-LABEL: TEST: past_the_end_of_the_buffer_is_rejected
# CHECK: 2 object(s) at object offset 3 runs to element 1280 of a buffer holding 1024
print("TEST: past_the_end_of_the_buffer_is_rejected")
try:
    build("over", np.int32, np.int32, N, objects=2, object_offset=3)
    raise AssertionError("expected an out-of-range object range to be rejected")
except ValueError as e:
    print(e)

# objects= is a third way to say what tap and sizes/strides already say, so it
# does not combine with either.
# CHECK-LABEL: TEST: objects_does_not_combine_with_an_explicit_pattern
# CHECK: not more than one
print("TEST: objects_does_not_combine_with_an_explicit_pattern")
try:
    build("both", np.int32, np.int32, N, objects=1, sizes=[1, 1, 1, 256])
    raise AssertionError("expected objects= plus sizes= to be rejected")
except ValueError as e:
    print(e)
