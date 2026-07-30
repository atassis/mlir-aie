# ln_affine_cast/ln_affine_cast.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Fused row-wise LayerNorm + affine + f32 -> bf16 cast — IRON API + ``@iron.jit``.

NPU2-only: the underlying ``ln_affine_cast.cc`` kernel lives under
``aie_kernels/aie2p/`` and has no aie2 counterpart.

Eight cores each normalize ``sequence_length // 8`` rows of ``embedding_dim``
f32 values, apply a real per-column affine (gamma, beta), and narrow the
result to bf16 in the same dispatch:

    out = ((x - mean(x)) / sqrt(var(x) + eps)) * gamma + beta -> bf16

gamma/beta are the same for every row and are packed into one
``[2 * embedding_dim]`` buffer (gamma then beta) that every core loads once,
before its row loop, rather than per row.

Structurally this mirrors ``ml/norm``'s row-split ``@iron.jit`` design (same
8-core-per-row shape), extended two ways: the input and output tiles have
different dtypes (f32 in, bf16 out -- as in ``ml/cast_f32_bf16``), and there
is a third, per-core-constant parameter tensor (gamma/beta) alongside the
per-row data tensor. Neither fits ``transform_parallel``/
``transform_parallel_binary`` (uniform dtype, two tensors only), hence the
explicit ``ObjectFifo``/``Worker``/``Runtime`` wiring below.
"""

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D
from aie.utils import config
from aie.utils.hostruntime.argparse import device_from_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

_KERNEL_DIR = Path(__file__).resolve().parents[3] / "aie_kernels/aie2p"


def _ln_affine_cast_extern(chunk_in_ty, chunk_gb_ty, chunk_out_ty):
    return ExternalFunction(
        "ln_affine_cast_row",
        source_file=str(_KERNEL_DIR / "ln_affine_cast.cc"),
        arg_types=[chunk_in_ty, chunk_gb_ty, chunk_out_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )


@iron.jit
def ln_affine_cast(
    a_in: In,
    gb_in: In,
    c_out: Out,
    *,
    sequence_length: CompileTime[int] = 64,
    embedding_dim: CompileTime[int] = 1024,
):
    n_cores = 8
    vec = 16  # ln_affine_cast_row<16> vectorizes cols by 16

    if sequence_length % n_cores != 0:
        raise ValueError(
            f"sequence_length ({sequence_length}) must be a multiple of {n_cores}"
        )
    if embedding_dim % vec != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be a multiple of {vec}")

    rows_per_core = sequence_length // n_cores

    in_ty = np.ndarray[(sequence_length, embedding_dim), np.dtype[np.float32]]
    gb_ty = np.ndarray[(1, 2 * embedding_dim), np.dtype[np.float32]]
    out_ty = np.ndarray[(sequence_length, embedding_dim), np.dtype[bfloat16]]

    chunk_in_ty = np.ndarray[(embedding_dim,), np.dtype[np.float32]]
    chunk_gb_ty = np.ndarray[(2 * embedding_dim,), np.dtype[np.float32]]
    chunk_out_ty = np.ndarray[(embedding_dim,), np.dtype[bfloat16]]

    of_ins = [ObjectFifo(chunk_in_ty, name=f"in_{i}") for i in range(n_cores)]
    # depth=1: gamma/beta are constant for the whole row loop (acquired once,
    # never released mid-loop below), so there is nothing to double-buffer --
    # and at depth=2 the [2 * embedding_dim] f32 buffer alone can exceed a
    # tile's 64 KB L1 budget once the row in/out buffers are added.
    of_gbs = [ObjectFifo(chunk_gb_ty, name=f"gb_{i}", depth=1) for i in range(n_cores)]
    of_outs = [ObjectFifo(chunk_out_ty, name=f"out_{i}") for i in range(n_cores)]

    ln_affine_cast_fn = _ln_affine_cast_extern(chunk_in_ty, chunk_gb_ty, chunk_out_ty)

    def core_fn(of_in, of_gb, of_out, kernel):
        # gamma/beta are constant for every row this core sees: acquire once,
        # hold for the whole row loop, release after the last row.
        elem_gb = of_gb.acquire(1)
        for _ in range_(rows_per_core):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kernel(elem_in, elem_gb, elem_out, embedding_dim)
            of_in.release(1)
            of_out.release(1)
        of_gb.release(1)

    workers = [
        Worker(
            core_fn,
            [of_ins[i].cons(), of_gbs[i].cons(), of_outs[i].prod(), ln_affine_cast_fn],
        )
        for i in range(n_cores)
    ]

    taps = TensorTiler2D.simple_tiler(
        (sequence_length, embedding_dim), (rows_per_core, embedding_dim)
    )
    # One tile == the whole gb tensor; every core loads the SAME full range.
    gb_taps = TensorTiler2D.simple_tiler((1, 2 * embedding_dim))

    def sequence(a, gb, c, in_prods, gb_prods, out_conses):
        for i in range(n_cores):
            in_prods[i].fill(a, taps[i])
        for i in range(n_cores):
            gb_prods[i].fill(gb, gb_taps[0])
        for i in range(n_cores):
            out_conses[i].drain(c, taps[i], wait=True)

    rt = Runtime(
        sequence,
        [
            in_ty,
            gb_ty,
            out_ty,
            [of_ins[i].prod() for i in range(n_cores)],
            [of_gbs[i].prod() for i in range(n_cores)],
            [of_outs[i].cons() for i in range(n_cores)],
        ],
    )

    device = iron.get_current_device()
    return Program(device, rt, workers=workers).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE LN + affine + cast")
    add_compile_args(p, with_elf=True)
    p.add_argument("-s", "--sequence_length", type=int, default=64, help="rows")
    p.add_argument("-e", "--embedding_dim", type=int, default=1024, help="cols per row")
    return p


def _compile_kwargs(opts):
    return dict(sequence_length=opts.sequence_length, embedding_dim=opts.embedding_dim)


def _ln_affine_reference(x_np, gamma_np, beta_np):
    eps = 1e-5
    x32 = x_np.astype(np.float32)
    mean = x32.mean(axis=1, keepdims=True)
    var = ((x32 - mean) ** 2).mean(axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    y = (x32 - mean) * inv_std * gamma_np + beta_np
    return y.astype(bfloat16)


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    rows, cols = opts.sequence_length, opts.embedding_dim

    a_np = rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(np.float32)
    gamma_np = rng.uniform(0.5, 1.5, size=(cols,)).astype(np.float32)
    beta_np = rng.uniform(-0.5, 0.5, size=(cols,)).astype(np.float32)
    gb_np = np.concatenate([gamma_np, beta_np]).reshape(1, 2 * cols)

    a_t = iron.tensor(a_np, dtype=np.float32, device="npu")
    gb_t = iron.tensor(gb_np, dtype=np.float32, device="npu")
    c_t = iron.zeros(rows * cols, dtype=bfloat16, device="npu")

    ln_affine_cast(a_t, gb_t, c_t, **_compile_kwargs(opts))

    expected = _ln_affine_reference(a_np, gamma_np, beta_np)
    out = c_t.numpy().reshape(rows, cols)
    # Two-pass f32 reductions on-chip vs an exact f32 numpy reference: not
    # bit-exact (the AIE reduction path is not IEEE round-to-nearest at every
    # step), so this checks a tolerance, not equality -- same shape of check
    # as ml/norm's op=layer path.
    assert_pass(out, expected, atol=0.1, fail_msg="ln_affine_cast output mismatch")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        ln_affine_cast,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
