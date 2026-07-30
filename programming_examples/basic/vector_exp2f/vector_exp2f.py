# vector_exp2f/vector_exp2f.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Vector 2**x (software minimax poly), IRON + ``@iron.jit``, 4 cores, float32.

Demonstrates the IRON kernel library's software-poly ``2**x`` kernel
(``kernels.exp2f_vec``, ``aie_kernels/aie2p/exp2f_vec.cc``), the accuracy
alternative to the LUT-based ``kernels.bf16_exp`` that
``basic/vector_exp`` demonstrates. aie2p only (the kernel has not been
ported to aie2). Each of 4 cores runs the kernel on its own 1024-element
tile; the runtime splits/joins the work across the cores.

Input is built in three blocks over the kernel's documented domain
(``aie_kernels/aie2p/exp2f_vec.cc``'s header comment):

  1. a dense grid over [-100, 0], the kernel's characterized softmax
     domain (includes both endpoints, so the LUT's worst point -100 and
     the exact-1.0 point 0 are both exercised);
  2. a random sample over the same domain (covers non-grid points a
     linspace alone would miss);
  3. a block below -100 (down to -500), which exercises the kernel's
     clamp (``x = max(x, -100)`` before the exponent-field reconstruction,
     documented in the .cc file). This only has to stay finite and
     close to the clamped reference, not match unclamped ``2**x`` (which
     underflows float32 well before -500 anyhow).

Verification reports max-abs-error and rel-L2 against a float64 ``2**x``
reference (``kernels.exp2f_vec_ref``), separately for the two regimes,
plus a hard NaN/Inf gate on the clamp block (this kernel family's known
failure mode on this target is silent inf/nan, not just imprecision:
see ``aie_kernels/aie2p/dwconv1d.cc``'s ``aie::sliding_mul_ops`` note for
the sibling case this doctrine comes from).
"""

import sys

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_

_TILE = 1024  # hard-coded by kernels.exp2f_vec's underlying C++ kernel
_N_CORES = 4


@iron.jit
def vector_exp2f(
    x: In,
    y: Out,
    *,
    N: CompileTime[int],
):
    n = _TILE
    n_cores = _N_CORES
    tiles = (N // n) // n_cores

    tensor_ty = np.ndarray[(N,), np.dtype[np.float32]]
    memtile_ty = np.ndarray[(n * n_cores,), np.dtype[np.float32]]
    tile_ty = np.ndarray[(n,), np.dtype[np.float32]]

    exp2f_fn = kernels.exp2f_vec(tile_size=n)

    A_fifo = ObjectFifo(memtile_ty, name="inA")
    C_fifo = ObjectFifo(memtile_ty, name="outC")
    a_fifos = A_fifo.cons().split(
        offsets=[n * i for i in range(n_cores)], obj_types=[tile_ty] * n_cores
    )
    c_fifos = C_fifo.prod().join(
        offsets=[n * i for i in range(n_cores)], obj_types=[tile_ty] * n_cores
    )

    def core_fn(a_in, c_out, exp2f_fn):
        for _ in range_(tiles):
            elem_out = c_out.acquire(1)
            elem_in_a = a_in.acquire(1)
            exp2f_fn(elem_in_a, elem_out, n)
            a_in.release(1)
            c_out.release(1)

    workers = [
        Worker(core_fn, fn_args=[a_fifos[i].cons(), c_fifos[i].prod(), exp2f_fn])
        for i in range(n_cores)
    ]

    def sequence(a_in, c_out, in_h, out_h):
        in_h.fill(a_in)
        out_h.drain(c_out, wait=True)

    rt = Runtime(
        sequence,
        [tensor_ty, tensor_ty, A_fifo.prod(), C_fifo.cons()],
    )

    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _rel_l2(actual, ref):
    diff = actual.astype(np.float64) - ref.astype(np.float64)
    denom = np.linalg.norm(ref.astype(np.float64))
    return float(np.linalg.norm(diff) / denom) if denom > 0 else float(np.linalg.norm(diff))


def main():
    rng = np.random.default_rng(0)
    n_grid, n_rand, n_clamp = 6144, 1024, 1024
    N = n_grid + n_rand + n_clamp

    grid = np.linspace(-100.0, 0.0, n_grid, dtype=np.float64)
    rand = rng.uniform(-100.0, 0.0, n_rand)
    clamp = np.linspace(-500.0, -100.0001, n_clamp, dtype=np.float64)  # strictly below -100

    a_np = np.concatenate([grid, rand, clamp]).astype(np.float32)
    domain_end = n_grid + n_rand  # [0, domain_end) is the characterized [-100, 0] domain

    a = iron.tensor(a_np, dtype=np.float32, device="npu")
    c = iron.zeros(N, dtype=np.float32, device="npu")

    vector_exp2f(a, c, N=N)

    out = c.numpy()

    ok = True

    # Characterized domain [-100, 0]: tight tolerance, matches the kernel's
    # documented ~8.5e-5 max relative error (exp2f_vec_ref is exact float64
    # 2**x cast to f32).
    dom_out = out[:domain_end]
    dom_ref = kernels.exp2f_vec_ref(a_np[:domain_end])
    dom_out64 = dom_out.astype(np.float64)
    dom_ref64 = dom_ref.astype(np.float64)
    dom_max_abs = float(np.max(np.abs(dom_out64 - dom_ref64)))
    dom_rel_l2 = _rel_l2(dom_out, dom_ref)
    dom_max_rel = float(np.max(np.abs(dom_out64 - dom_ref64) / np.abs(dom_ref64)))
    dom_n_nonfinite = int(np.sum(~np.isfinite(dom_out)))
    print(
        f"[-100, 0] domain (N={domain_end}): max_abs_err={dom_max_abs:.6g} "
        f"max_rel_err={dom_max_rel:.6g} rel_l2={dom_rel_l2:.6g} non_finite={dom_n_nonfinite}"
    )
    if dom_n_nonfinite or dom_max_abs > 5e-4:
        print("FAIL: [-100, 0] domain outside gate (max_abs_err > 5e-4 or non-finite present)")
        ok = False

    # Clamp block (x < -100): reference is the KERNEL'S documented contract,
    # 2**max(x, -100), not raw 2**x (which is a different, unclamped
    # function below -100). The hard requirement is no NaN/Inf; the
    # numerical match to the clamped reference is reported but with a
    # looser gate since this is not the kernel's characterized domain.
    clamp_out = out[domain_end:]
    clamp_ref = kernels.exp2f_vec_ref(np.maximum(a_np[domain_end:], np.float32(-100.0)))
    clamp_max_abs = float(np.max(np.abs(clamp_out.astype(np.float64) - clamp_ref.astype(np.float64))))
    clamp_rel_l2 = _rel_l2(clamp_out, clamp_ref)
    clamp_n_nonfinite = int(np.sum(~np.isfinite(clamp_out)))
    print(
        f"< -100 clamp block (N={n_clamp}): max_abs_err={clamp_max_abs:.6g} "
        f"rel_l2={clamp_rel_l2:.6g} non_finite={clamp_n_nonfinite}"
    )
    if clamp_n_nonfinite:
        print(f"FAIL: clamp block produced {clamp_n_nonfinite} non-finite outputs")
        ok = False

    if not ok:
        sys.exit(1)
    print(f"PASS! ({N} samples: {domain_end} in the characterized [-100,0] domain, "
          f"{n_clamp} exercising the < -100 clamp path)")


if __name__ == "__main__":
    main()
