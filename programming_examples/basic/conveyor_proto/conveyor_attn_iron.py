# REAL attention CONVEYOR, QUERY-TILED for pipeline OVERLAP (the latency-hiding win).
#   DDR{q x N_QT tiles, k, v} -> [A: scale*Q.K^T] --ac--> [B: softmax] --probs--> [C: probs.V] -> DDR{ctx}
# k and V are held RESIDENT (acquired once, reused across all N_QT query tiles) -- the conveyor's
# elegance: each stage keeps its weights, streams query tiles through. With depth-2 belts the 3 workers
# RUN CONCURRENTLY: while B softmaxes tile q, A computes q+1 and C finishes q-1 (up to 3 tiles in flight).
# VARIANT=mono builds the same math on ONE tile (all 3 stages sequential per tile) for the A/B baseline.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sys
import argparse
import numpy as np

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.iron.controlflow import range_

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = np.float16

import os
# Dims overridable via env (real Parakeet dims: TQ=8 T=176 DK=128 N_QT=22). Defaults = validated tiny proto.
TQ = int(os.environ.get("ATTN_TQ", 8))
T = int(os.environ.get("ATTN_T", 64))
DK = int(os.environ.get("ATTN_DK", 64))
N_QT = int(os.environ.get("ATTN_NQT", 16))  # query tiles streamed through the pipeline
N_HEADS = int(os.environ.get("ATTN_HEADS", 1))  # data-parallel heads, one 3-tile conveyor per column


P = 2 * T - 1  # relative-position length (NeMo/Parakeet rel-pos)


def build(dev, mono=False, TRIVIAL=False, relpos=False):
    q_ty = np.ndarray[(TQ * DK,), np.dtype[bfloat16]]      # one query tile (fifo object)
    # relpos (real-dims): query belt carries q[TQ,DK] then host-precomputed rel_shifted BD_shifted[TQ,T],
    # both bf16, in one object -> stage A does AC on-chip + adds BD_shifted (no p resident, no row_off).
    qbd_ty = np.ndarray[(TQ * DK + TQ * T,), np.dtype[bfloat16]]
    k_ty = np.ndarray[(T * DK,), np.dtype[bfloat16]]
    v_ty = np.ndarray[(T * DK,), np.dtype[bfloat16]]
    ac_ty = np.ndarray[(TQ * T,), np.dtype[np.float32]]
    probs_ty = np.ndarray[(TQ * T,), np.dtype[bfloat16]]
    ctx_ty = np.ndarray[(TQ * DK,), np.dtype[bfloat16]]    # one ctx tile (fifo object)
    # RUNTIME (L3) arg types = the WHOLE streamed buffer (N_QT tiles for q/ctx); the shim streams it
    # into the [TQ,*] fifo objects as N_QT blocks. (The fifo OBJECT is one tile; the SEQUENCE ARG is
    # the whole buffer -- mixing these up made N_QT>1 read only the first tile.)
    QELEM = TQ * DK + (TQ * T if relpos else 0)   # per-tile query-belt element count (q [+ BD_shifted])
    q_full_ty = np.ndarray[(N_QT * QELEM,), np.dtype[bfloat16]]
    ctx_full_ty = np.ndarray[(N_QT * TQ * DK,), np.dtype[bfloat16]]

    sfx = "_t" if TRIVIAL else ""   # TRIVIAL: same structure, trivial copy kernels (isolate race)
    # BISECT: TRIVIAL=2 -> only softmax trivial (scores+ctx real); TRIVIAL=3 -> only scores trivial.
    sc_sfx = "_t" if TRIVIAL in (1, 3) else ""
    sm_sfx = "_t" if TRIVIAL in (1, 2) else ""
    cx_sfx = "_t" if TRIVIAL == 1 else ""
    qbelt_ty = qbd_ty if relpos else q_ty  # query belt object: q+BD_shifted (relpos) or q (plain)
    scores = (Kernel("stage_scores_relpos_bd", "kernels.a", [qbd_ty, k_ty, ac_ty]) if relpos
              else Kernel("stage_scores" + sc_sfx, "kernels.a", [q_ty, k_ty, ac_ty]))
    softmax = Kernel("stage_softmax" + sm_sfx, "kernels.a", [ac_ty, probs_ty])
    ctx_k = Kernel("stage_ctx" + cx_sfx, "kernels.a", [probs_ty, v_ty, ctx_ty])

    # relpos: the query belt carries q+BD_shifted (~2x bigger) -> depth-1 to fit stage A's L1 alongside
    # the 44 KB resident k + the f32 ac belt (the A->B ac belt keeps depth-2 for pipeline overlap).
    of_q = ObjectFifo(qbelt_ty, name="q", depth=1 if relpos else 2)
    of_ctx = ObjectFifo(ctx_ty, name="ctx", depth=2)
    # ADVANCING taps for q (in) / ctx (out): outer dim N_QT strides by the per-tile element count.
    # relpos: one flat QELEM block per tile (q||BD_shifted contiguous). plain: [TQ,DK].
    q_tap = (TensorAccessPattern([N_QT * QELEM], 0, [N_QT, 1, 1, QELEM], [QELEM, 0, 0, 1]) if relpos
             else TensorAccessPattern([N_QT * TQ * DK], 0, [N_QT, 1, TQ, DK], [TQ * DK, 0, DK, 1]))
    ctx_tap = TensorAccessPattern([N_QT * TQ * DK], 0, [N_QT, 1, TQ, DK], [TQ * DK, 0, DK, 1])
    # stride-0 replay taps (deliver the read-only weight N_QT times, one per query tile).
    replay_tap = TensorAccessPattern([T * DK], 0, [N_QT, 1, T, DK], [0, 0, DK, 1])

    rt = Runtime()
    if mono:
        # MONOLITH baseline: ONE tile, all 3 ops per query tile; q + packed kv (2 inputs = channel budget).
        kv_ty = np.ndarray[(2 * T * DK,), np.dtype[bfloat16]]
        mono_k = Kernel("stage_mono", "kernels.a", [q_ty, kv_ty, ctx_ty])
        of_kv = ObjectFifo(kv_ty, name="kv", depth=2)
        kv_replay = TensorAccessPattern([2 * T * DK], 0, [N_QT, 1, 2 * T, DK], [0, 0, DK, 1])

        def mono_fn(f_q, f_kv, f_ctx, k_mono):
            for _ in range_(N_QT):
                eq = f_q.acquire(1); ekv = f_kv.acquire(1); ec = f_ctx.acquire(1)
                k_mono(eq, ekv, ec)
                f_q.release(1); f_kv.release(1); f_ctx.release(1)

        w = Worker(mono_fn, [of_q.cons(), of_kv.cons(), of_ctx.prod(), mono_k])
        with rt.sequence(q_full_ty, kv_ty, ctx_full_ty) as (Q, KV, CTX):
            rt.start(w)
            rt.fill(of_q.prod(), Q, tap=q_tap)
            rt.fill(of_kv.prod(), KV, tap=kv_replay)
            rt.drain(of_ctx.cons(), CTX, tap=ctx_tap, wait=True)
    else:
        # k, V are read-only weights. At real dims (T*DK bf16 = 44 KB) a depth-2 weight fifo blows the
        # 64 KB L1, so depth-1. Structure = the validated per-tile-acquire + stride-0 replay tap (the
        # tiny-dims proven path), only single-buffered. (True acquire-once residency deadlocked at
        # depth-1 -- deferred; this re-streams weights on-chip but is the fair, known-good dataflow.)
        # MULTI-HEAD: replicate the 3-tile conveyor per head; place-tiles assigns each head its own
        # column (data-parallel heads, no cross-head data). Per-head fifos (unique names).
        H = N_HEADS
        of_qh = [ObjectFifo(qbelt_ty, name=f"q{h}", depth=1 if relpos else 2) for h in range(H)]
        of_kh = [ObjectFifo(k_ty, name=f"k{h}", depth=1) for h in range(H)]
        of_vh = [ObjectFifo(v_ty, name=f"v{h}", depth=1) for h in range(H)]
        of_ach = [ObjectFifo(ac_ty, name=f"ac{h}", depth=2) for h in range(H)]
        of_ph = [ObjectFifo(probs_ty, name=f"probs{h}", depth=2) for h in range(H)]
        of_ctxh = [ObjectFifo(ctx_ty, name=f"ctx{h}", depth=2) for h in range(H)]

        def stage_a(f_q, f_k, f_ac, k_sc):
            for _ in range_(N_QT):
                eq = f_q.acquire(1); ek = f_k.acquire(1); eac = f_ac.acquire(1)
                k_sc(eq, ek, eac)
                f_q.release(1); f_k.release(1); f_ac.release(1)

        def stage_b(f_ac, f_probs, k_sm):
            for _ in range_(N_QT):
                eac = f_ac.acquire(1); ep = f_probs.acquire(1)
                k_sm(eac, ep)
                f_ac.release(1); f_probs.release(1)

        def stage_c(f_probs, f_v, f_ctx, k_cx):
            for _ in range_(N_QT):
                ep = f_probs.acquire(1); ev = f_v.acquire(1); ec = f_ctx.acquire(1)
                k_cx(ep, ev, ec)
                f_probs.release(1); f_v.release(1); f_ctx.release(1)

        # stage_b (softmax) holds `float srow[T]` on the AIE stack -> bump its stack_size (real T
        # overflows the 1 KB default -> silent hang). stage A (relpos BD-in-belt) / C need no bump.
        wl = []
        for h in range(H):
            wl += [Worker(stage_a, [of_qh[h].cons(), of_kh[h].cons(), of_ach[h].prod(), scores]),
                   Worker(stage_b, [of_ach[h].cons(), of_ph[h].prod(), softmax], stack_size=0x1000),
                   Worker(stage_c, [of_ph[h].cons(), of_vh[h].cons(), of_ctxh[h].prod(), ctx_k])]

        # ONE big buffer per role [H * per-head]; per-head fills index in with an offset tap. The flat
        # [N_QT,1,1,QELEM] query tap works for plain (QELEM=TQ*DK) and relpos (QELEM=TQ*DK+TQ*T) alike.
        QT, KT, VT, CT = N_QT * QELEM, T * DK, T * DK, N_QT * TQ * DK
        q_all_ty = np.ndarray[(H * QT,), np.dtype[bfloat16]]
        k_all_ty = np.ndarray[(H * KT,), np.dtype[bfloat16]]
        v_all_ty = np.ndarray[(H * VT,), np.dtype[bfloat16]]
        c_all_ty = np.ndarray[(H * CT,), np.dtype[bfloat16]]
        with rt.sequence(q_all_ty, k_all_ty, v_all_ty, c_all_ty) as (Q, K, V, CTX):
            rt.start(*wl)
            for h in range(H):
                qh = TensorAccessPattern([H * QT], h * QT, [N_QT, 1, 1, QELEM], [QELEM, 0, 0, 1])
                kh = TensorAccessPattern([H * KT], h * KT, [N_QT, 1, T, DK], [0, 0, DK, 1])
                vh = TensorAccessPattern([H * VT], h * VT, [N_QT, 1, T, DK], [0, 0, DK, 1])
                ch = TensorAccessPattern([H * CT], h * CT, [N_QT, 1, TQ, DK], [TQ * DK, 0, DK, 1])
                rt.fill(of_qh[h].prod(), Q, tap=qh)
                rt.fill(of_kh[h].prod(), K, tap=kh)
                rt.fill(of_vh[h].prod(), V, tap=vh)
                rt.drain(of_ctxh[h].cons(), CTX, tap=ch, wait=True)

    return Program(dev, rt).resolve_program()


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dev", required=True, dest="device")
ap.add_argument("--mono", action="store_true", help="single-tile baseline (all 3 stages on 1 tile)")
ap.add_argument("--trivial", type=int, default=0, help="0=real; 1=all trivial; 2=softmax trivial; 3=scores trivial")
ap.add_argument("--relpos", action="store_true", help="fused relpos scores stage (on-chip AC+BD+rel_shift)")
opts = ap.parse_args(sys.argv[1:])
dev = NPU2() if opts.device == "npu2" else NPU1()
print(build(dev, mono=opts.mono, TRIVIAL=opts.trivial, relpos=opts.relpos))
