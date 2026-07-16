#!/usr/bin/env python3
"""Verify + TIME the query-tiled attention conveyor (N_QT tiles streamed, k/V resident).
Usage: run_conveyor_attn.py [conv|mono]  (default conv). Verifies ctx vs numpy + reports per-dispatch ms."""
import os, sys, time, numpy as np, pyxrt
from ml_dtypes import bfloat16

EX = os.path.join(os.path.dirname(__file__), "build")
which = sys.argv[1] if len(sys.argv) > 1 else "conv"
TQ = int(os.environ.get("ATTN_TQ", 8))
T = int(os.environ.get("ATTN_T", 64))
DK = int(os.environ.get("ATTN_DK", 64))
N_QT = int(os.environ.get("ATTN_NQT", 16))
SCALE = float(os.environ.get("ATTN_SCALE", 1.0 / (DK ** 0.5)))
NQ = N_QT * TQ

RELPOS = os.environ.get("ATTN_RELPOS", "0") == "1"
P = 2 * T - 1
rng = np.random.default_rng(0)
q = rng.standard_normal((NQ, DK)).astype(bfloat16)
k = rng.standard_normal((T, DK)).astype(bfloat16)
v = rng.standard_normal((T, DK)).astype(bfloat16)
if RELPOS:
    # REAL-DIMS relpos (BD-in-belt): host computes BD = q.p^T THEN rel_shift -> BD_shifted[NQ,T] bf16,
    # packed after q per tile in the query belt. scores[i,j] = (q.k^T + BD_shifted)[i,j] * SCALE.
    # rel_shift: BD_shifted[i,j] = BD[i, (T-1-i)+j] (global row i; here NQ rows starting at 0).
    p = rng.standard_normal((P, DK)).astype(bfloat16)
    ac_raw = q.astype(np.float32) @ k.astype(np.float32).T          # [NQ,T]
    BD = q.astype(np.float32) @ p.astype(np.float32).T              # [NQ,P]
    BD_shifted = np.empty((NQ, T), np.float32)
    for i in range(NQ):
        base = (T - 1 - i) if i < T else 0
        BD_shifted[i] = BD[i, base:base + T]
    bd_bf = BD_shifted.astype(bfloat16)                             # kernel adds bf16 BD_shifted
    scores = (ac_raw + bd_bf.astype(np.float32)) * SCALE
    # query belt per tile = [q_tile (TQ*DK) || BD_shifted_tile (TQ*T)] bf16
    q_belt = np.concatenate([q.reshape(N_QT, TQ * DK), bd_bf.reshape(N_QT, TQ * T)], axis=1).reshape(-1)
    kpack = k
else:
    scores = SCALE * (q.astype(np.float32) @ k.astype(np.float32).T)
    q_belt = q
    kpack = k
sc = scores - scores.max(axis=1, keepdims=True)
e = np.exp(sc); probs = e / e.sum(axis=1, keepdims=True)
ctx_ref = probs @ v.astype(np.float32)

instr = np.fromfile(f"{EX}/{which}.insts", dtype=np.uint32)
xclbin = pyxrt.xclbin(f"{EX}/{which}.xclbin")
kname = xclbin.get_kernels()[0].get_name()
d = pyxrt.device(0); d.register_xclbin(xclbin)
hw = pyxrt.hw_context(d, xclbin.get_uuid())
kern = pyxrt.kernel(hw, kname)
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

qb = q_belt.reshape(-1).view(np.uint16); kb = kpack.reshape(-1).view(np.uint16); vb = v.reshape(-1).view(np.uint16)
bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
bo_q = pyxrt.bo(d, qb.nbytes, pyxrt.bo.host_only, kern.group_id(3))
bo_k = pyxrt.bo(d, kb.nbytes, pyxrt.bo.host_only, kern.group_id(4))
bo_v = pyxrt.bo(d, vb.nbytes, pyxrt.bo.host_only, kern.group_id(5))
bo_c = pyxrt.bo(d, NQ * DK * 2, pyxrt.bo.host_only, kern.group_id(6))
bo_instr.write(instr.tobytes(), 0); bo_instr.sync(TO)
bo_q.write(qb.tobytes(), 0); bo_q.sync(TO)
bo_k.write(kb.tobytes(), 0); bo_k.sync(TO)
bo_v.write(vb.tobytes(), 0); bo_v.sync(TO)

def once():
    r = kern(3, bo_instr, instr.size, bo_q, bo_k, bo_v, bo_c); r.wait()

once()
bo_c.sync(FROM)
dev_ctx = np.frombuffer(bo_c.read(NQ * DK * 2, 0), dtype=np.uint16).view(bfloat16).astype(np.float32).reshape(NQ, DK)
rel = np.linalg.norm(dev_ctx - ctx_ref) / np.linalg.norm(ctx_ref)
pt = [np.linalg.norm(dev_ctx[t*TQ:(t+1)*TQ]-ctx_ref[t*TQ:(t+1)*TQ])/max(np.linalg.norm(ctx_ref[t*TQ:(t+1)*TQ]),1e-9) for t in range(N_QT)]
print("  per-tile rel-err:", " ".join(f"{x:.2f}" for x in pt))

iters = 200
t0 = time.perf_counter()
for _ in range(iters):
    once()
dt = (time.perf_counter() - t0) / iters * 1e3

print(f"[{which}] kernel='{kname}'  rel-L2={rel:.4e}  {'PASS' if rel < 3e-2 else 'FAIL'}")
print(f"[{which}] {N_QT} query tiles/dispatch  ->  {dt:.4f} ms/dispatch  ({dt/N_QT*1e3:.2f} us/query-tile)")
