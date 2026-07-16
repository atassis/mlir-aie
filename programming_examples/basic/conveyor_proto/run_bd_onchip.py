#!/usr/bin/env python3
"""H=1 N_QT=1 arithmetic gate for the BD-ON-CHIP 4th stage.
5-BO ABI: kern(3, instr, n, qpv, p, k, v, ctx). Feeds qpv=q_pass||qv + resident p/k/v (NO host BD
precompute); the on-chip BD tile computes BD = rel_shift((q+bias_v)@p^T). Verifies ctx vs a host
relpos-MHA golden (bf16-faithful). Build first: BDON=1 ATTN_T=64 ATTN_NQT=1 ATTN_HEADS=1 make ...
Env BD_SPLIT selects the belt carriage in the golden (must match the KFLAGS the kernel was built with)."""
import os, numpy as np, pyxrt
from ml_dtypes import bfloat16

EX = os.path.join(os.path.dirname(__file__), "build")
TQ = int(os.environ.get("ATTN_TQ", 8))
T = int(os.environ.get("ATTN_T", 64))
DK = int(os.environ.get("ATTN_DK", 128))
N_QT = int(os.environ.get("ATTN_NQT", 1))
SCALE = float(os.environ.get("ATTN_SCALE", 1.0 / (DK ** 0.5)))
BD_SPLIT = int(os.environ.get("BD_SPLIT", 0))
NQ = N_QT * TQ
P = 2 * T - 1

def bf(x):  # round f32 -> bf16 -> f32
    return x.astype(bfloat16).astype(np.float32)

rng = np.random.default_rng(0)
q = rng.standard_normal((NQ, DK)).astype(np.float32)
k = bf(rng.standard_normal((T, DK)))
v = bf(rng.standard_normal((T, DK)))
p = bf(rng.standard_normal((P, DK)))
bias_v = bf(0.1 * rng.standard_normal((DK,)))

# scale q so scores spread (non-degenerate softmax)
std = (q @ k.T).std() + 1e-6
q = bf(q / std)
qv = bf(q + bias_v)                                   # qv = q + pos_bias_v (bf16 belt)

# ---- host golden (f32 dots on bf16 operands; BD carried bf16 hi[+lo] like the belt) ----
AC = q @ k.T                                          # [NQ,T]
BD = qv @ p.T                                         # [NQ,P]
BD_sh = np.stack([BD[i, (T - 1 - i):(T - 1 - i) + T] for i in range(NQ)])  # rel_shift, q0=0
BD_hi = bf(BD_sh)
BD_car = BD_hi + (bf(BD_sh - BD_hi) if BD_SPLIT else 0.0)
scores = (AC + BD_car) * SCALE
e = np.exp(scores - scores.max(1, keepdims=True))
ctx_ref = (e / e.sum(1, keepdims=True)) @ v           # [NQ,DK]

# ---- belts ----
qpv = np.concatenate([q.reshape(N_QT, TQ * DK), qv.reshape(N_QT, TQ * DK)], axis=1).reshape(-1)
qpv_b = qpv.astype(bfloat16).view(np.uint16)
p_b = p.astype(bfloat16).view(np.uint16)
k_b = k.astype(bfloat16).view(np.uint16)
v_b = v.astype(bfloat16).view(np.uint16)

# ---- device ----
instr = np.fromfile(f"{EX}/insts.bin", dtype=np.uint32)
xclbin = pyxrt.xclbin(f"{EX}/final.xclbin")
kname = xclbin.get_kernels()[0].get_name()
d = pyxrt.device(0); d.register_xclbin(xclbin)
hw = pyxrt.hw_context(d, xclbin.get_uuid())
kern = pyxrt.kernel(hw, kname)
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
bo_qpv = pyxrt.bo(d, qpv_b.nbytes, pyxrt.bo.host_only, kern.group_id(3))
bo_p = pyxrt.bo(d, p_b.nbytes, pyxrt.bo.host_only, kern.group_id(4))
bo_k = pyxrt.bo(d, k_b.nbytes, pyxrt.bo.host_only, kern.group_id(5))
bo_v = pyxrt.bo(d, v_b.nbytes, pyxrt.bo.host_only, kern.group_id(6))
bo_c = pyxrt.bo(d, NQ * DK * 2, pyxrt.bo.host_only, kern.group_id(7))
for bo, arr in ((bo_instr, instr), (bo_qpv, qpv_b), (bo_p, p_b), (bo_k, k_b), (bo_v, v_b)):
    bo.write(arr.tobytes(), 0); bo.sync(TO)

r = kern(3, bo_instr, instr.size, bo_qpv, bo_p, bo_k, bo_v, bo_c); r.wait()
bo_c.sync(FROM)
ctx_dev = np.frombuffer(bo_c.read(NQ * DK * 2, 0), dtype=np.uint16).view(bfloat16).astype(np.float32).reshape(NQ, DK)

rel = np.linalg.norm(ctx_dev - ctx_ref) / (np.linalg.norm(ctx_ref) + 1e-12)
print(f"[bd_onchip] T={T} TQ={TQ} DK={DK} N_QT={N_QT} P={P} BD_SPLIT={BD_SPLIT} kernel='{kname}'")
print(f"[bd_onchip] ctx_dev[0,:3]={ctx_dev[0,:3]}  ctx_ref[0,:3]={ctx_ref[0,:3]}")
print(f"[bd_onchip] rel-L2={rel:.5e}  gate<=5e-3  {'PASS' if rel <= 5e-3 else 'FAIL'}")
