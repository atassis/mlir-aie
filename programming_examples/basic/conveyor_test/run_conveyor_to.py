#!/usr/bin/env python3
"""Timeout-guarded conveyor runner. XCLBIN/INSTS from env; r.wait() bounded so a
hung core (disabled) reports TIMEOUT instead of blocking forever."""
import os, sys, numpy as np, pyxrt

EX = os.path.join(os.path.dirname(__file__), "build")
XB = os.environ.get("XB", "final.xclbin")
INS = os.environ.get("INS", "insts.bin")
N = 256
instr = np.fromfile(f"{EX}/{INS}", dtype=np.uint32)
xclbin = pyxrt.xclbin(f"{EX}/{XB}")
kname = xclbin.get_kernels()[0].get_name()
d = pyxrt.device(0); d.register_xclbin(xclbin)
ctx = pyxrt.hw_context(d, xclbin.get_uuid())
k = pyxrt.kernel(ctx, kname)
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

inp = np.arange(N, dtype=np.int32)
bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
bo_in = pyxrt.bo(d, inp.nbytes, pyxrt.bo.host_only, k.group_id(3))
bo_out = pyxrt.bo(d, N * 4, pyxrt.bo.host_only, k.group_id(4))
bo_instr.write(instr.tobytes(), 0); bo_instr.sync(TO)
bo_in.write(inp.tobytes(), 0); bo_in.sync(TO)

r = k(3, bo_instr, instr.size, bo_in, bo_out)
st = r.wait(8000)  # 8s timeout
if str(st) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
    print(f"TIMEOUT/INCOMPLETE state={st}  (core likely hung)")
    sys.exit(2)
bo_out.sync(FROM)
out = np.frombuffer(bo_out.read(N * 4, 0), dtype=np.int32)
expect = (inp + 1) * 2
ok = np.array_equal(out, expect)
print(f"XB={XB} INS={INS}  out[:6]={out[:6]}  expect[:6]={expect[:6]}")
print("PASS (out == (in+1)*2)" if ok else f"FAIL: {(out != expect).sum()} mismatches, out[:6]={out[:6]}")
sys.exit(0 if ok else 1)
