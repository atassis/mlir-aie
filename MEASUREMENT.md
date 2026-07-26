<!---//===- MEASUREMENT.md ----------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Measurement plan for `aiex.cascade_reconfigure` (design only -- op is not built)

Nothing here has been run. This is the test I would build alongside the op if `UPSTREAM-CASE.md`'s
"what would change my mind" condition is ever met. Recorded now so the design work isn't lost.

## Device test: correct-with / drift-without pair

Model: `test/npu-xrt/local_reset/dma_channel_reset_op/` (bad-BD/good-BD falsifiable pair). The equivalent
here needs a real routing choice to falsify, which is where this design is honest about an open
precondition:

**Open precondition, check before building the test:** `ConfigureCascadeOp`'s verifier and aie-rt both
constrain input to {North, West} and output to {South, East} -- a *choice* of neighbor, not a single fixed
wire. Whether a given aie2p tile can have both candidate physical links (e.g. cascade-in from North AND
from West) simultaneously present in the stream-switch configuration, with `ACCUMULATOR_CONTROL` alone
selecting which one is live, is a stream-switch routing question I have not verified against silicon or
against `mlir-air`'s cascade-flow legality checks. If only one candidate link can be routed to a tile's
cascade-in port at CDO-build time, the test needs two device-image variants (reconfigure) as its "before"
state and this reduces to testing a `write32` executes at all, not a real retarget -- a much weaker test.
Confirm the two-link case is representable before writing `aie.mlir`.

Assuming it is representable, the pair:

- **Topology:** 3 AIE tiles, `tN` (north neighbor), `tW` (west neighbor), `core` (device under test).
  `tN` and `tW` each hold a distinct known buffer value fed into the cascade fabric
  (`aie.cascade_flow(tN, core)` and `aie.cascade_flow(tW, core)` both declared at compile time).
  `core`'s kernel does nothing but forward its cascade-in to its cascade-out to a shim DMA collect buffer,
  so the collected value tells you unambiguously which neighbor was live.
- **Correct protocol (test):** dispatch 1 sets `InDir=WEST` via `aiex.cascade_reconfigure`, runs, collect
  matches `tW`'s value. `aiex.cascade_reconfigure(InDir=NORTH)` between dispatches, no CDO reload.
  Dispatch 2 runs, collect matches `tN`'s value.
- **No reconfigure (falsifies):** skip the second `cascade_reconfigure` call -- dispatch 2's collect still
  matches `tW`'s value (stale routing), a silent wrong-data failure, not a hang. This is the load-bearing
  negative case: it proves the register write is what moved the data, not something else in the sequence.
  Same failure shape `dma_channel_reset_op` uses (wrong array, not a timeout).
- **Reconfigure without the mask (regression guard):** swap `maskwrite32` for a plain `write32` in the
  lowering and confirm CI catches it if a future aie2p reginit table ever defines bits 2-31 in that word
  -- won't fire on current silicon (nothing else is defined there today), so this is a lowering-review
  guard, not a device assertion; keep it as a lit-level check on the emitted `npu.maskwrite32` mask
  operand, not a device test.

Both dispatches share one resident load (`aie.runtime_sequence`, no intervening CDO/xclbin reload) --
that is the entire point of the op, so the test must assert it stays that way (single `xclbin`, single
`hw_context`, two `run()` calls).

## Performance test: numeric gate

**Correctness gate:** exact match on the collected `i32` array, same as `core_reset`/`dma_channel_reset`'s
own tests. This is a deterministic integer control-plane routing check, not a numeric kernel -- rel-L2 and
token-parity are the right gates for bf16/int8 kernel changes elsewhere in this codebase, not for this op;
using a statistical gate here would be gating precision the op doesn't have and hiding an exact-match
regression behind a tolerance. (17-clip WER does not apply either, for the same reason it never applies
to a numerically-equivalent control-plane change: chaotic at the ~1e-5 level, can't distinguish "routed
correctly" from noise.)

**What this op can plausibly move, and what it can't:** `cascade_reconfigure` is a single `maskwrite32` --
its own execution cost is a few cycles, unmeasurable against anything else in a dispatch. It has **no**
performance story on its own. The only performance claim that could ever attach to it is comparative: does
using it to retarget cascade routing mid-residency cost less than the alternative (a full CDO/xclbin
reload) for a workload that needs to switch cascade topology between dispatches. That comparison needs a
real caller to be meaningful, and there isn't one -- so there is no perf number to gate on today. If a
caller shows up, the metric would be:

- **Metric:** wall-clock time from "last op of dispatch N" to "first op of dispatch N+1 begins executing
  on core," compared between (a) `cascade_reconfigure` + no reload and (b) a full CDO/xclbin reload
  carrying the same topology change.
- **How measured:** XRT event timestamps around the `hw_context`/`run()` boundary, on-device, not host
  wall-clock (host-side timing over-counts scheduling jitter). Cross-check against the known
  ~2.67 ms/hw-context-switch shape-reload cost already measured for full reconfigures
  (`docs/log/` shape-reload microbench) -- `cascade_reconfigure` should land near-zero against that
  baseline if it's doing its job.
- **Pass threshold:** (a) at least 10x cheaper than (b), reproduced over >=20 timed retarget events after
  quiesce. A weaker win wouldn't justify the added op surface over just eating the reload.
  This threshold is illustrative, not derived from data -- there is no measured (a) or (b) for this
  specific op yet, only the general shape-reload number the comparison would be checked against.

**Quiesce discipline before any timed run** (device-measurement-discipline): the NPU is single-tenant --
announce the run, `fuser` the device node, stop `npu-asr`/any other resident service, don't let anything
auto-restart mid-measurement, serialize against other device work. RAPL/energy claims are whole-SoC, so if
energy ever gets measured alongside this, idle-subtract after quiescing, not before.

## Honest bottom line

This op is control-plane routing, not compute or bandwidth. Even if built, it would not move tokens/sec,
watts, or any resident-decode throughput number by itself -- its only possible win is avoiding a full
reconfigure for a workload that doesn't exist in this codebase yet. Do not attach a decode-speedup story
to this op; it doesn't have one.
