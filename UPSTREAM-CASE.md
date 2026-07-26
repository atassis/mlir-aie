<!---//===- UPSTREAM-CASE.md ---------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# The case for `aiex.cascade_reconfigure` -- and why I am not filing it

This is not a PR proposal. It is the write-up I would need before filing one, and it stops short of a
"file it" conclusion. Read it as: here is the correctness case, and here is why it currently has no
caller.

## The hazard, if this gets built

`ACCUMULATOR_CONTROL` (aie2p core-module offset `0x36060`, `driver/src/core/xaie_core.c:973`) packs two
logical fields into one 32-bit register: input direction (bit 0, North/West) and output direction (bit 1,
South/East). Everything above bit 1 is reserved. aie-rt's own write is a plain `XAie_Write32`, safe only
because `XAie_CoreConfigAccumulatorControl` always sets both fields together and nothing else in that word
is defined today. A runtime-sequence op must not inherit that assumption: the same family already burned
this once. `dma_channel_reset` and `core_reset` both had to switch to `maskwrite32` mid-review because
their target registers share bits with fields the op has no business touching, and a later hardware
generation redefining reserved bits in a shared register is exactly the failure mode a plain `write32`
walks into silently. Any future `cascade_reconfigure` inherits that constraint: `maskwrite32`, mask
`0x3`, non-negotiable, verified by construction the same way as its two predecessors.

That is a real correctness requirement, not a hypothetical one -- it is what I would insist on in review
if someone else proposed this op. It is not, on its own, a reason to build the op.

## Why I am not proposing it

`ConfigureCascadeOp` (`AIEOps.td:1663`, `HasParent<"DeviceOp">`) is the only thing in the AIE/AIEX
dialects that reaches `XAie_CoreConfigAccumulatorControl`, and it lowers exactly twice, both inside
`AIERTControl::addInitConfig` (`AIERT.cpp:789`, called from `AIETargetCDODirect.cpp:96` and `:125`) --
the one-time device-image CDO/xclbin build, never from `aie.runtime_sequence`. `grep -rniI cascade
include/aie/Dialect/AIEX lib/Dialect/AIEX` returns nothing: the whole `aiex.*` runtime-op family has no
cascade awareness. That is a real gap in what the runtime sequence can express.

But every user of cascade I can find, in this codebase or upstream, treats cascade wiring as a placement
property, not a per-dispatch parameter. `aie.cascade_flow(src, dst)` fixes which physical neighbor a
core's accumulator talks to at compile time; nothing recompiles that relationship for a second dispatch
against the same resident device image. A maintainer reading a `cascade_reconfigure` PR would ask, fairly,
"who calls this, and why can't they get a fresh CDO load instead" -- and I do not have an answer, because
I looked for one and did not find a caller. Landing a runtime-seq op nobody calls adds API surface mlir-aie
has to keep supporting (verifier, lowering pass, docs, test matrix) for a capability the ecosystem has not
asked for. That is a cost with no offsetting benefit today.

## What would change my mind

A dataflow that rewires the same physical cascade-connected tiles into two different topologies within
one resident device-image load, without an intervening CDO/xclbin reload. I don't have one on my roadmap
and haven't seen one upstream. If that shows up, the op sketch is already written (see
`AIEX-CASCADE-RECONFIG-VERDICT.md`, section "Op sketch") and follows `core_reset`/`dma_channel_reset`'s
shape exactly -- the only work left at that point is the lowering pass and the two test pairs.
