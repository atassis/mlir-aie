# Measuring aiex.bd_length

Not implemented yet (see UPSTREAM-CASE.md / OUR-JUSTIFICATION.md for why). This is the test design to
build against once `aiex-write-field-abstraction` lands and a real need shows up -- written now so the
next session doesn't have to redesign it.

## Device test (npu-xrt): correct-with / drift-without pair

Same shape as the merged `test/npu-xrt/local_reset/dma_channel_reset_op` test, which proves
`dma_channel_reset` by making its absence produce wrong data, not a hang.

Fixture: a source buffer with two distinguishable regions -- a "prefix" of N1 words (value = 100+i) and
an "extension" tail of N2-N1 words (value = 200+i), source length N2 total. One programmed S2MM BD,
initial length N1. Output buffer sentinel-filled with `0xdeadbeef` before every dispatch (same discipline
as the existing test -- a hung or partial transfer must read as garbage, not as stale-but-plausible data).

Runtime sequence, per dispatch:
- Dispatch 0: BD length is still N1 (unchanged). Collect must equal exactly the N1-word prefix; words
  N1..N2 in the output stay at the sentinel, because the BD never asked for them.
- Between dispatch 0 and 1: `aiex.bd_length(%bd, bd_id, N2)` widens the same BD in place -- the op under
  test, no other field of the BD touched.
- Dispatch 1: collect must equal the full N2 words, prefix and extension both matching source.

**Correct protocol (this test):** dispatch 0 returns the N1-word prefix; dispatch 1, after the resize,
returns all N2 words matching source, extension included.

**No resize (falsifies the test):** drop the `aiex.bd_length` between dispatches -- dispatch 1 still only
transfers N1 words, so its tail (N1..N2) stays at `0xdeadbeef` instead of the extension pattern. Wrong
data, not a hang -- the same falsification signature `dma_channel_reset_op` uses, and deliberately so:
a hang-based test can't distinguish "op did nothing" from "op did something else wrong."

**Resize without a follow-up dispatch:** not a meaningful third case here (unlike the reset test's
"reset without re-push" case) -- there's no queue-arm step this op could skip; it only rewrites one
field of an already-armed BD.

Location: `test/npu-xrt/local_reset/bd_length_op/`, host oracle adapted from
`test/npu-xrt/local_reset/dma_channel_reset_op`'s `test.cpp` (same dispatch-loop-and-compare structure,
two source regions instead of two source BDs).

## Perf test: numeric gate

**What the op can move:** control-program instruction count for a length-only BD update. A full BD
re-issue (`dma_configure_task`/`dma_memcpy_nd`) emits one `npu.maskwrite32`/`npu.write32` per programmed
field (address hi/lo, length, wrap/use-next, etc. -- 4-6 instructions depending on tile type); the scoped
op emits exactly one.

**What it cannot move:** LPDDR bytes moved. The DMA still transfers exactly N2-N1 additional bytes when
the length grows, physics-bound like any other transfer of that size -- this op changes how the BD gets
reprogrammed, not the cost of the data it then moves.

**Gate (static, deterministic -- not a device timing number):**
- Compile the same length-only change through both lowering paths (full re-issue vs scoped
  `aiex.bd_length`) for mem/core/shim tile fixtures.
- Count `npu.maskwrite32`/`npu.write32` instructions emitted at the length-change site.
- Pass: scoped path emits exactly 1 instruction per tile type vs the full-rewrite path's measured count
  (>= 3). This is a compile-time instruction count, immune to device noise -- the right kind of gate here
  because a device-timing measurement of a few-instruction difference inside a DMA-bound dispatch will
  not resolve above noise (see below).

**Correctness gate, if/when this is validated against a real model:** rel-L2 or token-parity between the
scoped-op path and the full-rewrite path on identical inputs. **Never the 17-clip WER** -- it's chaotic
at the ~1e-5 level and cannot validate a numerically-equivalent change (this op, if correct, is bit-exact
equivalent to the full rewrite; the WER gate would just add noise, not signal).

**Optional device-timing companion, only worth running once a real workload exercises this op:** median
wall-clock over >=30 repeated resize+dispatch cycles, scoped-op vs full-rewrite, same length change.
Gate on a bootstrapped CI over the relative median delta, not a point estimate. Go in expecting this to
be noise-dominated at today's BD counts -- removing 3-5 maskwrite32s (tens of ns in the instruction
stream) is unlikely to clear the noise floor of a DMA-bound dispatch running hundreds of microseconds to
milliseconds. Don't run this companion at all unless the target scenario does enough resizes per dispatch
(deep BD chains, per-token resize in a tight decode loop) that the instruction-count reduction could
plausibly matter -- decide that before running it, not after.

**NPU quiescing:** single-tenant, shared box. Before any timed run: announce on the shared channel,
`fuser` the device node, confirm no other `xrt::device` holder, stop `npu-asr`. Serialize -- don't
interleave with another device test. `bd_length` itself is not an energy-relevant op, so no RAPL
idle-subtract is needed unless a power number ever gets attached to the timing companion above.

## Bottom line

This is a correctness/ergonomics/control-program-size op, not a throughput op. Its performance case is
entirely prospective -- there's no shipped workload today that resizes a BD between dispatches, so there's
nothing for it to speed up yet. The instruction-count gate above is real and worth having on file, but
don't report it as a decode tok/s win until a target model actually exercises variable-length dispatch.
