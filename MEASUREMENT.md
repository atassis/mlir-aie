# Measuring aiex.buffer_clear

## Device test: what it proves

`test/npu-xrt/buffer_clear_op/` is a correct-with / drift-without pair, the same shape as the merged
`dma_channel_reset` device test.

- A core on tile (0,2) runs once and fills an 8-word buffer `acc` with `[1..8]` (distinct nonzero values per
  word, so an all-zero readback cannot be a coincidence).
- Batch 1 DMA's `acc` to the host: must read back `[1..8]`, confirming the accumulator really holds
  nonzero data before the op runs.
- `aiex.buffer_clear(%tile_0_2, 0x400, 8)` runs directly from the runtime sequence. No core re-run: the
  sequence re-triggers the readback DMA by setting the consumer lock itself, so the compute core is out of
  the picture between batch 1 and batch 2. That isolates the op as the only thing that can have changed
  `acc`'s contents.
- Batch 2 DMA's `acc` again: must read back all-zero.
- **Drift case:** delete the `aiex.buffer_clear` line and rerun. Batch 2 now equals batch 1 (`[1..8]`),
  since nothing else touches `acc` after the core's one run. That is the falsifiable signal: correct-with
  passes only because the op did the clear, not because of some other effect in the design.

This is a functional/correctness test, pass or fail, run via `run.lit` (`REQUIRES: ryzen_ai`,
`--alloc-scheme=basic-sequential`). It proves the op writes zeros to exactly the region it claims to, and
nothing else changes the outcome. It does not, and is not meant to, produce a performance number.

**Status:** the buffer address bug in this remediation pass (see commit) is fixed by inspection against
`AIEAssignBuffers.cpp`'s stack-overlap check, not by an actual build -- I do not have a built toolchain
instance in this environment. Before this is called validated, someone needs to run
`%aiecc --alloc-scheme=basic-sequential ... --get-xclbin --get-npu-insts` on real hardware and confirm both
the compile succeeds and the on-device readback matches.

## Perf test: there is no throughput story, and I am not inventing one

`aiex.buffer_clear` is a correctness/safety primitive (bounds-checked memset), not a data-movement
optimization. It lowers to exactly the `npu.blockwrite` a hand-rolled caller would already emit; using the
op instead of the hand-rolled composition changes zero bytes in the control stream and zero cycles on
device. The only thing it changes is that a length/address mistake is now a compile-time verifier error
instead of a silent wrong write. So there is nothing to gate on a device timed run for the op in isolation,
and I am not going to manufacture a speedup number for a memset that was already a bare blockwrite.

What *would* need a numeric gate, and when:

1. **Structural parity (available now, no device needed).** FileCheck the lowering: for a given
   `(address, length)`, `aiex.buffer_clear` must emit the identical `npu.blockwrite` (same address, same
   word count, same zero payload) a hand-written composition would. This is exact-match, not statistical --
   already covered by `test/Passes/lower-buffer-clear/buffer_clear.mlir`.

2. **Correctness gate for a real caller, once one exists.** The op only has an in-context perf question once
   it is wired into an actual resident block -- for example, clearing a norm-state or accumulator buffer
   between decode steps in a resident per-token loop. At that point the gate is **rel-L2 of the resident
   block's output tensor against a reference that reloads/reinitializes the same state instead of clearing
   it in place** (or token-parity across N decode steps if the consumer is a full decode path), not the
   17-clip WER: WER on that clip set is chaotic at the ~1e-5 level (`seventeen-clip-wer-gate-is-chaotic`) and
   cannot distinguish a numerically-equivalent change from noise. Use the same bar this repo already applies
   to bf16-accumulation-sensitive changes: aggregate rel-L2 or mean-abs-diff, not elementwise atol, threshold
   set from the existing bf16-norm-numerics precedent (~1e-3 rel-L2), not derived fresh for this op.
   Concretely: run the resident block N times with `aiex.buffer_clear` performing the reset, compare its
   output tensor to the same block run with a full state reload at the same points; rel-L2 must clear the
   same bar already used for other bf16 accumulation-sensitive changes in this codebase.

3. **Control-stream cost, only if `length` grows.** If a real caller ever wants a large clear (multi-KB,
   not a small accumulator), the honest thing to measure is added runtime-sequence instruction count /
   bytes for that `length`, compared against the DMA-of-zeros composition the op's own description points
   to for that case. That comparison is a byte count from the compiled artifact, not a timed run -- no
   device time needed, and no device time would tell you more than the byte count already does.

## What to quiesce before any timed run

The NPU is single-tenant and shared. Before any run that reads a timer: announce the run, `fuser` the device
node to confirm nothing else holds it, stop `npu-asr` (and any other resident device consumer), and do not
run concurrently with another device session. None of the above actually needs a timed run today -- this
section only applies once a real caller exists and item 3 becomes relevant.

## Honest summary

`aiex.buffer_clear` can be expected to move: nothing, on its own. It is not a speedup; it is a bounds-check
that turns a silent out-of-range memset into a compile-time error. The only artifact worth a device run is
the correctness pair above, and that artifact is what the "device test" column in the op's tracking exists
to prove, not a benchmark number.
