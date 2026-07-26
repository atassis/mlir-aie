# Measuring the bulk-reset pattern

This commit adds no lowering path and no new op. `aiex.dma_channel_reset` is
already merged, already lowers through `AIELowerDmaChannelReset`, and already has
an on-board correct-with/drift-without pair at
`test/npu-xrt/local_reset/dma_channel_reset_op/`. What this commit adds is a
documented *loop* over that existing op, plus a cross-reference to the re-arm
hazard. What needs measuring is therefore narrower than a new-op validation: does
the documented loop actually reset every channel in the stated order with no
missed or duplicated channel, and does looping it change anything about cost.

## Device test: prove the loop, not just the single call

The merged `dma_channel_reset_op` test proves one channel. It does not prove the
paragraph's claim that MM2S 0..N-1 then S2MM 0..N-1 covers every channel on the
tile. Propose extending it with a `dma_channel_reset_op_bulk` family, same shared
design as the rest of `local_reset` (resident workload, reset while settled,
driven from the runtime sequence), on a mem tile so N > 1 in each direction
(6 MM2S + 6 S2MM on aie2p, vs. 2 + 2 on a core tile, so a missed or misordered
channel in the loop is distinguishable from a coincidence):

- Stage a "bad" BD queued ahead of a "good" BD, gated on a `cons` lock exactly
  like the existing `dma` / `dma_channel_reset_op` tests, independently on every
  channel in both directions.
- **Correct-with:** the runtime sequence emits the full documented loop (MM2S
  0..N-1, then S2MM 0..N-1, N from `getNumSourceSwitchboxConnections` /
  `getNumDestSwitchboxConnections`) before re-pushing the good BD and re-arming
  every lock. Every channel's collect returns the good pattern.
- **Drift-without (channel coverage):** drop one channel from the loop, e.g. the
  last S2MM channel. Only that channel's collect returns the bad pattern; this is
  the wrong-data failure the existing single-channel test already establishes,
  now showing a loop bug (off-by-one in N, wrong direction order) is
  distinguishable from a correct loop instead of passing by accident on a
  1-channel tile.
- **Drift-without (re-arm hazard, not part of the automated `run.lit` — hangs by
  design, same convention `local_reset/README.md` already uses for the raw
  `dma`/`core` negatives):** run the documented loop against a tile with a live
  objectFIFO-bound channel and omit the `aiex.set_lock` re-arm this commit's new
  sentence calls out. The channel's peer blocks on acquire and the collect never
  completes. This is the direct on-board demonstration of the finding fixed in
  this commit: it shows the hazard the added sentence warns about is real, not
  hypothetical, and that `AIEVerifyRuntimeRearmPass` catching it at compile time
  (when opted in) is load-bearing, not decorative.

This is proposed, not implemented in this commit. It is worth adding as a
follow-up because it is cheap (same harness, same host oracle pattern, one more
`aie.mlir`) and it guards the one thing this documentation asserts that the
existing single-channel test does not: loop coverage. It is not required to land
the documentation itself, since the documentation makes no claim the merged
`dma_channel_reset` device test does not already cover per channel.

## Performance test: honest scope

This is a documentation and ergonomics change, not a new lowering, so there is no
new performance surface to gate. The bulk-reset loop is N calls to a primitive
that is already the shipped, already-benchmarked `DmaChannelResetToMaskWrite32`
pattern (two `npu.maskwrite32` per channel, mask-preserving pulse, no other
register touched). There is no story where looping it is faster or slower per
call than calling it once; the only thing worth checking numerically is that the
loop is linear in channel count with no hidden per-iteration overhead (queue
serialization, scheduler stall) that would not show up testing one channel.

- **Metric:** wall-clock span from the first reset `maskwrite32` to the last
  channel's start-queue re-push, captured via NPU device trace (packet
  timestamps, same trace infra already used for dispatch profiling), for the
  full-tile loop (12 writes on a mem tile: 6 MM2S + 6 S2MM channels, 2 writes
  each) versus the single-channel case scaled by count.
- **Pass threshold:** measured full-loop span within 10% of
  (single-channel span x channel count). Above that, something in the loop is not
  linear (e.g. an implicit sync between channels) and is worth a separate
  investigation; this is a regression gate, not a target to optimize toward.
- **Correctness gate for any decode-loop integration:** token-parity, not the
  17-clip WER (chaotic at ~1e-5, cannot validate a numerically-equivalent
  change). Run the resident-decode dispatch loop for N dispatches with the
  bulk-reset pattern applied at each dispatch boundary versus a cold-reload
  baseline for the same input; require identical top-1 token sequence across all
  N dispatches. This is a correctness check on the re-arm discipline (every lock
  actually comes back armed), not a speed claim.
- **What this cannot be expected to move:** end-to-end decode tok/s. The reset
  writes are a fixed, small, already-amortized cost inside a dispatch boundary
  that is dominated by LPDDR movement, not by control-register writes; there is
  no speedup story here and I am not claiming one.

## Device discipline for any timed run

The NPU is single-tenant and shared. Before any timed run: announce on the shared
channel, `fuser` the device node to confirm nothing else holds it, and stop
`npu-asr` (and any other resident service) rather than assuming a nothing-else-
running state. Do not run trace-timed passes concurrently with another
session's device work.
