# Case for the dma_channel_reset bulk-reset paragraph (and its re-arm cross-reference)

I looked at adding a dedicated `aiex` op mirroring aie-rt's `XAie_DmaChannelResetAll`
before writing this. It is a bare loop over `XAie_DmaChannelReset`, MM2S then S2MM,
for every channel on a tile; no tile-wide reset-all register bit exists under it,
and no register knowledge in it that the merged `aiex.dma_channel_reset` does not
already expose per channel (`AIETargetModel::getNumSourceSwitchboxConnections` /
`getNumDestSwitchboxConnections` with `WireBundle::DMA` gives the same channel
count the op's own verifier uses to bound `channel`). A dedicated op would add
dialect surface, a verifier, and a lowering pass that reproduce
`DmaChannelResetToMaskWrite32` in a loop, for nothing aie-rt does that the existing
op does not. So I did not build it; I documented the loop instead, on
`dma_channel_reset`'s own description, so nobody reinvents it as a new op.

That paragraph had a gap. It sits two paragraphs below the op's stated primary use
case, re-arming a resident objectFIFO across dispatches, and its only caveat is
that resetting an unconfigured channel is safe. It says nothing about a channel
that *was* configured. I merged `AIEVerifyRuntimeRearmPass` two commits earlier for
exactly that case: a reset drains the DMA task queue and freezes the objectFIFO
lock counters, and a lock that is never re-armed with `aiex.set_lock` leaves the
channel's peer blocked on an acquire forever, no compiler diagnostic, just a
host-side `qds_device::wait()` hang. The pass is opt-in, not in the default
pipeline, so it does not catch this at compile time by default. A reader who takes
the bulk-reset paragraph's loop recipe at face value and applies it to a tile
carrying live objectFIFO channels reproduces that deadlock and gets no signal
pointing back at the reset.

The fix is one sentence where the loop recipe is given: resetting a channel bound
to live objectFIFO locks requires re-arming each one afterward or its peer
deadlocks, and the pass exists to catch a missing re-arm in this pattern the same
way it catches a single one. No code change, no new dialect surface; it closes the
gap between where the hazard is documented (a pass docstring nobody reads until
they already suspect a problem) and where a caller is most likely to hit it (the
paragraph that tells them how to write the loop and stops one sentence short of
telling them what it costs).
