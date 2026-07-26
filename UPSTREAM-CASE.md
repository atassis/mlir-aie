# aiex.bd_length: scoped BD-length rewrite for the runtime sequence

## Problem

The AIEX runtime-sequence dialect already promotes several aie-rt driver calls into first-class ops
(`aiex.dma_channel_reset`, `aiex.core_reset`), but `XAie_DmaUpdateBdLen` never got the same treatment.
Today the only way to change a programmed BD's transfer length between dispatches is to re-emit the
whole BD (a fresh `dma_memcpy_nd`/`dma_configure_task`, or a hand-rolled `npu.maskwrite32`/`npu.write32`
at the call site), even when address, stride, and every other field are unchanged.

## Hazard this removes

Hand-rolling that length-only write means re-deriving mask/shift/offset per tile type at the call site,
three times over (mem tile, core/memory-module tile, shim/NOC) -- and aie-rt's own C driver gets one of
the three bound checks wrong. `XAie_DmaUpdateBdLen`'s check is `BdNum > DmaMod->NumBds`, not `>=`
(`driver/src/dma/xaie_dma.c`). For a mem tile, `NumBds = 48` (valid indices 0..47), so `BdNum == 48`
passes aie-rt's own check and the caller goes on to compute a `RegAddr` one BD slot past the last real
BD -- a silent out-of-range register write, not a diagnostic. A dedicated op with its own verifier is one
place to fix this instead of leaving every future maskwrite32 call site free to reproduce it.

The same shape of mistake already happened once in this dialect: `dma_channel_reset`'s first draft used
a plain `write32` for the reset pulse and clobbered co-packed control fields; the merged version scoped
it to a `maskwrite32` on the reset bit only. A length-only field write is the same bug class waiting to
recur at the next call site -- worth closing in the op rather than trusting every caller to remember it.

## Design

Thin, follows the shape of `dma_channel_reset`/`core_reset`:

- `aiex.bd_length(%tile, bd_id, length)`, `HasParent<AIE::RuntimeSequenceOp>`.
- Verifier: `bd_id < NumBds` for the tile's DMA module -- strict less-than, a deliberate divergence from
  aie-rt's own `BdNum > NumBds` check, not an oversight. A reviewer who "corrects" it back to match
  aie-rt's C driver would reintroduce the off-by-one; the op's doc comment should say so explicitly.
- Lowering: one `npu.maskwrite32` per tile type, mask/shift taken from the tile's BD-length field
  (mem: `0x0001FFFF`, core/memory-module: `0x00003FFF`, shim/NOC: `0xFFFFFFFF`, all at word offset 0 in
  the BD's register block on aie2p). The shim mask is the full word, so a maskwrite32 there is
  bit-for-bit equivalent to a plain write32 -- no need to fork a second write-kind to mirror aie-rt's C
  driver, which uses `XAie_Write32` for shim only because that particular register happens to carry no
  other packed field.

## Status

Not implemented yet. It sits behind `aiex-write-field-abstraction`, a helper that derives the mask/shift
above from the aie-rt `RegFldAttr` tables instead of hand-coding it, currently in flight in a sibling
worktree. Landing that first turns this op into the same few-line pattern as the two ops it follows,
instead of a third hand-rolled copy of masks the abstraction is about to make redundant. See
MEASUREMENT.md for how I'd validate it on device once it lands.
