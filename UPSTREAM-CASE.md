Title: [AIEX] Add aiex.buffer_clear to zero tile-local data memory from a runtime sequence

## What this adds

A runtime-sequence op that zeros `length` 32-bit words of a core or mem tile's local data memory starting
at a word-aligned byte offset, lowering to a single `npu.blockwrite` of a zero-filled buffer. It follows the
same pattern as `aiex.core_reset` / `aiex.dma_channel_reset`: promote a composition that is already
possible by hand into a typed, verified op.

## The hazard this removes

Nothing today stops a hand-rolled zero-fill blockwrite from writing outside a tile's actual local data
memory. `npu.blockwrite` takes an arbitrary address and has no verifier at all -- by design, since it is
the shared low-level primitive behind BD writes, RTP writes, and raw register writes, each with a different
address space and no single meaningful bound. A caller who builds a zero-filled `memref.global` +
`memref.get_global` + `npu.blockwrite` by hand to clear, say, a resident accumulator has to get the byte
range right against the tile's real local memory size (0x10000 on a core tile, 0x80000 on a mem tile, both
aie2p) with nothing checking the arithmetic. Get `address + length*4` wrong and the write does not fail, it
silently lands in whatever the array's address map puts next to that tile's data memory window, and the
first symptom is wrong results somewhere else in the design. This is the same class of bug the project just
closed from the other direction in #1097/#3398 (validating DMA BD `buffer_length` against the tile-type
field width): a hardware-adjacent numeric field with no compiler-side bound, silently wrong instead of
loudly rejected.

`aiex.buffer_clear`'s verifier closes exactly this gap for the memset case: it rejects zero length,
non-word-aligned addresses, shim tiles (no local data memory to clear -- matching aie-rt's
`XAie_DataMemBlockWrite`, which only accepts `AIETILE`/`MEMTILE`), and any `[address, address + length*4)`
range that exceeds `AIETargetModel::getLocalMemorySize()` / `getMemTileSize()` for the tile it targets.

I looked at folding this bound into `npu.blockwrite`'s own verifier instead of adding a new op, since that
would benefit every blockwrite call site, not just zero-fills. It does not work: blockwrite's `address` is
only interpretable against the tile's local data memory size when the caller's *intent* is "write into this
tile's data memory." For a BD write, an RTP write, or an absolute-address register write, "does this exceed
the tile's local memory size" is not the right check, or not a check at all. A typed op that means "this is
a tile-local-data-memory clear" is what makes the bound well-defined and safe to enforce unconditionally;
`npu.blockwrite` cannot know that without being told, which is what the new op tells it.

## What it deliberately does not do

No DMA-of-zeros lowering. AIE2P's BD constant-pad field looked like a free destination fill at first, but it
augments the MM2S (read) side of a transfer for halo-style stream padding, is mem-tile-only, and is capped
at a handful of words per dimension -- it cannot fill an S2MM destination. A real zero-source DMA clear
needs a persistent zero buffer plus a routed flow between two channels, both compile-time resources a
runtime-sequence-scoped lowering cannot synthesize the way it synthesizes a fixed register pulse. So unlike
`core_reset`/`dma_channel_reset`, this op's cost in the runtime sequence scales with `length`: it is meant
for the small, hot-path case (an accumulator or norm state), not a whole KV-cache reset. That composition
is already available today directly via `aiex.dma_configure_task`/`aie.dma_bd`/`aie.flow` for the large,
infrequent case, and this op does not attempt to replace it.

There is also no dedicated memset at the driver layer to skip: aie-rt's `XAie_DataMemBlockWrite` builds the
payload in a caller buffer and issues a block write, the same mechanism this op lowers to. This op does not
route around any unwritten lower layer, it promotes a pattern that is already the state of the art one level
down.

## Notes for review

1. The verifier rejects AIE1 (the runtime sequence has no meaning there, matching `set_lock`/`core_reset`),
   shim tiles, zero length, unaligned addresses, and out-of-bounds ranges.
2. The lowering (`AIELowerBufferClearPass`) shares zero-data globals across call sites of the same length,
   the same dedup mechanism `AIEDmaToNpuPass::WriteBdToBlockWritePattern` already uses for BD data.
3. FileCheck coverage: a lowering test checking the emitted `npu.blockwrite` (address, data, column/row) and
   an invalid-input test covering every verifier rejection.
4. npu-xrt device test: a core tile fills an 8-word accumulator with distinct nonzero values, a batch read
   confirms it, `aiex.buffer_clear` zeros it directly from the runtime sequence (no core re-run), and a
   second batch read confirms all-zero. Dropping the op from the design is the drift case: without it,
   batch 2 equals batch 1, since nothing else touches the buffer after the core's single run.
5. I have not yet run this on real hardware. I do not have a built toolchain instance in the environment I
   used to write this, so what I have verified is: the register/memory-size constants against aie-rt's
   source (`xaie2pgbl_reginit.c`), the SSA-tile-operand shape against `core_reset`/`dma_channel_reset`'s
   precedent, and the lit tests by inspection. The device test needs a real run on Ryzen AI hardware before
   I would call this validated to the bar the merged reset ops were held to.
