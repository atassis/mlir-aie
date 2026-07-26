# Why the resident decode regime would want aiex.bd_length -- and why it doesn't yet

## The need, honestly stated

The current resident decode path dispatches fixed-shape BDs: every token's transfer length is the same
because the shape was pinned at compile time (static-shape batch regime -- see
`docs/kb/npu-optimization-map.md`). Nothing shipped today resizes a BD between dispatches, so nothing
shipped today needs `aiex.bd_length`.

The op's own task file names the scenario where that would change: variable-length prefill chunks or
ragged per-token KV tiles, where the transfer length legitimately differs dispatch to dispatch while the
buffer address and everything else about the BD stays fixed. That is a real shape a resident engine could
hit -- prefill chunking and ragged batching are standard techniques elsewhere -- but nothing on the
current model roster (Parakeet encoder, the Gemma decode work) has actually needed it yet. I looked for a
demonstrated case in `decode-north-star` and `runtime-op-completeness` and found none.

## YAGNI verdict: speculative

Not demonstrated. This is a plausible future need inferred from the general shape of ragged/variable-length
dispatch, not a recurring need any current task has actually hit. The op's own task file already gates it
this way (`priority: low`, `next: confirm a real recurring need... before promoting`) -- I checked that
gate rather than overriding it. Per the build methodology's "earn generality from instances" rule, the
correct move is to leave this as a captured, well-understood candidate (this doc + the register facts
below) and build it when a model actually needs a per-dispatch length change the full-writebd composition
makes awkward -- not before.

## What's expressible today without it

Nothing is blocked. A length change today costs a full BD re-issue (`dma_memcpy_nd`/`dma_configure_task`),
which is more control-program instructions per resize than the scoped op would need, but it is not a
correctness gap -- it works, it's just not the minimal form. That is exactly the "Region 2: expressible,
not minimal" bucket the task file places it in, not a "Region 1: blocked" one.

## Register facts worth keeping (so the next implementer doesn't re-derive them)

Confirmed directly against `third_party/aie-rt` at the pinned submodule commit
(`e2aca220d75965e16912fa1d6b593bb2c22200cb`), aie2p reginit only:

- `BufferLen.Idx = 0U` for all three aie2p tile types (mem tile `xaie2pgbl_reginit.c:1579`,
  core/memory-module tile `:1812`, shim/NOC `:2064`). The `6U` figure a grep of the AIE1 file
  (`xaiegbl_reginit.c:1411`) turns up is gen1, not aie2p -- do not carry it over.
- BD-length field masks (`xaie2pgbl_params.h`): mem tile `0x0001FFFF`, core/memory-module tile
  `0x00003FFF`, shim/NOC `0xFFFFFFFF` -- all LSB 0.
- `NumBds`: mem tile 48 (`xaie2pgbl_reginit.c:1656`), core/memory-module and shim 16
  (`:1893`, `:2144`). Valid BD indices are `0..NumBds-1`; aie-rt's own bound check
  (`XAie_DmaUpdateBdLen`, `xaie_dma.c:2282`, and every sibling BD-bound check in that file) is
  `BdNum > NumBds`, off by one against that range. Do not mirror it.
- Dispatch split (`xaie_dma_aieml.c:1331` / `:1363`): mem/core go through `_XAieMl_DmaUpdateBdLen`
  (`XAie_MaskWrite32`), shim goes through `_XAieMl_ShimDmaUpdateBdLen` (`XAie_Write32`) -- but since the
  shim mask is the full word, the maskwrite32 form is bit-identical there too, so a single lowering path
  covers all three tile types.
