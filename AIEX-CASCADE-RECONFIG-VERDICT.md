# aiex cascade-reconfigure op: regime verdict (do not build yet)

Branch: `aiex-cascade-reconfig`, based on upstream/main @ 4976487604d. Local record only, not for
upstream; not committed to production source. I looked into whether a runtime-sequence op to retarget
cascade routing (`ACCUMULATOR_CONTROL`) between dispatches is worth building right now, and concluded
no. Evidence and the op I would build if that changes are both below.

## The register (grounded in aie-rt, third_party/aie-rt @ e2aca220d75965e16912fa1d6b593bb2c22200cb)

`XAie_CoreConfigAccumulatorControl(DevInst, Loc, InDir, OutDir)` --
`driver/src/core/xaie_core.c:973-1025`. For aie2p the register table is
`Aie2PCoreAccumCtrlReg` in `driver/src/global/xaie2pgbl_reginit.c:158-165`:

```c
static const XAie_RegCoreAccumCtrl Aie2PCoreAccumCtrlReg =
{
    .RegOff = XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL,
    .CascadeInput.Lsb  = XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL_INPUT_LSB,
    .CascadeInput.Mask = XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL_INPUT_MASK,
    .CascadeOutput.Lsb  = XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL_OUTPUT_LSB,
    .CascadeOutput.Mask = XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL_OUTPUT_MASK,
};
```

Offsets/masks, `driver/src/global/xaie2pgbl_params.h:3610-3620`:

```c
#define XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL                 0x00036060
#define XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL_WIDTH           32
#define XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL_MASK            0x00000003
#define XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL_OUTPUT_LSB      1
#define XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL_OUTPUT_MASK     0x00000002
#define XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL_INPUT_LSB       0
#define XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL_INPUT_MASK      0x00000001
```

So on aie2p, core-module-relative offset `0x36060`, a 32-bit register whose only defined content is
bit 0 (`CascadeInput`: 0=North, 1=West) and bit 1 (`CascadeOutput`: 0=South, 1=East) -- everything above
bit 1 is reserved/undefined in the aie2p table (`MASK = 0x3`). It genuinely packs two logical fields
(input direction, output direction) into one register, which is why this family needs `maskwrite32`
mandatory the same way `dma_channel_reset`/`core_reset` did -- even though aie-rt itself does a plain
`XAie_Write32(DevInst, RegAddr, RegVal)` at `xaie_core.c:1024` (safe only because it always sets both
fields together and nothing else is defined in that word on aie2p; an `aiex` op should not inherit that
assumption across generations).

Valid directions per aie-rt: `InDir` must be `NORTH` or `WEST`; `OutDir` must be `SOUTH` or `EAST`
(`xaie_core.c:1002-1006`), matching the AIE-dialect verifier's constraint (see below) -- confirms aie2p
here, not a gen1 table (gen1's `xaiegbl_reginit.c:1155` sets `.CoreAccumCtrl = NULL`, i.e. gen1 core tiles
don't support this at all).

## Who calls this today (grounded in mlir-aie source, this worktree)

Exposed exactly once at the AIE-dialect level: `aie.configure_cascade`
(`ConfigureCascadeOp`, `include/aie/Dialect/AIE/IR/AIEOps.td:1718-1737`), itself normally produced by
lowering `aie.cascade_flow` (`AIELowerCascadeFlows.cpp:80`). Its verifier
(`lib/Dialect/AIE/IR/AIEDialect.cpp:1226-1249`) enforces the same North/West-in, South/East-out
constraint aie-rt enforces at the register level.

It lowers to `XAie_CoreConfigAccumulatorControl` in exactly two places, both device-CONFIGURATION
codegen, neither reachable from `aie.runtime_sequence`:

- `lib/Targets/AIERT.cpp:793-802`, inside `AIERTControl::configureSwitches`, called only from
  `AIERTControl::addInitConfig` (`AIERT.cpp:812`, invoked at `AIERT.cpp:878`), which itself is called
  only from `lib/Targets/AIETargetCDODirect.cpp:96` and `:125` -- the initial-configuration CDO
  generator (the PDI/xclbin config blob, built once per device-image load).
- `lib/Targets/AIETargetXAIEV2.cpp:764-777`, the legacy XAIEv2 C host-code generator's
  `mlir_aie_configure_cascade`.

`grep -rniI cascade include/aie/Dialect/AIEX lib/Dialect/AIEX` returns zero hits -- the whole AIEX
runtime-op family (the thing `aiex.core_reset`/`aiex.dma_channel_reset` belong to, and the thing
`aie.runtime_sequence` lowers through) has no cascade awareness at all today. Cascade routing direction
is fixed at device-configuration (CDO/xclbin build) time and can only change via a fresh configuration
load. This matches the private-KB red-team finding this task was seeded from
(`journal/docs/tasks/done/red-team-completeness-map.md`, finding A1) -- I independently re-derived it
from source rather than trusting the prior note, and it holds on the register/call-site facts.

Where I diverge from A1: A1 tags this `LAYER+REGIME, high` and its `next:` field says to promote it
straight to a candidate. The register facts are LAYER (the mechanism exists a layer down and was never
promoted); the priority call is not, once you ask whether anything needs it -- see the regime analysis
below. This verdict supersedes A1's priority tag with a REGIME-only, non-gap classification. I've closed
the loop on the originating task (`journal/docs/tasks/done/aiex-cascade-reconfigure-op.md`, now
`state: done` with a Resolution section) and on `red-team-completeness-map.md`'s own `next:` field and A1
bullet, rather than leaving this doc as the only record of the decision.

One more scoping note: the originating task doc justifies its high priority/capture by citing "the brick
catalog independently rates fused cascade-accumulator the #1 toolchain-capture item"
(`reference/aie2p-brick-catalog.md:183`). That citation is about a different mechanism -- the in-datapath
cascade-ADD lowering (`npu_cascade` doing buffer-copy + software-add instead of `get_scd()+local ->
put_mcd`, tracked in `upstream-cascade-accumulator-rfc.md`), not about retargeting cascade routing
direction between dispatches (`ACCUMULATOR_CONTROL`, what this doc evaluates). It is a misattribution for
this specific op; don't read it as a live high-capture argument for building `cascade_reconfigure`.

## Regime question: does any planned dataflow need this without a full reconfigure?

No, on the evidence I could find in the private KB (`xdna-engine-private/journal/docs/`):

- Cascade direction is a placement property, not a per-dispatch knob: `aie.cascade_flow(src, dst)`
  requires `dst` East or South of `src` (`method-aie2p-device-test-build.md`), i.e. it describes which
  physical neighbor tile a core's accumulator talks to. That is fixed by the compiled kernel's spatial
  layout, not something a resident per-token decode loop would flip while keeping the same ELF/config
  resident.
- The one place cascade is used for real, `cascade_ffn` (`route_b_kernels/cascade_ffn/`), keeps one
  fixed chain shape for its whole resident lifetime. The open work around it
  (`upstream-cascade-accumulator-rfc.md`) is about making that FIXED chain's reduction happen in-datapath
  (`get_scd()+local -> put_mcd`) instead of buffer-copy + software-add -- an efficiency change to the
  existing wiring, not a request to rewire it between dispatches.
- Cascade LOCK re-arming across re-dispatch is a real, already-solved problem
  (`fused-ffn-phase0-gate.md`: single-trip `npu_cascade` aborted on dispatch #2 pre-fix), but that is
  `mlir-air#1694`'s lock-relock mechanism, a different register/resource than `ACCUMULATOR_CONTROL`
  entirely. It does not need this op.
- `cascade-decode-onchip.md` (the on-chip cascade decode dataflow that would most plausibly have wanted
  dynamic retargeting) is PARKED: measured to tie-or-lose against the existing dispatch-stitched form on
  silicon, and -- more to the point here -- found the real ~78% inter-op cost lives in attention stalls,
  which cascade retargeting does not touch.
- `resident-dataflow-lifecycle-flagship.md`, the actual design doc for "what needs a compiler-managed
  re-arm across resident dispatches," enumerates exactly `dma_channel_reset` + `core_reset` as the
  composition. Cascade is absent from that list -- the team that scoped the resident re-arm lifecycle did
  not find a cascade-retarget need either.
- The originating task note itself already carries a related verified-negative: "the accumulator itself
  is NOT a separate resource... Verified negative result -- do not chase it separately."

LAYER vs REGIME: this is REGIME, not LAYER. The op is not "missing because AMD targeted batch and never
promoted a per-dispatch primitive" (that would be LAYER, like `dma_channel_reset` was) -- it is missing
because cascade routing is architecturally a placement-time property everywhere it is used today, and no
built or planned dataflow reuses the same physical cascade-wired tiles for a second, differently-shaped
chain within one residency period. Building the op now would be a mechanism with no caller.

## Verdict

Do not build `aiex.cascade_reconfigure` yet. Nothing in the current or planned dataflow needs cascade
retargeting without a full reconfigure; a full reconfigure (fresh CDO/config load) is what every planned
use of cascade already assumes. Revisit only if a specific dataflow design shows up that reuses the same
physical cascade chain for two different topologies within one resident load -- I don't see one on the
roadmap right now.

## Op sketch (not implemented, for reference if the regime changes)

If a real caller shows up, the shape would follow `aiex.core_reset` / `aiex.dma_channel_reset` exactly:

```tablegen
def AIEX_CascadeReconfigureOp: AIEX_Op<"cascade_reconfigure", [
    HasParent<"AIE::RuntimeSequenceOp">, SkipAccessibilityCheck]> {
  let summary = "Retarget the cascade input/output routing of a core between dispatches";
  let arguments = (ins
    Index:$tile,
    CascadeDir:$inputDir,
    CascadeDir:$outputDir
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{ `(` $tile `,` $inputDir `,` $outputDir `)` attr-dict }];
  let extraClassDeclaration = [{
    xilinx::AIE::TileOp getTileOp();
  }];
}
```

Verifier: reject non-AIE-tile targets (shim/memtile have no cascade port, same check
`ConfigureCascadeOp::verify` already does), and reject `inputDir` not in {North, West} / `outputDir` not
in {South, East} for `AIE2TargetModel` -- copy `AIEDialect.cpp:1226-1249` almost verbatim.

Lowering (`AIELowerCascadeReconfigure.cpp`, an `OpConversionPattern` matching `AIELowerCoreReset.cpp`'s
shape): rewrite to one `npu.maskwrite32` per tile, at core-module-relative offset
`0x36060` from the tile's base address, `mask = 0x3`, `value` built by placing the input-direction bit
(0=North,1=West) at bit 0 and the output-direction bit (0=South,1=East) at bit 1 -- i.e. reproduce
`XAie_SetField` from `xaie_core.c:1017-1022` at compile time instead of runtime. `maskwrite32` is
correct here (not `write32`) because the register packs two logical fields and I should not assume no
generation ever defines bits 2-31; that only matters if this ever gets built, since aie-rt's own runtime
write is unconditional today.

Pass registration would go in `AIEXPasses.td` next to `AIELowerCoreReset`/`AIELowerDmaChannelReset`.
Test pair: `test/Passes/lower-cascade-reconfigure/{cascade_reconfigure.mlir,cascade_reconfigure_invalid.mlir}`
plus an `npu-xrt` device test under `test/npu-xrt/local_reset/cascade_reconfigure_op/`, modeled on
`core_reset_op/`.
