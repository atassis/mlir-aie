# Red-team remediation notes (2026-07-27)

Record of what I checked against the red-team report and what I did about each finding. No code lands
in this pass -- see UPSTREAM-CASE.md for why (queued behind `aiex-write-field-abstraction`) and
OUR-JUSTIFICATION.md for the YAGNI verdict (speculative, not demonstrated).

## Finding 1 (major, premise) -- zero implementation delivered

Confirmed the factual claims: `wt-op-bd-length` was clean, on `aiex-bd-length`, tracking
`upstream/main`, with no diff -- verified independently before touching anything. Also confirmed
`wt-op-bd-length` is a registered worktree of the same `mlir-aie` repo as the pinned session cwd
(`git worktree list` from `mlir-aie/` lists both `wt-op-bd-length` and `wt-op-write-field`).

The report's suggested fix -- "re-run the task with the session explicitly entered into
`wt-op-bd-length` via `EnterWorktree(path=...)`" -- does not actually work for a pinned subagent: I
tried it and got "Cannot enter worktree: the current working directory ... is not inside the
repository at .../mlir-aie", because a subagent's pinned cwd only supports `EnterWorktree(path=...)`
into worktrees under `.claude/worktrees/` of the same repo; `wt-op-bd-length` was created outside that
convention (a sibling directory registered via plain `git worktree add`). This is a real tool
limitation, worth reporting as the swarm-launch nuance the original implementer's open question
suspected -- but it does not make the worktree unreachable. All `Read`/`Write`/`Edit`/`Bash -C` calls in
this session used the worktree's absolute path directly, which works fine regardless of which directory
the session's cwd is pinned to. That's what I did for every file in this pass.

So: the finding's substance (real, safe work was available and unused) holds. The specific mechanism it
named to fix it doesn't work as stated; the actual fix is "use absolute paths, don't rely on
`EnterWorktree` for a sibling worktree outside `.claude/worktrees/`."

## Finding 2 (major, verifier-gap) -- aie-rt BdNum off-by-one

Confirmed directly: `XAie_DmaUpdateBdLen` (`third_party/aie-rt/driver/src/dma/xaie_dma.c:2282`) checks
`BdNum > DmaMod->NumBds`, and mem-tile `NumBds = 48` (`xaie2pgbl_reginit.c:1656`) with valid indices
0..47. `BdNum == 48` passes aie-rt's own check. Recorded in UPSTREAM-CASE.md as an explicit design
constraint (`bd_id < NumBds`, not `<=` / not mirroring aie-rt's `>`), with a note for the reviewer not to
"fix" it back to match aie-rt.

## Finding 3 (minor, register-truth) -- BufferLen.Idx

Confirmed: `BufferLen.Idx = 0U` for all three aie2p tile types (`xaie2pgbl_reginit.c:1579`, `:1812`,
`:2064`). The `6U` the original implementer flagged as unverified is `xaiegbl_reginit.c:1411`, the AIE1
(gen1) file -- not aie2p. Recorded in OUR-JUSTIFICATION.md so the next implementer doesn't re-derive it
or accidentally carry the gen1 value over.

## Finding 4 (minor, convention-fit) -- don't fork a write32 branch for shim

Confirmed: `XAIE2PGBL_NOC_MODULE_DMA_BD0_0_BUFFER_LENGTH_MASK = 0xFFFFFFFF` (`xaie2pgbl_params.h:16308`),
a full 32-bit mask, so `maskwrite32` and `write32` are bit-identical there. aie-rt's C driver still
forks (`_XAieMl_ShimDmaUpdateBdLen` uses `XAie_Write32`,
`xaie_dma_aieml.c:1363`) but that's a C-driver convenience, not a hardware requirement. Agreed this
should be a single lowering path (`maskwrite32` uniformly), not a mirrored fork -- recorded in
UPSTREAM-CASE.md. This is also the concrete reason to wait for `aiex-write-field-abstraction`: hand-
rolling three RegField constants now to get this right, then re-deriving them from the abstraction's
table once it lands, is duplicated work for no benefit.

## What I did not do

Did not touch `xdna-engine-private/journal/docs/tasks/aiex-bd-length-resize-op.md`. That repo is a
single shared checkout (not a per-task worktree), and at the time of this session it already had other
uncommitted changes in flight from a different task/session (`aiex-cascade-reconfigure-op.md` moved to
`done/`, `runtime-op-completeness.md` and `red-team-completeness-map.md` modified, and the bd-length
task file itself sitting untracked) -- editing it here would risk clobbering or racing that concurrent
work with no worktree isolation. This is the same shared-checkout hazard the standing rules already name
for the amd/IRON checkout; it applies here too. The register facts and the defer decision are recorded
in this worktree instead (OUR-JUSTIFICATION.md); whoever next touches the private task file should fold
this doc's "Register facts worth keeping" section into its body as a Worklog entry.

## Net decision

`recommend: defer`. Feasibility is clean (no silicon/toolchain wall), the register model is fully
understood and confirmed above, and the design in UPSTREAM-CASE.md is ready to implement in a few lines
-- once `aiex-write-field-abstraction` lands and a target model actually needs a per-dispatch length
change. Neither condition holds today. Building it now would mean hand-coding three RegField constants
a sibling worktree is actively generalizing, against a need nothing on the current model roster has hit.
That's forcing the op into existence ahead of both its prerequisite and its justification -- exactly what
the task file's own YAGNI gate says not to do.
