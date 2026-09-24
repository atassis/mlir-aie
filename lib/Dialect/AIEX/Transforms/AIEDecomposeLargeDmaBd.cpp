//===- AIEDecomposeLargeDmaBd.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decomposes oversized non-contiguous aiex.npu.dma_memcpy_nd ops and task-path
// aie.dma_bd ops into one or more hardware-legal ND transfers before
// aie-dma-to-npu / aie-dma-tasks-to-npu lowering.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/BdLowering.h"
#include "aie/Dialect/AIEX/Utils/DmaDecomposition.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

#include <chrono>
#include <cstdlib>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEDECOMPOSELARGEDMABD
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

// Progress heartbeat; see the identical helper's comment in
// AIEMaterializeRuntimeSequences.cpp. Same env var, same semantics, kept as
// a file-local copy rather than a shared header for two call sites.
void traceProgressIfEnabled(const char *label, long &counter,
                            std::chrono::steady_clock::time_point &start,
                            bool &started) {
  static const long every = [] {
    const char *v = getenv("AIE_TRACE_PROGRESS_EVERY");
    return v ? std::atol(v) : 0;
  }();
  if (every <= 0)
    return;
  if (!started) {
    start = std::chrono::steady_clock::now();
    started = true;
  }
  if (++counter % every == 0) {
    double elapsedS =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    llvm::errs() << "[" << label << "] " << counter << " done, elapsed "
                 << elapsedS << "s\n";
  }
}

static bool allConstant(NpuDmaMemcpyNdOp op) {
  return llvm::all_of(op.getMixedSizes(),
                      [](OpFoldResult s) {
                        return getConstantIntValue(s).has_value();
                      }) &&
         llvm::all_of(op.getMixedStrides(),
                      [](OpFoldResult s) {
                        return getConstantIntValue(s).has_value();
                      }) &&
         llvm::all_of(op.getMixedOffsets(), [](OpFoldResult s) {
           return getConstantIntValue(s).has_value();
         });
}

static bool allConstant(AIE::DMABDOp op) {
  if (op.getPadDimensions().has_value())
    return false;
  if (op.getOffset() && !op.getConstantOffset())
    return false;
  if (op.getLen() && !op.getConstantLen())
    return false;
  if (op.getMixedSizes().empty())
    return false;
  return llvm::all_of(op.getMixedSizes(),
                      [](OpFoldResult s) {
                        return getConstantIntValue(s).has_value();
                      }) &&
         llvm::all_of(op.getMixedStrides(), [](OpFoldResult s) {
           return getConstantIntValue(s).has_value();
         });
}

static NdDmaPattern patternFromOp(NpuDmaMemcpyNdOp op) {
  NdDmaPattern pattern;
  pattern.offsets = llvm::map_to_vector(
      llvm::reverse(op.getMixedOffsets()),
      [](OpFoldResult s) { return getConstantIntValue(s).value(); });
  pattern.sizes = llvm::map_to_vector(
      llvm::reverse(op.getMixedSizes()),
      [](OpFoldResult s) { return getConstantIntValue(s).value(); });
  pattern.strides = llvm::map_to_vector(
      llvm::reverse(op.getMixedStrides()),
      [](OpFoldResult s) { return getConstantIntValue(s).value(); });
  return pattern;
}

// Only called after allConstant(op) confirmed every size/stride resolves to
// a constant, so the has_value() checks below are redundant in practice --
// asserted rather than re-verified to keep that invariant visible here too.
static NdDmaPattern patternFromDmaBd(AIE::DMABDOp op) {
  SmallVector<int64_t, 4> outerSizes;
  SmallVector<int64_t, 4> outerStrides;
  for (OpFoldResult s : op.getMixedSizes()) {
    auto c = getConstantIntValue(s);
    assert(c && "size must be constant (already checked by allConstant)");
    outerSizes.push_back(*c);
  }
  for (OpFoldResult s : op.getMixedStrides()) {
    auto c = getConstantIntValue(s);
    assert(c && "stride must be constant (already checked by allConstant)");
    outerStrides.push_back(*c);
  }
  while (outerSizes.size() < 4) {
    outerSizes.insert(outerSizes.begin(), 1);
    outerStrides.insert(outerStrides.begin(), 0);
  }

  NdDmaPattern pattern;
  pattern.offsets = {0, 0, 0, 0};
  pattern.sizes = llvm::map_to_vector(llvm::reverse(outerSizes),
                                      [](int64_t v) { return v; });
  pattern.strides = llvm::map_to_vector(llvm::reverse(outerStrides),
                                        [](int64_t v) { return v; });
  return pattern;
}

static SmallVector<int64_t, 4> toOuter(ArrayRef<int64_t> inner) {
  return llvm::map_to_vector(llvm::reverse(inner), [](int64_t v) { return v; });
}

static int64_t flatOffsetFromPattern(int64_t baseFlatOffset,
                                     const NdDmaPattern &pattern) {
  int64_t flat = baseFlatOffset;
  for (unsigned k = 0; k < 4; ++k)
    flat += pattern.offsets[k] * pattern.strides[k];
  return flat;
}

static int64_t lenFromInnermost3(ArrayRef<int64_t> sizesInnermostFirst) {
  int64_t len = 1;
  for (unsigned i = 0; i < 3; ++i)
    len *= sizesInnermostFirst[i];
  return len;
}

static AIE::BDDimLayoutArrayAttr outerDimsAttr(MLIRContext *ctx,
                                               ArrayRef<int64_t> outerSizes,
                                               ArrayRef<int64_t> outerStrides) {
  SmallVector<AIE::BDDimLayoutAttr> dims;
  dims.reserve(outerSizes.size());
  for (auto [s, t] : llvm::zip(outerSizes, outerStrides))
    dims.push_back(AIE::BDDimLayoutAttr::get(ctx, static_cast<uint32_t>(s),
                                             static_cast<uint32_t>(t)));
  return AIE::BDDimLayoutArrayAttr::get(ctx, dims);
}

static void updateTaskBdInPlace(AIE::DMABDOp bd, int32_t offset, int32_t len,
                                ArrayRef<int64_t> outerSizes,
                                ArrayRef<int64_t> outerStrides) {
  bd.getOffsetMutable().clear();
  bd.setStaticOffset(offset);
  bd.getLenMutable().clear();
  bd.setStaticLen(len);
  bd.getSizesMutable().clear();
  bd.getStridesMutable().clear();
  bd.setStaticSizes(DenseI64ArrayAttr::get(bd.getContext(), outerSizes));
  bd.setStaticStrides(DenseI64ArrayAttr::get(bd.getContext(), outerStrides));
}

static AIE::DMABDOp createTaskBd(PatternRewriter &rewriter, Location loc,
                                 AIE::DMABDOp tmpl, int32_t offset, int32_t len,
                                 ArrayRef<int64_t> outerSizes,
                                 ArrayRef<int64_t> outerStrides) {
  auto dims = outerDimsAttr(rewriter.getContext(), outerSizes, outerStrides);
  auto bd =
      AIE::DMABDOp::create(rewriter, loc, tmpl.getBuffer(), offset, len, dims);
  if (tmpl.getPacketAttr())
    bd.setPacketAttr(tmpl.getPacketAttr());
  if (tmpl.getBurstLengthAttr())
    bd.setBurstLengthAttr(tmpl.getBurstLengthAttr());
  if (tmpl.getAxcacheAttr())
    bd.setAxcacheAttr(tmpl.getAxcacheAttr());
  if (tmpl.getOffsetParameterAttr())
    bd.setOffsetParameterAttr(tmpl.getOffsetParameterAttr());
  // length_parameter is never copied here: splitting a length_parameter BD
  // into several is rejected before this is reached.
  // out_of_order_id is not copied because slicing an OoO BD is rejected above.
  return bd;
}

static unsigned countTaskBds(Operation *taskOp) {
  unsigned n = 0;
  taskOp->walk([&](AIE::DMABDOp) { ++n; });
  return n;
}

static Region *getTaskBody(Operation *taskOp) {
  if (auto cfg = dyn_cast<DMAConfigureTaskOp>(taskOp))
    return &cfg.getBody();
  if (auto cfgFor = dyn_cast<DMAConfigureTaskForOp>(taskOp))
    return &cfgFor.getBody();
  return nullptr;
}

static bool isUnderRuntimeControlFlow(AIE::DMABDOp op) {
  auto seq = op->getParentOfType<AIE::RuntimeSequenceOp>();
  if (!seq)
    return false;
  for (Operation *parent = op->getParentOp(); parent && parent != seq;
       parent = parent->getParentOp()) {
    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(parent))
      return true;
  }
  return false;
}

static std::optional<std::pair<AIE::TileOp, Operation *>>
resolveTaskAndTile(AIE::DMABDOp op, const mlir::SymbolTable &symbolTable) {
  if (auto cfg = op->getParentOfType<DMAConfigureTaskOp>()) {
    // Decomposition is a shape rewrite, so an unplaced tile is not an error
    // here: decline and let the pattern run again after placement.
    AIE::TileOp tile = cfg.tryGetTileOp();
    if (!tile)
      return std::nullopt;
    return std::make_pair(tile, cfg.getOperation());
  }
  if (auto cfgFor = op->getParentOfType<DMAConfigureTaskForOp>()) {
    // symbolTable.lookup instead of ShimDMAAllocationOp::getForSymbol's
    // per-call linear device scan -- see the pass's own symbolTable comment.
    auto allocOp = symbolTable.lookup<AIE::ShimDMAAllocationOp>(
        cfgFor.getAlloc().getRootReference());
    if (!allocOp)
      return std::nullopt;
    AIE::TileOp tile = allocOp.getTileOp();
    if (!tile)
      return std::nullopt;
    return std::make_pair(tile, cfgFor.getOperation());
  }
  return std::nullopt;
}

static NpuDmaMemcpyNdOp createDecomposedOp(PatternRewriter &rewriter,
                                           NpuDmaMemcpyNdOp op,
                                           const NdDmaPattern &pattern,
                                           int64_t id, bool issueToken) {
  auto outerOffsets = toOuter(pattern.offsets);
  auto outerSizes = toOuter(pattern.sizes);
  auto outerStrides = toOuter(pattern.strides);

  return NpuDmaMemcpyNdOp::create(
      rewriter, op.getLoc(), op.getMemref(),
      /*offsets=*/ValueRange{}, /*sizes=*/ValueRange{},
      /*strides=*/ValueRange{},
      DenseI64ArrayAttr::get(op.getContext(), outerOffsets),
      DenseI64ArrayAttr::get(op.getContext(), outerSizes),
      DenseI64ArrayAttr::get(op.getContext(), outerStrides), op.getPacketAttr(),
      op.getMetadata(), rewriter.getI64IntegerAttr(id),
      rewriter.getBoolAttr(issueToken), op.getD0ZeroBeforeAttr(),
      op.getD1ZeroBeforeAttr(), op.getD2ZeroBeforeAttr(),
      op.getD0ZeroAfterAttr(), op.getD1ZeroAfterAttr(), op.getD2ZeroAfterAttr(),
      op.getBurstLengthAttr(), op.getAxcacheAttr(), op.getOffsetParameterAttr(),
      op.getOffsetStateTableIdxAttr(), op.getLengthParameterAttr(),
      op.getLengthStateTableIdxAttr(), op.getLengthGranuleAttr());
}

static int64_t allocateNextId(NpuDmaMemcpyNdOp op, int64_t startId,
                              llvm::DenseSet<int64_t> &usedIds) {
  int64_t id = startId;
  while (usedIds.contains(id))
    ++id;
  usedIds.insert(id);
  return id;
}

struct DecomposeLargeDmaBdPattern : OpRewritePattern<NpuDmaMemcpyNdOp> {
  const mlir::SymbolTable &symbolTable;

  DecomposeLargeDmaBdPattern(MLIRContext *ctx,
                             const mlir::SymbolTable &symbolTable)
      : OpRewritePattern<NpuDmaMemcpyNdOp>(ctx), symbolTable(symbolTable) {}

  // used-id sets, keyed by enclosing RuntimeSequenceOp then by metadata,
  // populated by one walk per sequence the first time it is matched against
  // instead of one walk per decomposed op (same anti-pattern/fix shape as the
  // getOrCreateDataMemref cache, #3212). Mutable: matchAndRewrite is const.
  mutable llvm::DenseMap<Operation *,
                        llvm::DenseMap<Attribute, llvm::DenseSet<int64_t>>>
      usedIdsCache;

  // See traceProgressIfEnabled's comment.
  mutable long tracedDecomposeCount = 0;
  mutable std::chrono::steady_clock::time_point traceStart;
  mutable bool traceStarted = false;

  llvm::DenseMap<Attribute, llvm::DenseSet<int64_t>> &
  getUsedIdsForSeq(AIE::RuntimeSequenceOp seq) const {
    auto [it, inserted] = usedIdsCache.try_emplace(seq.getOperation());
    if (!inserted)
      return it->second;
    seq.walk([&](NpuDmaMemcpyNdOp other) {
      it->second[other.getMetadataAttr()].insert(other.getId());
    });
    return it->second;
  }

  LogicalResult matchAndRewrite(NpuDmaMemcpyNdOp op,
                                PatternRewriter &rewriter) const override {
    if (!allConstant(op))
      return failure();

    NdDmaPattern pattern = patternFromOp(op);
    // A length_parameter op must reach the "not implemented" diagnostic below
    // even when isContiguousTransfer misreads its size-1 d2 dimension as
    // truly contiguous (see BdLowering.h's hasLengthParameter).
    if (isContiguousTransfer(pattern.sizes, pattern.strides) &&
        !hasLengthParameter(op))
      return failure();

    // symbolTable.lookup instead of ShimDMAAllocationOp::getForSymbol's
    // per-call linear device scan -- see the pass's own symbolTable comment.
    auto allocOp =
        symbolTable.lookup<AIE::ShimDMAAllocationOp>(op.getMetadata().getRootReference());
    if (!allocOp)
      return failure();

    AIE::TileOp tile = allocOp.getTileOp();
    if (!tile)
      return failure();

    int col = tile.getCol();
    int row = tile.getRow();
    const AIE::AIETargetModel &targetModel = AIE::getTargetModel(op);
    auto bufferType = cast<BaseMemRefType>(op.getMemref().getType());

    if (patternPassesVerification(op, bufferType, targetModel, col, row,
                                  pattern))
      return failure();

    auto decomposed =
        decomposeNdDmaPattern(op, bufferType, pattern, targetModel, col, row);
    // failed() already guards both dereferences below via short-circuit /
    // prior control flow; the checker just doesn't associate FailureOr's
    // failed()/succeeded() idiom with the std::optional base it derives from.
    if (failed(decomposed) ||
        decomposed->empty()) // NOLINT(bugprone-unchecked-optional-access)
      return failure();
    // Bind a plain reference now that decomposed is known non-failed and
    // non-empty, so nothing past this point looks like an optional access.
    SmallVector<NdDmaPattern> &bds =
        *decomposed; // NOLINT(bugprone-unchecked-optional-access)
    if (bds.size() > targetModel.getNumBDs(col, row))
      return failure();

    // See emitUpdateBdLengthFromParameter: splitting would patch one slice only.
    if (bds.size() > 1 && op.getLengthParameterAttr())
      return op.emitOpError()
             << "splitting a length_parameter buffer descriptor into "
                "multiple descriptors is not implemented";

    if (bds.size() == 1) {
      rewriter.replaceOpWithNewOp<NpuDmaMemcpyNdOp>(
          op, op.getMemref(), ValueRange{}, ValueRange{}, ValueRange{},
          DenseI64ArrayAttr::get(op.getContext(), toOuter(bds.front().offsets)),
          DenseI64ArrayAttr::get(op.getContext(), toOuter(bds.front().sizes)),
          DenseI64ArrayAttr::get(op.getContext(), toOuter(bds.front().strides)),
          op.getPacketAttr(), op.getMetadata(), op.getIdAttr(),
          op.getIssueTokenAttr(), op.getD0ZeroBeforeAttr(),
          op.getD1ZeroBeforeAttr(), op.getD2ZeroBeforeAttr(),
          op.getD0ZeroAfterAttr(), op.getD1ZeroAfterAttr(),
          op.getD2ZeroAfterAttr(), op.getBurstLengthAttr(), op.getAxcacheAttr(),
          op.getOffsetParameterAttr(), op.getOffsetStateTableIdxAttr(),
          op.getLengthParameterAttr(), op.getLengthStateTableIdxAttr(),
          op.getLengthGranuleAttr());
      return success();
    }

    llvm::DenseSet<int64_t> fallbackUsedIds;
    llvm::DenseSet<int64_t> *usedIds = &fallbackUsedIds;
    if (auto seq = op->getParentOfType<AIE::RuntimeSequenceOp>()) {
      usedIds = &getUsedIdsForSeq(seq)[op.getMetadataAttr()];
      // op is about to be erased -- its own id is not a collision to avoid,
      // and excluding it lets the first decomposed sub-op below reuse it.
      usedIds->erase(op.getId());
    }

    int64_t nextId = op.getId();
    rewriter.setInsertionPoint(op);
    for (auto [idx, subPattern] : llvm::enumerate(bds)) {
      bool last = idx + 1 == bds.size();
      int64_t id = allocateNextId(op, nextId, *usedIds);
      nextId = id + 1;
      createDecomposedOp(rewriter, op, subPattern, id,
                         last && op.getIssueToken());
    }
    rewriter.eraseOp(op);
    traceProgressIfEnabled("aie-decompose-dma: multi-BD ops decomposed",
                           tracedDecomposeCount, traceStart, traceStarted);
    return success();
  }
};

struct DecomposeLargeDmaBdTaskPattern : OpRewritePattern<AIE::DMABDOp> {
  const mlir::SymbolTable &symbolTable;

  DecomposeLargeDmaBdTaskPattern(MLIRContext *ctx,
                                 const mlir::SymbolTable &symbolTable)
      : OpRewritePattern<AIE::DMABDOp>(ctx), symbolTable(symbolTable) {}

  LogicalResult matchAndRewrite(AIE::DMABDOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<AIE::MemOp>() ||
        op->getParentOfType<AIE::ShimDMAOp>() ||
        op->getParentOfType<AIE::MemTileDMAOp>() ||
        op->getParentOfType<AIE::DMAOp>())
      return failure();

    auto taskAndTile = resolveTaskAndTile(op, symbolTable);
    if (!taskAndTile)
      return failure();

    AIE::TileOp tile = taskAndTile->first;
    Operation *taskOp = taskAndTile->second;

    if (!allConstant(op))
      return failure();
    if (countTaskBds(taskOp) != 1)
      return failure();

    NdDmaPattern pattern = patternFromDmaBd(op);
    // See the NpuDmaMemcpyNdOp pattern above.
    if (isContiguousTransfer(pattern.sizes, pattern.strides) &&
        !hasLengthParameter(op))
      return failure();

    int col = tile.getCol();
    int row = tile.getRow();
    const AIE::AIETargetModel &targetModel = AIE::getTargetModel(op);
    auto bufferType = cast<BaseMemRefType>(op.getBuffer().getType());

    if (patternPassesVerification(op, bufferType, targetModel, col, row,
                                  pattern))
      return failure();

    auto decomposed =
        decomposeNdDmaPattern(op, bufferType, pattern, targetModel, col, row);
    // failed() already guards both dereferences below via short-circuit /
    // prior control flow; the checker just doesn't associate FailureOr's
    // failed()/succeeded() idiom with the std::optional base it derives from.
    if (failed(decomposed) ||
        decomposed->empty()) // NOLINT(bugprone-unchecked-optional-access)
      return failure();
    // Bind a plain reference now that decomposed is known non-failed and
    // non-empty, so nothing past this point looks like an optional access.
    SmallVector<NdDmaPattern> &bds =
        *decomposed; // NOLINT(bugprone-unchecked-optional-access)
    if (bds.size() > targetModel.getNumBDs(col, row))
      return failure();

    if (bds.size() > 1 && isUnderRuntimeControlFlow(op)) {
      op.emitRemark()
          << "deferring multi-BD decomposition under runtime control flow "
             "(dynamic BD pool supports single-BD tasks only)";
      return failure();
    }

    int64_t baseFlatOffset = op.getConstantOffset().value_or(0);

    if (bds.size() == 1) {
      const NdDmaPattern &sub = bds.front();
      int64_t flatOffset = flatOffsetFromPattern(baseFlatOffset, sub);
      auto outerSizes = toOuter(sub.sizes);
      auto outerStrides = toOuter(sub.strides);
      rewriter.modifyOpInPlace(op, [&]() {
        op.getOffsetMutable().clear();
        op.setStaticOffset(static_cast<int32_t>(flatOffset));
        op.getSizesMutable().clear();
        op.getStridesMutable().clear();
        op.setStaticSizes(DenseI64ArrayAttr::get(op.getContext(), outerSizes));
        op.setStaticStrides(
            DenseI64ArrayAttr::get(op.getContext(), outerStrides));
      });
      return success();
    }

    // A chain split would make every slice reuse the one out_of_order_id slot.
    // TODO: BD iteration plus padding may enable this feature.
    if (op.getOutOfOrderId().has_value())
      return op.emitOpError() << "splitting an out-of-order buffer descriptor "
                                 "into multiple descriptors is not implemented";

    // See emitUpdateBdLengthFromParameter: splitting would patch one slice only.
    if (op.getLengthParameterAttr())
      return op.emitOpError()
             << "splitting a length_parameter buffer descriptor into "
                "multiple descriptors is not implemented";

    Region *body = getTaskBody(taskOp);
    if (!body || body->empty())
      return failure();

    SmallVector<Block *> blocks;
    blocks.push_back(op->getBlock());
    for (unsigned i = 1; i < bds.size(); ++i)
      blocks.push_back(rewriter.createBlock(body));

    for (auto [idx, subPattern] : llvm::enumerate(bds)) {
      Block *block = blocks[idx];
      int64_t flatOffset = flatOffsetFromPattern(baseFlatOffset, subPattern);
      auto outerSizes = toOuter(subPattern.sizes);
      auto outerStrides = toOuter(subPattern.strides);
      int32_t len = static_cast<int32_t>(lenFromInnermost3(subPattern.sizes));

      if (idx == 0) {
        rewriter.modifyOpInPlace(op, [&]() {
          updateTaskBdInPlace(op, static_cast<int32_t>(flatOffset), len,
                              outerSizes, outerStrides);
        });
        Operation *oldTerm = block->getTerminator();
        rewriter.setInsertionPoint(oldTerm);
        if (idx + 1 < bds.size())
          AIE::NextBDOp::create(rewriter, op.getLoc(), blocks[idx + 1]);
        else
          AIE::EndOp::create(rewriter, op.getLoc());
        rewriter.eraseOp(oldTerm);
      } else {
        rewriter.setInsertionPointToStart(block);
        createTaskBd(rewriter, op.getLoc(), op,
                     static_cast<int32_t>(flatOffset), len, outerSizes,
                     outerStrides);
        rewriter.setInsertionPointToEnd(block);
        if (idx + 1 < bds.size())
          AIE::NextBDOp::create(rewriter, op.getLoc(), blocks[idx + 1]);
        else
          AIE::EndOp::create(rewriter, op.getLoc());
      }
    }

    return success();
  }
};

struct AIEDecomposeLargeDmaBdPass
    : xilinx::AIEX::impl::AIEDecomposeLargeDmaBdBase<
          AIEDecomposeLargeDmaBdPass> {
  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    // Built once: neither pattern creates, erases or renames a
    // ShimDMAAllocationOp (both only query one), so this stays valid across
    // the whole greedy run -- same contract as AIESubstituteShimDMAAllocations's
    // own device-wide SymbolTable, replacing getForSymbol's per-call linear
    // scan (O(top-level symbols) per candidate op, confirmed live at
    // ~90% of pass samples on a real batched-prefill build, 2026-09-13).
    mlir::SymbolTable symbolTable(device);
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeLargeDmaBdPattern, DecomposeLargeDmaBdTaskPattern>(
        &getContext(), symbolTable);
    if (failed(applyPatternsGreedily(device, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEDecomposeLargeDmaBdPass() {
  return std::make_unique<AIEDecomposeLargeDmaBdPass>();
}
