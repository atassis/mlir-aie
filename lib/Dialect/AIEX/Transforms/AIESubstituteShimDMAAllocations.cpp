//===- AIESubstituteShimDMAAllocations.cpp -----------------------*- C++
//-*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <iterator>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIESUBSTITUTESHIMDMAALLOCATIONS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct DMAConfigureTaskForOpPattern
    : public mlir::OpRewritePattern<DMAConfigureTaskForOp> {

  // Build-speed lever (byte-identical): the canonical path resolves each
  // task's shim-DMA allocation via AIE::ShimDMAAllocationOp::getForSymbol ->
  // device.lookupSymbol, a LINEAR symbol-table scan run once per
  // DMAConfigureTaskForOp. At B=128/L12 there are tens of thousands of these,
  // so the per-op O(allocations) scan makes the pass O(n^2) (measured
  // super-linear: L6 40s -> L12 156s = 3.86x for 2x layers). We instead build a
  // SymbolTable for the device ONCE and look up in O(1). Same fix as the
  // AIETargetNPU getDataWords symbol-cache. The pattern never erases/creates a
  // symbol (it rewrites task ops, not ShimDMAAllocationOps), so the prebuilt
  // table stays valid across the greedy run.
  const mlir::SymbolTable &symbolTable;

  DMAConfigureTaskForOpPattern(mlir::MLIRContext *ctx,
                               const mlir::SymbolTable &symbolTable)
      : OpRewritePattern<DMAConfigureTaskForOp>(ctx), symbolTable(symbolTable) {
  }

  LogicalResult matchAndRewrite(DMAConfigureTaskForOp op,
                                PatternRewriter &rewriter) const override {
    AIE::ShimDMAAllocationOp alloc_op =
        symbolTable.lookup<AIE::ShimDMAAllocationOp>(
            op.getAlloc().getRootReference());
    if (!alloc_op) {
      return op.emitOpError("no shim DMA allocation found for symbol");
    }

    AIE::TileOp tile = alloc_op.getTileOp();
    if (!tile) {
      return op.emitOpError(
          "shim DMA allocation must reference a valid TileOp");
    }

    // A shared_input_channel group's members resolve to the same
    // (tile, dir, channel) by construction (assignChannels copied it there;
    // see ObjectFifoLinkOp::verify, AIEDialect.cpp). Each BD gets its OWN
    // member's packet info instead of the task-wide default, matching the
    // chain 1:1, leader first.
    ArrayAttr interleave = op.getInterleaveAttr();
    SmallVector<AIE::PacketInfoAttr> bdPackets;
    if (interleave && !interleave.empty()) {
      bdPackets.push_back(alloc_op.getPacket().value_or(nullptr));
      for (Attribute a : interleave) {
        auto sym = cast<FlatSymbolRefAttr>(a);
        AIE::ShimDMAAllocationOp memberAlloc =
            symbolTable.lookup<AIE::ShimDMAAllocationOp>(sym.getValue());
        if (!memberAlloc)
          return op.emitOpError("no shim DMA allocation found for interleave "
                                "member '")
                 << sym.getValue() << "'";
        if (memberAlloc.getTileOp() != tile ||
            memberAlloc.getChannelDir() != alloc_op.getChannelDir() ||
            memberAlloc.getChannelIndex() != alloc_op.getChannelIndex())
          return op.emitOpError("interleave member '")
                 << sym.getValue()
                 << "' does not share the leader's (tile, direction, "
                    "channel)";
        bdPackets.push_back(memberAlloc.getPacket().value_or(nullptr));
      }
    }

    DMAConfigureTaskOp new_op = DMAConfigureTaskOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(), tile.getResult(),
        alloc_op.getChannelDirAttr(),
        rewriter.getI32IntegerAttr((int32_t)alloc_op.getChannelIndex()),
        rewriter.getBoolAttr(op.getIssueToken()),
        rewriter.getI32IntegerAttr(op.getRepeatCount()),
        /*repeat_count_val=*/op.getRepeatCountVal(),
        alloc_op.getPacket().value_or(nullptr), /*out_of_order=*/nullptr,
        /*fot_mode=*/nullptr);
    rewriter.replaceAllUsesWith(op.getResult(), new_op.getResult());
    rewriter.inlineRegionBefore(op.getBody(), new_op.getBody(),
                                new_op.getBody().begin());

    if (!bdPackets.empty()) {
      size_t i = 0;
      for (auto &block : new_op.getBody())
        block.walk([&](AIE::DMABDOp bd) {
          if (i < bdPackets.size() && bdPackets[i])
            bd.setPacketAttr(bdPackets[i]);
          ++i;
        });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct AIESubstituteShimDMAAllocationsPass
    : xilinx::AIEX::impl::AIESubstituteShimDMAAllocationsBase<
          AIESubstituteShimDMAAllocationsPass> {

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    // Build the device symbol table ONCE (O(1) per-task allocation lookup
    // instead of getForSymbol's per-task linear scan -> O(n^2)).
    // Byte-identical.
    mlir::SymbolTable symbolTable(device);

    // Convert DMAConfigureTaskForOps that reference shim DMA allocations
    // to regular DMAConfigureTaskOps
    RewritePatternSet patterns(&getContext());
    patterns.insert<DMAConfigureTaskForOpPattern>(&getContext(), symbolTable);

    (void)applyPatternsGreedily(device, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIESubstituteShimDMAAllocationsPass() {
  return std::make_unique<AIESubstituteShimDMAAllocationsPass>();
}
