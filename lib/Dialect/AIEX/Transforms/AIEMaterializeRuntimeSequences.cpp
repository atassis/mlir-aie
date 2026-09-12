//===- AIEMaterializeRuntimeSequences.cpp -----------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Targets/AIERT.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

#include <chrono>
#include <cstdlib>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEMATERIALIZERUNTIMESEQUENCES
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

#define DEBUG_TYPE "aie-materialize-runtime-sequence"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

// Progress heartbeat for monitoring a long-running run from outside the
// process: this pass has no per-op signal otherwise, so a build stuck here
// for tens of minutes looks identical to one making steady progress. Off
// (zero perturbation) unless AIE_TRACE_PROGRESS_EVERY is a positive integer;
// then emits one stderr line every that many matches, with an elapsed timer
// started on first use. `label` distinguishes multiple call sites sharing
// the counter's caller-provided storage.
static void traceProgressIfEnabled(const char *label, long &counter,
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

struct RuntimeCallGraphCyclicityAnalysis {
  AnalysisManager &analysisManager;

  // if invalid, analysis failed and results should not be considered
  bool isValid = false;

  // Call graph is cyclic
  bool isCyclic = false;

  RuntimeCallGraphCyclicityAnalysis(Operation *op, AnalysisManager &am)
      : analysisManager(am) {
    AIE::RuntimeSequenceOp runtimeSequenceOp =
        llvm::dyn_cast<AIE::RuntimeSequenceOp>(op);
    if (!runtimeSequenceOp) {
      op->emitError("RuntimeCallGraphCyclicityAnalysis can only be called on "
                    "aiex.runtime_sequence operations.");
      return;
    }

    // Use DFS with a stack to detect cycles
    // A cycle exists if we encounter a sequence already on the current path
    llvm::DenseSet<AIE::RuntimeSequenceOp> callStack;
    llvm::DenseSet<AIE::RuntimeSequenceOp> visited;

    // Module-level symbol cache for calleeSequenceCached below; see
    // populateModuleSymbolCache's comment. Kept lazy (populated inside the
    // lambda, not here): this analysis runs once per runtime_sequence op,
    // most with no RunOps at all, and an eager scan here paid module-wide
    // cost on every one of them regardless (measured regression, 2026-09-12).
    llvm::DenseMap<StringAttr, Operation *> moduleSymbolCache;
    bool moduleSymbolCachePopulated = false;
    auto calleeSequenceCached =
        [&moduleSymbolCache, &moduleSymbolCachePopulated,
         op](RunOp runOp) -> AIE::RuntimeSequenceOp {
      ConfigureOp configureOp =
          runOp.getOperation()->getParentOfType<ConfigureOp>();
      if (!configureOp)
        return nullptr;
      if (!moduleSymbolCachePopulated) {
        if (ModuleOp moduleOp = op->getParentOfType<ModuleOp>()) {
          for (Operation &topOp : moduleOp.getOps()) {
            if (auto symName = topOp.getAttrOfType<StringAttr>(
                    SymbolTable::getSymbolAttrName()))
              moduleSymbolCache[symName] = &topOp;
          }
        }
        moduleSymbolCachePopulated = true;
      }
      auto devIt = moduleSymbolCache.find(
          configureOp.getSymbolAttr().getRootReference());
      if (devIt == moduleSymbolCache.end())
        return nullptr;
      auto calleeDevice = llvm::dyn_cast<AIE::DeviceOp>(devIt->second);
      if (!calleeDevice)
        return nullptr;
      return llvm::dyn_cast_or_null<AIE::RuntimeSequenceOp>(
          SymbolTable::lookupSymbolIn(calleeDevice,
                                      runOp.getRuntimeSequenceSymbol()));
    };

    std::function<bool(AIE::RuntimeSequenceOp)> hasCycle =
        [&](AIE::RuntimeSequenceOp seq) -> bool {
      if (callStack.contains(seq)) {
        return true; // Found a cycle
      }
      if (visited.contains(seq)) {
        return false; // Already checked this sequence
      }

      callStack.insert(seq);
      visited.insert(seq);

      // Check all sequences called by this one
      bool foundCycle = false;
      seq.walk([&](RunOp runOp) {
        if (AIE::RuntimeSequenceOp callee = calleeSequenceCached(runOp)) {
          if (hasCycle(callee)) {
            foundCycle = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });

      callStack.erase(seq);
      return foundCycle;
    };

    if (hasCycle(runtimeSequenceOp)) {
      isCyclic = true;
      isValid = true;
      return;
    }
    isCyclic = false;
    isValid = true;
  }
};

// Turn aie.configure @device into aie.run %.. @configure
// TODO: add check that liveness of two aie.configures do not overlap
// (i.e., when we configure A, then configure B, cannot call runtime sequence of
// A after configuring B)
// TODO: add code to remove repeated @configure ops
struct InsertLoadPdiForConfigurePattern : RewritePattern {

  InsertLoadPdiForConfigurePattern(MLIRContext *context,
                                   PatternBenefit benefit = 1)
      : RewritePattern(ConfigureOp::getOperationName(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    ConfigureOp configureOp = llvm::dyn_cast<ConfigureOp>(op);
    if (!configureOp) {
      return failure();
    }

    // LoadPDI resets the whole device, hence cannot do partial reconfiguration;
    // therefore, this only supports top-level configure ops
    if (!llvm::isa<AIE::RuntimeSequenceOp>(configureOp->getParentOp())) {
      return failure();
    }

    AIE::DeviceOp referencedDevice = configureOp.getReferencedDeviceOp();
    if (!referencedDevice) {
      configureOp.emitError("Referenced symbol is not a device");
      return failure();
    }

    Block *configureBlock;
    if (configureOp.getBody().empty()) {
      configureBlock = rewriter.createBlock(&configureOp.getBody());
    } else {
      configureBlock = &configureOp.getBody().front();
    }

    rewriter.setInsertionPointToStart(configureBlock);
    AIEX::NpuLoadPdiOp::create(
        rewriter, configureOp.getLoc(),
        FlatSymbolRefAttr::get(referencedDevice.getSymNameAttr()),
        /*id=*/nullptr, /*size=*/nullptr, /*address=*/nullptr,
        /*expand_mode=*/configureOp.getExpandModeAttr());

    return success();
  }
};

// Collects all external SSA values referenced by an operation (and its nested
// operations).
// 1. Collects SSA values from the operation's operands.
// 2. Recursively walks through all operations in the operation's regions.
// 3. For each nested operation, collects SSA values from its operands.
// 4. Skips values that are already in argMap or defined within the operation.
// 5. For memref.subview operations, traces to the root block argument
static void
collectReferencedSSAValues(Operation *op, const IRMapping &argMap,
                           llvm::SetVector<Value> &referencedValues) {

  auto processValue = [&](Value operand) {
    if (argMap.contains(operand)) {
      return;
    }

    // If this is a subview, trace to the root block argument
    if (auto traceResult = traceSubviewToBlockArgument(operand)) {
      // Check if the root argument is already mapped
      if (!argMap.contains(traceResult->rootArg)) {
        referencedValues.insert(traceResult->rootArg);
      }
      return;
    }

    // Not a subview chain leading to block arg, add as-is
    referencedValues.insert(operand);
  };

  // Collect SSA values from the operation's direct operands.
  for (Value operand : op->getOperands()) {
    processValue(operand);
  }

  // Recursively collect SSA values from nested operations in all regions.
  for (Region &region : op->getRegions()) {
    region.walk([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        if (argMap.contains(operand)) {
          continue;
        }

        // Check if defined within the parent operation.
        if (Operation *defOp = operand.getDefiningOp()) {
          if (op->isProperAncestor(defOp)) {
            continue;
          }
        } else if (auto blockArg = llvm::dyn_cast<BlockArgument>(operand)) {
          // A block argument has no defining op, so the check above cannot see
          // it. One belonging to a region nested inside `op` -- an scf.for
          // induction variable, most commonly -- is nonetheless defined within
          // `op` and must not be collected as an external reference: it would
          // reach copyReferencedSSAValues, whose getDefiningOp() is null, and
          // fail with "Referenced value is not defined by an operation".
          Operation *owner = blockArg.getOwner()->getParentOp();
          if (owner && (owner == op || op->isProperAncestor(owner))) {
            continue;
          }
        }

        processValue(operand);
      }
    });
  }
}

// Return the operation in the caller device that stands for `op`, cloning `op`
// on first use. `clonedDefs` spans the aiex.run calls of one caller device, so
// several calls that name one definition share one clone and one symbol.
static Operation *
getOrClone(PatternRewriter &rewriter, Operation *op, IRMapping &argMap,
           llvm::DenseMap<Operation *, Operation *> &clonedDefs,
           mlir::OpBuilder::InsertPoint &insertPoint) {
  auto it = clonedDefs.find(op);
  if (it == clonedDefs.end()) {
    rewriter.restoreInsertionPoint(insertPoint);
    it = clonedDefs.try_emplace(op, rewriter.clone(*op, argMap)).first;
    insertPoint = rewriter.saveInsertionPoint();
  }
  argMap.map(op->getResult(0), it->second->getResult(0));
  return it->second;
}

// Copies SSA value definitions into the caller device.
// Currently, only `aie.tile` operations are supported.
// Updates argMap to map old values to new/existing values.
static LogicalResult
copyReferencedSSAValues(PatternRewriter &rewriter,
                        const llvm::SetVector<Value> &referencedValues,
                        AIE::DeviceOp callerDevice, IRMapping &argMap,
                        llvm::DenseMap<Operation *, Operation *> &clonedDefs,
                        mlir::OpBuilder::InsertPoint &clonedSSAInsertPoint,
                        Operation *errorReportOp) {

  llvm::SetVector<Value> referencedValuesToVisit = referencedValues;
  std::vector<Operation *> referencedOpsToClone = {};
  while (!referencedValuesToVisit.empty()) {
    Value referencedValue = referencedValuesToVisit.pop_back_val();
    Operation *definingOp = referencedValue.getDefiningOp();
    if (!definingOp) {
      return errorReportOp->emitError()
             << "Referenced value is not defined by an operation";
    }
    if (llvm::find(referencedOpsToClone, definingOp) !=
        referencedOpsToClone.end()) {
      continue;
    }

    if (auto tileOp = llvm::dyn_cast<AIE::TileOp>(definingOp)) {
      referencedOpsToClone.insert(referencedOpsToClone.begin(), definingOp);
    } else if (auto lockOp = llvm::dyn_cast<AIE::LockOp>(definingOp)) {
      Value lockTile = lockOp.getTile();
      if (lockTile) {
        referencedValuesToVisit.insert(lockTile);
      }
      referencedOpsToClone.push_back(definingOp);
    } else {
      return errorReportOp->emitError()
             << "Referenced SSA value defined by unsupported operation type: "
             << definingOp->getName().getStringRef()
             << ". Currently only aie.tile and aie.lock operations are "
                "supported.";
    }
  }

  for (Operation *definingOp : referencedOpsToClone) {
    if (auto tileOp = llvm::dyn_cast<AIE::TileOp>(definingOp)) {
      int col = tileOp.getCol();
      int row = tileOp.getRow();

      // A tile is its coordinates, so a tile the caller already declares stands
      // for the callee's tile.
      AIE::TileOp existingTile = nullptr;
      for (AIE::TileOp tile : callerDevice.getOps<AIE::TileOp>()) {
        if (tile.getCol() == col && tile.getRow() == row) {
          existingTile = tile;
          break;
        }
      }

      if (existingTile) {
        // Verify that all attributes match
        if (tileOp->getAttrDictionary() != existingTile->getAttrDictionary()) {
          // Filter out result type attributes and symbol attributes for
          // comparison
          auto filterAttrs = [](DictionaryAttr dict) -> DictionaryAttr {
            SmallVector<NamedAttribute> filteredAttrs;
            for (auto namedAttr : dict) {
              StringRef name = namedAttr.getName().getValue();
              if (name != "col" && name != "row") {
                filteredAttrs.push_back(namedAttr);
              }
            }
            return DictionaryAttr::get(dict.getContext(), filteredAttrs);
          };

          DictionaryAttr tileAttrs = filterAttrs(tileOp->getAttrDictionary());
          DictionaryAttr existingAttrs =
              filterAttrs(existingTile->getAttrDictionary());

          if (tileAttrs != existingAttrs) {
            return errorReportOp->emitError()
                   << "aie.tile(" << col << ", " << row
                   << ") already exists in the device with different "
                      "attributes";
          }
        }
        clonedDefs[definingOp] = existingTile.getOperation();
      }
    } else if (!llvm::isa<AIE::LockOp>(definingOp)) {
      return errorReportOp->emitError()
             << "Referenced SSA value defined by unsupported operation type: "
             << definingOp->getName().getStringRef()
             << ". Currently only aie.tile and aie.lock operations are "
                "supported.";
    }

    Operation *clonedOp = getOrClone(rewriter, definingOp, argMap, clonedDefs,
                                     clonedSSAInsertPoint);
    rewriter.replaceOpUsesWithIf(
        definingOp, clonedOp->getResult(0), [&](OpOperand &operand) {
          return operand.getOwner()->getParentOfType<AIE::DeviceOp>() ==
                 callerDevice;
        });
  }

  return success();
}

// Indexes moduleOp's top-level symbols once instead of paying
// SymbolTable::lookupSymbolIn's per-call linear scan of every top-level op
// (device count in practice) on each lookup -- O(devices) per lookup,
// O(devices * inlines) total across a pass that resolves one cross-device
// reference per inlined symbol AND (via RunOp/ConfigureOp's own
// getCalleeDeviceOp/getCalleeRuntimeSequenceOp) one callee-device lookup per
// RunOp match. Profiled 2026-09-12: this callee-device resolution, not the
// cross-device symbol fallback below, is the dominant cost -- fixing the
// latter alone left the pass's growth curve essentially unchanged.
static void
populateModuleSymbolCache(ModuleOp moduleOp,
                          llvm::DenseMap<StringAttr, Operation *> &cache,
                          bool &populated) {
  if (populated)
    return;
  for (Operation &topOp : moduleOp.getOps()) {
    if (auto symName =
            topOp.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      cache[symName] = &topOp;
  }
  populated = true;
}

// Inlines the definitions of all symbols referenced in the given operation
// at the current insertion point in the given rewriter, unless the symbol
// definition is in the "previouslyInlinedSymbolMap" map. While inlining,
// symbols will be renamed to have a unique name.
// Also copies in SSA values referenced by the inlined symbol definitions.
static LogicalResult inlineReferencedSymbolDefinitions(
    PatternRewriter &rewriter, Operation *op, Operation *lookupFrom,
    IRMapping argMap,
    llvm::DenseMap<SymbolRefAttr, SymbolRefAttr> &previouslyInlinedSymbolMap,
    AIE::DeviceOp callerDevice,
    llvm::DenseMap<Operation *, Operation *> &clonedDefs,
    mlir::OpBuilder::InsertPoint &clonedDefOpsInsertionPoint,
    llvm::SetVector<SymbolRefAttr> &allSymbolNames,
    llvm::StringMap<unsigned> &nextSuffixCounter,
    llvm::DenseMap<StringAttr, Operation *> &moduleSymbolCache,
    bool &moduleSymbolCachePopulated) {
  MLIRContext *ctx = op->getContext();
  for (NamedAttribute namedAttr : op->getAttrs()) {
    Attribute attr = namedAttr.getValue();
    auto newAttr = attr.replace([&](SymbolRefAttr oldSymbolRef) {
      SymbolRefAttr newSymbolRef;
      if (!previouslyInlinedSymbolMap.count(oldSymbolRef)) {
        llvm::StringRef oldName = oldSymbolRef.getRootReference().getValue();
        std::string uniqueName = oldName.str();
        if (allSymbolNames.count(SymbolRefAttr::get(ctx, uniqueName))) {
          unsigned &uniquingCounter = nextSuffixCounter[oldName];
          do {
            uniqueName = oldName.str() + "_" + std::to_string(uniquingCounter);
            ++uniquingCounter;
          } while (allSymbolNames.count(SymbolRefAttr::get(ctx, uniqueName)));
        }
        newSymbolRef = SymbolRefAttr::get(ctx, uniqueName);
        allSymbolNames.insert(newSymbolRef);
        previouslyInlinedSymbolMap[oldSymbolRef] = newSymbolRef;

        // Add the new symbol definition
        // First try to look up from the lookupFrom operation (e.g., within the
        // callee device). If not found, try looking up from the module level
        // (for cross-device references).
        Operation *symbolDefOp =
            SymbolTable::lookupNearestSymbolFrom(lookupFrom, oldSymbolRef);
        if (!symbolDefOp && oldSymbolRef.getNestedReferences().empty()) {
          symbolDefOp =
              AIE::lookupNamedOp(lookupFrom, oldSymbolRef.getRootReference());
        }
        if (!symbolDefOp) {
          if (ModuleOp moduleOp = lookupFrom->getParentOfType<ModuleOp>()) {
            if (!oldSymbolRef.getNestedReferences().empty()) {
              // Nested refs are rare here (this pass flattens cross-device
              // refs to single names); fall back to the general resolver
              // rather than teach the cache below to walk nested scopes too.
              symbolDefOp = SymbolTable::lookupSymbolIn(moduleOp, oldSymbolRef);
            } else {
              // See populateModuleSymbolCache's comment: this replaces
              // lookupSymbolIn's per-call linear scan of every top-level
              // device with a one-time index.
              populateModuleSymbolCache(moduleOp, moduleSymbolCache,
                                        moduleSymbolCachePopulated);
              auto cached =
                  moduleSymbolCache.find(oldSymbolRef.getRootReference());
              symbolDefOp = cached != moduleSymbolCache.end() ? cached->second
                                                              : nullptr;
            }
          }
        }
        if (!symbolDefOp) {
          return std::make_pair(newSymbolRef, WalkResult::interrupt());
        }

        // If the symbol is a device, don't clone it - keep the original
        // reference. Device ops must stay at module level.
        if (llvm::isa<AIE::DeviceOp>(symbolDefOp)) {
          return std::make_pair(oldSymbolRef, WalkResult::advance());
        }

        // Collect SSA values referenced by the symbol definition operation
        llvm::SetVector<Value> symbolReferencedValues;
        collectReferencedSSAValues(symbolDefOp, argMap, symbolReferencedValues);

        // Copy SSA values referenced by the symbol definition
        // This updates clonedDefOpsInsertionPoint to be after the copied SSA
        // values
        if (failed(copyReferencedSSAValues(rewriter, symbolReferencedValues,
                                           callerDevice, argMap, clonedDefs,
                                           clonedDefOpsInsertionPoint, op))) {
          return std::make_pair(newSymbolRef, WalkResult::interrupt());
        }

        // Insert the cloned symbol at the device level, after its SSA
        // dependencies
        rewriter.restoreInsertionPoint(clonedDefOpsInsertionPoint);
        Operation *clonedSymbolDefOp = rewriter.clone(*symbolDefOp, argMap);
        clonedSymbolDefOp->setAttr(SymbolTable::getSymbolAttrName(),
                                   StringAttr::get(ctx, uniqueName));
        clonedDefOpsInsertionPoint = rewriter.saveInsertionPoint();
      } else {
        newSymbolRef = previouslyInlinedSymbolMap[oldSymbolRef];
      }

      return std::make_pair(newSymbolRef, WalkResult::advance());
    });
    if (!newAttr) {
      return failure();
    }
    op->setAttr(namedAttr.getName(), newAttr);
  }
  return success();
}

struct InlineRuntimeCallsPattern : RewritePattern {

  mlir::OpBuilder::InsertPoint &ssaDefInsertPoint;
  mlir::OpBuilder::InsertPoint &symbolDefInsertPoint;
  llvm::SetVector<SymbolRefAttr> &allSymbolNames;
  llvm::DenseMap<Operation *, Operation *> &clonedDefs;

  // Per-base-name resume point for the uniquing search below, so the k-th
  // inline of a repeated base name probes once instead of re-walking every
  // suffix already claimed by this pass (same anti-pattern/fix shape as the
  // id-uniquing cache in AIEDecomposeLargeDmaBd.cpp). Safe to resume from
  // rather than restart at 0: allSymbolNames only grows, so a suffix this
  // pattern already claimed stays claimed, and any pre-existing name at an
  // unvisited suffix is still caught by the membership check below.
  mutable llvm::StringMap<unsigned> nextSuffixCounter;

  // Module-level symbol cache; see populateModuleSymbolCache's comment.
  // Scoped to this pattern instance (one per device, matching
  // allSymbolNames' own scope), shared by the cross-device symbol lookup
  // below and by this pattern's own callee-device resolution.
  mutable llvm::DenseMap<StringAttr, Operation *> moduleSymbolCache;
  mutable bool moduleSymbolCachePopulated = false;

  // See traceProgressIfEnabled's comment.
  mutable long tracedInlineCount = 0;
  mutable std::chrono::steady_clock::time_point traceStart;
  mutable bool traceStarted = false;

  InlineRuntimeCallsPattern(
      MLIRContext *ctx, mlir::OpBuilder::InsertPoint &ssaDefInsertPoint,
      mlir::OpBuilder::InsertPoint &symbolDefInsertPoint,
      llvm::SetVector<SymbolRefAttr> &allSymbolNames,
      llvm::DenseMap<Operation *, Operation *> &clonedDefs)
      : RewritePattern(RunOp::getOperationName(), PatternBenefit(1), ctx),
        ssaDefInsertPoint(ssaDefInsertPoint),
        symbolDefInsertPoint(symbolDefInsertPoint),
        allSymbolNames(allSymbolNames), clonedDefs(clonedDefs) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    llvm::DenseMap<SymbolRefAttr, SymbolRefAttr> previouslyInlinedSymbolMap;

    RunOp runOp = llvm::dyn_cast<RunOp>(op);
    if (!runOp) {
      return failure();
    }

    // Resolve the callee device via the module-symbol cache rather than
    // runOp.getCalleeDeviceOp()/getCalleeRuntimeSequenceOp(), which each
    // re-scan every top-level device for the same device_ref on every
    // match -- see populateModuleSymbolCache's comment.
    AIEX::ConfigureOp configureOp =
        runOp.getOperation()->getParentOfType<AIEX::ConfigureOp>();
    ModuleOp moduleOp = runOp.getOperation()->getParentOfType<ModuleOp>();
    if (!configureOp || !moduleOp) {
      return failure();
    }
    populateModuleSymbolCache(moduleOp, moduleSymbolCache,
                              moduleSymbolCachePopulated);
    auto calleeIt = moduleSymbolCache.find(
        configureOp.getSymbolAttr().getRootReference());
    AIE::DeviceOp calleeDevice =
        calleeIt != moduleSymbolCache.end()
            ? llvm::dyn_cast<AIE::DeviceOp>(calleeIt->second)
            : nullptr;
    if (!calleeDevice) {
      // Matches getReferencedDeviceOp()'s own diagnostic for this case.
      configureOp.emitError()
          << "No such device: '" << configureOp.getSymbolAttr() << "'";
      return failure();
    }
    AIE::RuntimeSequenceOp calleeRuntimeSequence =
        llvm::dyn_cast_or_null<AIE::RuntimeSequenceOp>(
            SymbolTable::lookupSymbolIn(calleeDevice,
                                        runOp.getRuntimeSequenceSymbol()));
    if (!calleeRuntimeSequence) {
      return failure();
    }

    // rewrite logic

    // Get caller and callee bodies. The callee body will be inlined into the
    // caller body at the point of the RunOp.
    Region &calleeBody = calleeRuntimeSequence.getBody();
    AIE::DeviceOp callerDevice =
        runOp.getOperation()->getParentOfType<AIE::DeviceOp>();
    if (!callerDevice) {
      runOp.emitError() << "needs to be in a DeviceOp";
      return failure();
    }

    // The argMap maps callee arguments to caller SSA values.
    IRMapping argMap;
    ValueRange values = runOp.getArgs();
    for (unsigned i = 0, n = calleeBody.getNumArguments(); i < n; i++) {
      BlockArgument arg = calleeBody.getArgument(i);
      Value val = values[i];
      argMap.map(arg, val);
    }

    // The callee body may reference SSA values and symbols that are defined
    // in the callee device (outside the callee runtime sequence). We will
    // inline a supported set of these and error otherwise.

    // Collect SSA values referenced in the callee not defined by the callee and
    // not in the argMap.
    llvm::SetVector<Value> referencedValues;
    for (Operation &op : calleeBody.getOps()) {
      collectReferencedSSAValues(&op, argMap, referencedValues);
    }
    llvm::SetVector<Value> filteredValues;
    for (Value val : referencedValues) {
      if (val.getParentRegion() != &calleeBody) {
        filteredValues.insert(val);
      }
    }
    referencedValues = std::move(filteredValues);

    // Copy the operations that define these SSA values into the caller device
    if (failed(copyReferencedSSAValues(rewriter, referencedValues, callerDevice,
                                       argMap, clonedDefs, ssaDefInsertPoint,
                                       runOp))) {
      return failure();
    }

    // Find the calling runtime sequence so we can hoist certain ops to its
    // start instead of placing them at the call site.
    AIE::RuntimeSequenceOp callerRuntimeSequence =
        runOp.getOperation()->getParentOfType<AIE::RuntimeSequenceOp>();
    if (!callerRuntimeSequence) {
      runOp.emitError() << "needs to be (transitively) inside a "
                           "aie.runtime_sequence operation";
      return failure();
    }
    Block &callerBodyBlock = callerRuntimeSequence.getBody().front();
    mlir::Block::iterator hoistPos = callerBodyBlock.begin();

    // Now, also inline symbol definitions referenced in the callee body;
    // this may pull in additional SSA values referenced by the symbol
    // definitions.
    //
    // npu.create_scratchpad ops are hoisted to the start of the calling
    // runtime sequence instead of being placed at the call site, so a single
    // scratchpad is created per sequence regardless of how many callees were
    // inlined.
    rewriter.setInsertionPoint(runOp);
    mlir::OpBuilder::InsertPoint clonedOpInsertionPoint =
        rewriter.saveInsertionPoint();
    for (Operation &op : calleeBody.getOps()) {
      bool shouldHoist = llvm::isa<NpuCreateScratchpadOp>(&op);
      if (shouldHoist) {
        rewriter.setInsertionPoint(&callerBodyBlock, hoistPos);
        rewriter.clone(op, argMap);
        continue;
      }

      rewriter.restoreInsertionPoint(clonedOpInsertionPoint);
      Operation *clonedOp = rewriter.clone(op, argMap);
      clonedOpInsertionPoint = rewriter.saveInsertionPoint();

      // Inline symbol references in all nested ops.
      WalkResult symbolWalk = clonedOp->walk([&](Operation *nestedOp) {
        if (failed(inlineReferencedSymbolDefinitions(
                rewriter, nestedOp, calleeRuntimeSequence.getOperation(),
                argMap, previouslyInlinedSymbolMap, callerDevice, clonedDefs,
                symbolDefInsertPoint, allSymbolNames, nextSuffixCounter,
                moduleSymbolCache, moduleSymbolCachePopulated))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (symbolWalk.wasInterrupted()) {
        return failure();
      }
    }

    // The aiex.run op has been inlined; erase it.
    rewriter.eraseOp(runOp);

    traceProgressIfEnabled("aie-materialize: RunOps inlined", tracedInlineCount,
                           traceStart, traceStarted);
    return success();
  }
};

/// Validate all aiex.run ops inside a aiex.configure op against the referenced
/// device. This must be called sequentially (before any parallel pass
/// execution) because it performs cross-DeviceOp symbol table lookups.
/// Placing this validation in RunOp::verify() is unsafe: MLIR's pass manager
/// may invoke op verifiers concurrently on sibling DeviceOps, causing a data
/// race on the referenced device's symbol table.
static LogicalResult verifyRunOpsInConfigureOp(ConfigureOp configureOp,
                                               AIE::DeviceOp referencedDev) {
  if (configureOp.getBody().empty())
    return success();
  for (RunOp runOp : configureOp.getBody().front().getOps<RunOp>()) {
    auto seqName = runOp.getRuntimeSequenceSymbol();
    Operation *maybeSeq = SymbolTable::lookupSymbolIn(referencedDev, seqName);
    if (!maybeSeq) {
      auto err = runOp.emitError()
                 << "No such runtime sequence for device '"
                 << referencedDev.getSymName() << "': '" << seqName << "'";
      err.attachNote(referencedDev.getLoc())
          << "This device does not have a '" << seqName << "' runtime sequence";
      return failure();
    }
    auto runtimeSeq = llvm::dyn_cast<AIE::RuntimeSequenceOp>(maybeSeq);
    if (!runtimeSeq) {
      return runOp.emitError()
             << "'" << seqName << "' is not a runtime sequence";
    }

    // Validate argument count and types against the callee signature.
    // An empty body region (no blocks) means the sequence takes no arguments.
    Region &calleeRegion = runtimeSeq.getBody();
    unsigned numCalleeArgs =
        calleeRegion.empty() ? 0 : calleeRegion.getNumArguments();
    ValueRange args = runOp.getArgs();
    if (numCalleeArgs != args.size())
      return runOp.emitOpError() << "argument count mismatch: expected "
                                 << numCalleeArgs << " but got " << args.size();
    for (unsigned i = 0; i < numCalleeArgs; i++) {
      Type expected = calleeRegion.front().getArgument(i).getType();
      Type actual = args[i].getType();
      if (expected != actual)
        return runOp.emitOpError()
               << "argument " << i << " type mismatch: expected " << expected
               << " but got " << actual;
    }
  }
  return success();
}

struct AIEMaterializeRuntimeSequencesPass
    : xilinx::AIEX::impl::AIEMaterializeRuntimeSequencesBase<
          AIEMaterializeRuntimeSequencesPass> {

  // After inlining, a runtime sequence may contain multiple
  // npu.create_scratchpad ops (one from the sequence itself and one from each
  // inlined callee).  Keep only the first; all must agree on size because the
  // scratchpad size is a module-level property.  Returns failure if a size
  // mismatch is detected.
  LogicalResult
  deduplicateCreateScratchpadOps(AIE::RuntimeSequenceOp runtimeSeqOp) {
    if (runtimeSeqOp.getBody().empty())
      return success();

    Block &body = runtimeSeqOp.getBody().front();
    NpuCreateScratchpadOp firstScratchpad;
    SmallVector<NpuCreateScratchpadOp> duplicates;
    for (NpuCreateScratchpadOp scratchpadOp :
         body.getOps<NpuCreateScratchpadOp>()) {
      if (!firstScratchpad) {
        firstScratchpad = scratchpadOp;
        continue;
      }
      if (firstScratchpad.getSize() != scratchpadOp.getSize()) {
        scratchpadOp.emitError(
            "create_scratchpad size mismatch after inlining: ")
            << scratchpadOp.getSize() << " != " << firstScratchpad.getSize();
        return failure();
      }
      duplicates.push_back(scratchpadOp);
    }
    for (NpuCreateScratchpadOp dup : duplicates)
      dup.erase();

    return success();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Same cache as InlineRuntimeCallsPattern's (see populateModuleSymbolCache):
    // this pre-materialization verification loop resolves one callee device
    // per ConfigureOp via getReferencedDeviceOp(), which re-scans every
    // top-level device per call -- O(devices * configures) otherwise.
    llvm::DenseMap<StringAttr, Operation *> verifyModuleSymbolCache;
    bool verifyModuleSymbolCachePopulated = false;

    // Process each device in the module
    for (AIE::DeviceOp deviceOp : moduleOp.getOps<AIE::DeviceOp>()) {

      // Verify all runtime sequences before materialization
      for (AIE::RuntimeSequenceOp runtimeSequenceOp :
           deviceOp.getOps<AIE::RuntimeSequenceOp>()) {
        if (failed(runtimeSequenceOp.verifyBeforeMaterialization())) {
          return signalPassFailure();
        }

        // Validate aiex.run ops inside aiex.configure ops. This cross-device
        // check is performed here (sequentially) rather than in RunOp::verify()
        // to avoid a race condition: MLIR's pass manager runs verifiers on
        // sibling DeviceOps concurrently, and looking up symbols in a sibling
        // DeviceOp from a verifier causes a data race on its symbol table.
        for (ConfigureOp configureOp :
             runtimeSequenceOp.getOps<ConfigureOp>()) {
          populateModuleSymbolCache(moduleOp, verifyModuleSymbolCache,
                                    verifyModuleSymbolCachePopulated);
          auto devIt = verifyModuleSymbolCache.find(
              configureOp.getSymbolAttr().getRootReference());
          AIE::DeviceOp referencedDev =
              devIt != verifyModuleSymbolCache.end()
                  ? llvm::dyn_cast<AIE::DeviceOp>(devIt->second)
                  : nullptr;
          if (!referencedDev) {
            // ConfigureOp::verify() already reported the error (no such
            // device, not a device, or device type mismatch) at parse time;
            // this is a safety net if verification was disabled.
            return signalPassFailure();
          }
          if (failed(verifyRunOpsInConfigureOp(configureOp, referencedDev)))
            return signalPassFailure();
        }
      }

      // Check for cycles in runtime sequence calls
      for (AIE::RuntimeSequenceOp runtimeSequenceOp :
           deviceOp.getOps<AIE::RuntimeSequenceOp>()) {
        AnalysisManager am =
            getAnalysisManager().nest(deviceOp).nest(runtimeSequenceOp);
        RuntimeCallGraphCyclicityAnalysis cyclicity =
            am.getAnalysis<RuntimeCallGraphCyclicityAnalysis>();
        if (!cyclicity.isValid) {
          return signalPassFailure();
        }
        if (cyclicity.isCyclic) {
          runtimeSequenceOp.emitError(
              "Runtime sequence call graph contains a cycle");
          return signalPassFailure();
        }
      }

      // Greedily inline all runtime sequences that can be inlined;
      // this will start with runtime sequences that do not call other runtime
      // sequences (leaves); once their callers inline them, the callers can
      // be inlined as well, and so on
      mlir::Block &deviceBodyFirstBlock = deviceOp.getBodyRegion().front();
      auto runtimeSequenceOps = deviceOp.getOps<AIE::RuntimeSequenceOp>();
      if (runtimeSequenceOps.begin() == runtimeSequenceOps.end()) {
        // No runtime sequences to materialize
        continue;
      }
      AIE::RuntimeSequenceOp firstRuntimeSequenceOp =
          *runtimeSequenceOps.begin();
      mlir::OpBuilder::InsertPoint ssaDefInsertPoint(
          &deviceBodyFirstBlock, deviceBodyFirstBlock.begin());
      mlir::OpBuilder::InsertPoint symbolDefInsertPoint(
          &deviceBodyFirstBlock, mlir::Block::iterator(firstRuntimeSequenceOp));
      llvm::SetVector<SymbolRefAttr> allSymbolNames = {};
      for (Operation &op : deviceBodyFirstBlock) {
        if (auto symbolName = op.getAttrOfType<StringAttr>(
                SymbolTable::getSymbolAttrName())) {
          allSymbolNames.insert(SymbolRefAttr::get(symbolName));
        }
      }

      MLIRContext *ctx = &getContext();
      GreedyRewriteConfig rewriter_config = GreedyRewriteConfig();
      rewriter_config.setRegionSimplificationLevel(
          GreedySimplifyRegionLevel::Disabled);

      RewritePatternSet patterns_0(ctx);
      llvm::DenseMap<Operation *, Operation *> clonedDefs;
      patterns_0.insert<InlineRuntimeCallsPattern>(ctx, ssaDefInsertPoint,
                                                   symbolDefInsertPoint,
                                                   allSymbolNames, clonedDefs);
      if (failed(applyPatternsGreedily(deviceOp, std::move(patterns_0),
                                       rewriter_config))) {
        return signalPassFailure();
      }

      // Deduplicate create_scratchpad ops that were hoisted during inlining.
      for (AIE::RuntimeSequenceOp runtimeSeqOp :
           deviceOp.getOps<AIE::RuntimeSequenceOp>()) {
        if (failed(deduplicateCreateScratchpadOps(runtimeSeqOp)))
          return signalPassFailure();
      }

      // Insert LoadPDI ops for each aiex.configure op
      RewritePatternSet patterns_1(ctx);
      patterns_1.insert<InsertLoadPdiForConfigurePattern>(ctx);
      walkAndApplyPatterns(deviceOp, std::move(patterns_1));

      // Canonicalize to remove duplicate back-to-back load_pdi ops
      RewritePatternSet canonicalize_patterns(ctx);
      AIEX::NpuLoadPdiOp::getCanonicalizationPatterns(canonicalize_patterns,
                                                      ctx);
      if (failed(applyPatternsGreedily(
              deviceOp, std::move(canonicalize_patterns), rewriter_config))) {
        return signalPassFailure();
      }

      // Flatten the IR: hoist all operations inside aiex.configure to be direct
      // children of the runtime sequence, preserving order
      for (AIE::RuntimeSequenceOp runtimeSequenceOp :
           deviceOp.getOps<AIE::RuntimeSequenceOp>()) {
        SmallVector<ConfigureOp> configureOps;

        for (ConfigureOp configureOp :
             runtimeSequenceOp.getOps<ConfigureOp>()) {
          configureOps.push_back(configureOp);
        }

        IRRewriter rewriter(ctx);
        for (ConfigureOp configureOp : configureOps) {
          Block &configureBlock = configureOp.getBody().front();

          // Collect all operations in the configure block
          SmallVector<Operation *> opsToHoist;
          for (Operation &op : configureBlock) {
            opsToHoist.push_back(&op);
          }

          // Hoist operations to be right before the configure op
          rewriter.setInsertionPoint(configureOp);
          for (Operation *op : opsToHoist) {
            op->moveBefore(configureOp);
          }

          // Erase the now-empty configure op
          rewriter.eraseOp(configureOp);
        }
      }

    } // end for each device
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
AIEX::createAIEMaterializeRuntimeSequencesPass() {
  return std::make_unique<AIEMaterializeRuntimeSequencesPass>();
}
