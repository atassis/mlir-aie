//===- ProfileTrace.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// --profile-trace: a Chrome Trace Event JSON (complete "X" events) with one
// span per graph edge (ExecutionEngine.h) and, inside every PassPipeline
// edge, one span per pass execution per anchor op (via PassInstrumentation
// below). Disabled by default; `enabled` gates every call so the instruments
// cost nothing when the flag is absent.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_PROFILETRACE_H
#define AIECC_PROFILETRACE_H

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace xilinx::aiecc {

inline uint64_t nowEpochUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// Small stable integer per calling thread, for the trace's "tid" field.
inline int64_t traceThreadId() {
  static std::mutex m;
  static std::unordered_map<std::thread::id, int64_t> ids;
  std::lock_guard<std::mutex> lock(m);
  auto it = ids.find(std::this_thread::get_id());
  if (it != ids.end())
    return it->second;
  int64_t id = (int64_t)ids.size();
  ids[std::this_thread::get_id()] = id;
  return id;
}

// Process-wide sink for trace spans. Collected under a mutex (span counts are
// small relative to edge/pass work) and written once at the end.
class ProfileTrace {
public:
  static ProfileTrace &instance() {
    static ProfileTrace t;
    return t;
  }

  bool enabled = false;
  std::string path;
  uint64_t startEpochUs = 0;

  void addSpan(llvm::StringRef name, int64_t pid, int64_t tid,
              uint64_t tsUs, int64_t durUs,
              std::vector<std::pair<std::string, std::string>> args = {}) {
    if (!enabled)
      return;
    llvm::json::Object obj{{"name", name.str()},
                           {"ph", "X"},
                           {"pid", pid},
                           {"tid", tid},
                           {"ts", (int64_t)tsUs},
                           {"dur", durUs}};
    if (!args.empty()) {
      llvm::json::Object a;
      for (auto &kv : args)
        a[kv.first] = kv.second;
      obj["args"] = std::move(a);
    }
    std::lock_guard<std::mutex> lock(mtx);
    events.push_back(std::move(obj));
  }

  void write() {
    if (!enabled || path.empty())
      return;
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec) {
      llvm::errs() << "aiecc: could not write --profile-trace file '" << path
                   << "': " << ec.message() << "\n";
      return;
    }
    llvm::json::Array traceEvents;
    traceEvents.reserve(events.size());
    for (llvm::json::Object &e : events)
      traceEvents.push_back(llvm::json::Value(std::move(e)));
    llvm::json::Object root{{"traceEvents", std::move(traceEvents)},
                            {"displayTimeUnit", "ms"},
                            {"otherData",
                             llvm::json::Object{
                                 {"processStartEpochUs",
                                  (int64_t)startEpochUs}}}};
    os << llvm::json::Value(std::move(root));
  }

private:
  std::mutex mtx;
  std::vector<llvm::json::Object> events;
};

// Best-effort anchor identity for a pass's operation: its `sym_name`, or for
// aie.tile / aie.core the (col,row) of the tile (aie.core's tile is looked up
// through its `tile` operand). Generic on Operation* so this header does not
// depend on the AIE dialect.
inline std::string anchorLabel(mlir::Operation *op) {
  if (auto sym = op->getAttrOfType<mlir::StringAttr>("sym_name"))
    return sym.getValue().str();
  mlir::Operation *tile = op;
  if (op->getName().getStringRef() == "aie.core" && op->getNumOperands() > 0)
    if (mlir::Operation *def = op->getOperand(0).getDefiningOp())
      tile = def;
  if (tile->getName().getStringRef() == "aie.tile") {
    auto col = tile->getAttrOfType<mlir::IntegerAttr>("col");
    auto row = tile->getAttrOfType<mlir::IntegerAttr>("row");
    if (col && row)
      return "(" + std::to_string(col.getInt()) + "," +
             std::to_string(row.getInt()) + ")";
  }
  return "";
}

// mlir::PassInstrumentation that emits one span per runBeforePass/
// runAfterPass pair. Nested/parallel passes on different anchor ops are
// distinguished by keying the in-flight start time on (thread, pass, op):
// MLIR serializes instrumentation calls under its own lock, but the passes
// themselves run concurrently across threads, so the trace's "tid" still
// shows them on separate lanes.
class TracingPassInstrumentation : public mlir::PassInstrumentation {
public:
  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override {
    std::lock_guard<std::mutex> lock(mtx);
    starts[{pass, op}] = nowEpochUs();
  }

  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override {
    uint64_t start;
    {
      std::lock_guard<std::mutex> lock(mtx);
      auto it = starts.find({pass, op});
      if (it == starts.end())
        return;
      start = it->second;
      starts.erase(it);
    }
    uint64_t end = nowEpochUs();
    std::vector<std::pair<std::string, std::string>> args = {
        {"pass", pass->getArgument().str()},
        {"anchor_op", op->getName().getStringRef().str()}};
    if (std::string label = anchorLabel(op); !label.empty())
      args.push_back({"anchor", label});
    ProfileTrace::instance().addSpan(pass->getArgument(), getpid(),
                                     traceThreadId(), start,
                                     (int64_t)(end - start), std::move(args));
  }

private:
  std::mutex mtx;
  std::map<std::pair<mlir::Pass *, mlir::Operation *>, uint64_t> starts;
};

} // namespace xilinx::aiecc

#endif // AIECC_PROFILETRACE_H
