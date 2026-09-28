//===- AIEXAttrs.td ----------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_ATTRS
#define AIEX_ATTRS

include "mlir/IR/EnumAttr.td"

def StateTableFuncMul  : I32EnumAttrCase<"Mul", 0, "mul">;
def StateTableFuncIncr : I32EnumAttrCase<"Incr", 1, "incr">;
def StateTableFuncDecr : I32EnumAttrCase<"Decr", 2, "decr">;

def StateTableFunc : I32EnumAttr<"StateTableFunc",
    "Function to apply between scratchpad value and register pair",
    [StateTableFuncMul, StateTableFuncIncr, StateTableFuncDecr]> {
  let cppNamespace = "::xilinx::AIEX";
}

def ScratchpadParameterKindCore : I32EnumAttrCase<"Core", 0, "core">;
def ScratchpadParameterKindAddr : I32EnumAttrCase<"Addr", 1, "addr">;

def ScratchpadParameterKind : I32EnumAttr<"ScratchpadParameterKind",
    "Usage kind of a scratchpad runtime parameter: "
    "`core` parameters are read by AIE cores via "
    "`aiex.read_scratchpad_parameter` (values are shift-2 encoded); "
    "`addr` parameters are used as BD address offsets via the "
    "`offset_parameter` attribute on DMA ops (values written raw)",
    [ScratchpadParameterKindCore, ScratchpadParameterKindAddr]> {
  let cppNamespace = "::xilinx::AIEX";
}

// DMA_S2MM_*_Ctrl.FoT_Mode: what an S2MM channel does when it sees TLAST.
// Values and encoding match aie_registers_aie2.json's FoT_Mode bit field
// (bits 17:16 of the channel's Ctrl register).
def FoTModeDisabled : I32EnumAttrCase<"Disabled", 0, "disabled">;
def FoTModeNoCounts : I32EnumAttrCase<"NoCounts", 1, "no_counts">;
def FoTModeCountsWithTaskTokens
    : I32EnumAttrCase<"CountsWithTaskTokens", 2, "counts_with_task_tokens">;
def FoTModeCountsFromMMRegister
    : I32EnumAttrCase<"CountsFromMMRegister", 3, "counts_from_mm_register">;

def FoTMode : I32EnumAttr<"FoTMode", "DMA S2MM finish-on-TLAST mode",
    [FoTModeDisabled, FoTModeNoCounts, FoTModeCountsWithTaskTokens,
     FoTModeCountsFromMMRegister]> {
  let cppNamespace = "::xilinx::AIEX";
}

// How a `load_pdi` op should be expanded by the `aie-expand-load-pdi` pass.
def AIEX_ExpandNone : I32EnumAttrCase<"none", 0>;
def AIEX_ExpandWrite32 : I32EnumAttrCase<"write32", 1>;
def AIEX_ExpandCtrlPkt : I32EnumAttrCase<"ctrlpkt", 2>;

def AIEX_ExpandMode
    : I32EnumAttr<"ExpandMode", "How to expand a load_pdi op",
                  [AIEX_ExpandNone, AIEX_ExpandWrite32, AIEX_ExpandCtrlPkt]> {
  let cppNamespace = "::xilinx::AIEX";
}

#endif // AIEX_ATTRS
