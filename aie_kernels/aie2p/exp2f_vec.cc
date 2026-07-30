//===- exp2f_vec.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Software f32 2^x, for callers (softmax, sigmoid, and other exp-family ops
// in bf16/f32 pipelines) that need better accuracy than the hardware
// aie::exp2<bfloat16> LUT provides.
//
// aie::exp2<bfloat16> (used by bf16_exp.cc and programming_examples/basic/
// vector_exp) is a bf16-OUTPUT lookup table. Measured on aie2p against a
// float64 2**x reference (same call shape and input on both paths, so the
// comparison isolates the LUT's own error), its max relative error is
// domain-dependent and grows sharply as the input goes negative:
//
//   domain        max rel-err (hw LUT)   max rel-err (this poly)
//   [-1, 0]         6.1%                   8.49e-5
//   [0, 10]         6.1%                   8.49e-5
//   [-10, 0]       10.1%                   8.49e-5
//   [-100, 0]      49.1%                   8.51e-5
//
// The [-100, 0] domain is where softmax lives (scores are shifted by the row
// max before exponentiating, so every input is <= 0 and can be very
// negative for non-max keys), which is also where the LUT is worst.
//
// x = k + f, k = floor(x), f in [0, 1); 2^x = 2^k * poly5(f), poly5
// evaluated by Horner's method; 2^k is reconstructed by placing k + 127
// directly in the f32 exponent field (no multiply/divide needed).
//
// x is clamped to >= -100 before the exponent reconstruction: 2^k relies on
// the biased exponent (k + 127) staying non-negative, which breaks for
// k < -127. -100 is comfortably inside that margin, and 2^-100 ~ 8e-31 is
// already ~0 for any softmax weight, so the clamp is lossless in practice.
// Valid for x <= 0 (softmax's actual domain); not characterized for
// positive x, since that was never this kernel's use case.
//
// exp2f_vec is __attribute__((noinline)). This is LOAD-BEARING, not a style
// choice: inlining it into a register-pressure-heavy caller (a fused
// multi-pass softmax loop) makes Peano -O2 miscompile it to NaN on this
// target. Confirmed with a 3-way A/B on such a caller: "noinline" and "no
// attribute at all" both pass (rel-L2 5.37e-3 against a 0.08 gate,
// correlation 0.999991), while forcing always_inline reproduces the
// corruption (rel-L2 3.35, correlation 0.224). The trap for anyone
// re-testing this: simply deleting the attribute is NOT a valid test of
// whether the bug is still present, because Peano's own -O2 cost model
// already declines to inline this function at that call site, so "attribute
// removed" and "noinline" produce identical passing output for a reason
// that has nothing to do with whether the underlying codegen bug is fixed.
// Only forcing inlining (always_inline) exercises the bug.
//
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// float-domain vector width (512-bit register / 32-bit lanes).
static constexpr int EXP2F_VEC_LEN = 16;

static __attribute__((noinline)) aie::vector<float, EXP2F_VEC_LEN>
exp2f_vec(aie::vector<float, EXP2F_VEC_LEN> x) {
  x = aie::max(x, aie::broadcast<float, EXP2F_VEC_LEN>(-100.0f));
  aie::vector<int32_t, EXP2F_VEC_LEN> ki =
      aie::to_fixed<int32_t>(x); // round-to-nearest on aie2p
  aie::vector<float, EXP2F_VEC_LEN> kf = aie::to_float<float>(ki);
  // floor: where x < kf, k -= 1 (correct even though to_fixed rounds, not
  // truncates)
  aie::vector<int32_t, EXP2F_VEC_LEN> one =
      aie::broadcast<int32_t, EXP2F_VEC_LEN>(1);
  aie::vector<int32_t, EXP2F_VEC_LEN> zero =
      aie::broadcast<int32_t, EXP2F_VEC_LEN>(0);
  ki = aie::sub(ki, aie::select(zero, one, aie::lt(x, kf)));
  aie::vector<float, EXP2F_VEC_LEN> f =
      aie::sub(x, aie::to_float<float>(ki)); // f in [0,1)
  aie::vector<float, EXP2F_VEC_LEN> p =
      aie::broadcast<float, EXP2F_VEC_LEN>(0.0013333558f);
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(0.0096181291f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(0.0555041087f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(0.2402265069f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(0.6931471805f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(1.0f));
  aie::vector<int32_t, EXP2F_VEC_LEN> ebits = aie::upshift(
      aie::add(ki, aie::broadcast<int32_t, EXP2F_VEC_LEN>(127)), 23);
  aie::vector<float, EXP2F_VEC_LEN> p2k = ebits.cast_to<float>();
  return aie::mul(p, p2k).to_vector<float>();
}

extern "C" {

// Buffer entry point: input/output are f32, vector_size must be a multiple
// of EXP2F_VEC_LEN (16).
void exp2f_vec_f32(float *restrict input, float *restrict output,
                   int32_t vector_size) {
  event0();

  auto it_in = aie::cbegin_vector<EXP2F_VEC_LEN>((float *)input);
  auto it_out = aie::begin_vector<EXP2F_VEC_LEN>((float *)output);
  const int elem_iters = vector_size / EXP2F_VEC_LEN;

  for (int i = 0; i < elem_iters; i++) {
    *it_out++ = exp2f_vec(*it_in++);
  }

  event1();
}

} // extern "C"
