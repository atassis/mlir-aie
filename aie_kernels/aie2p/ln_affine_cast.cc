//===- ln_affine_cast.cc ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

// Fused row-wise LayerNorm + affine + f32 -> bfloat16 narrowing cast, one
// kernel / one dispatch.
//
//   per row of `cols`, with inv = 1 / sqrt(var + eps):
//     mean = Sx / cols
//     var  = S(x - mean)^2 / cols        (centered two-pass, f32)
//     out  = ((x - mean) * inv) * gamma + beta -> bfloat16
//
// gamma/beta are the same for every row of a call, packed into one
// `[2 * cols]` parameter buffer `gb = [gamma(0..cols) | beta(cols..2*cols)]`
// on a single DMA input channel, alongside the `[cols]` `x` input -- two
// inputs total, the AIE2p compute-tile input-DMA limit.
//
// The variance is computed as the centered two-pass sum, not
// `E[x^2] - mean^2`: the latter catastrophically cancels for rows whose mean
// is close to their RMS, which is the common case for near-zero-mean
// activations.
//
// Rounding: the two reduction passes (sum, sum-of-squares) run in the AIE
// core's default rounding mode. Only the final affine + narrowing write
// explicitly sets `aie::rounding_mode::conv_even` (round-to-nearest-even).
// The AIE2p core's rounding mode is a single sticky register shared by every
// kernel that runs on that core, so a kernel that cares about its own
// narrowing behavior has to set the mode itself rather than assume what a
// previously-run kernel left it at. conv_even also matches a typical host
// f32 -> bf16 pack (e.g. AVX512-BF16's vcvtne2ps2bf16), so an on-chip and a
// host cast of the same value agree bit-for-bit.
template <int N>
void ln_affine_cast_row(const float *restrict input, const float *restrict gb,
                        bfloat16 *restrict output, int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;
  const int chunks = cols / N;
  const float *gamma = gb;
  const float *beta = gb + cols;

  // pass 1: mean = Sx / cols
  ::aie::vector<float, N> sum_v = ::aie::zeros<float, N>();
  for (int i = 0; i < chunks; i++)
    sum_v = ::aie::add(sum_v, ::aie::load_v<N>(input + i * N));
  float mean = ::aie::reduce_add(sum_v) / float(cols);
  ::aie::vector<float, N> mean_v = ::aie::broadcast<float, N>(mean);

  // pass 2: var = S(x - mean)^2 / cols
  ::aie::vector<float, N> var_v = ::aie::zeros<float, N>();
  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> d =
        ::aie::sub(::aie::load_v<N>(input + i * N), mean_v);
    ::aie::vector<float, N> sq = ::aie::mul(d, d);
    var_v = ::aie::add(var_v, sq);
  }
  float var = ::aie::reduce_add(var_v) / float(cols);
  float inv_std = ::aie::invsqrt(var + epsilon);
  ::aie::vector<float, N> inv_v = ::aie::broadcast<float, N>(inv_std);

  // write: out = ((x - mean) * inv) * gamma + beta -> bf16
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> d =
        ::aie::sub(::aie::load_v<N>(input + i * N), mean_v);
    ::aie::vector<float, N> norm = ::aie::mul(d, inv_v);
    ::aie::vector<float, N> g = ::aie::load_v<N>(gamma + i * N);
    ::aie::vector<float, N> b = ::aie::load_v<N>(beta + i * N);
    ::aie::vector<float, N> ng = ::aie::mul(norm, g);
    ::aie::vector<float, N> y = ::aie::add(ng, b);
    ::aie::accum<accfloat, N> a;
    a.from_vector(y);
    ::aie::store_v(output + i * N, a.template to_vector<bfloat16>());
  }
  event1();
}

extern "C" {
void ln_affine_cast_row(float *input, float *gb, bfloat16 *output,
                        int32_t cols) {
  ln_affine_cast_row<16>(input, gb, output, cols);
}
}
