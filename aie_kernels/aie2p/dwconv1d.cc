//===- dwconv1d.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Depthwise conv1d, 'same' padding, stride 1, bf16 -- one channel (row) per
// call:
//
//   out[t] = bias + sum_{p=0..K-1} w[p] * in_pad[t + p],   t = 0 .. T-1
//
// This is a cross-correlation (no kernel flip), matching torch.nn.Conv1d and
// most framework "same" depthwise convs. `in_pad` is the ALREADY-PADDED row
// the caller supplies -- this kernel does no boundary handling and needs no
// local scratch buffer:
//
//   in_pad[0 .. T+15]: [P zeros | T real samples | P zeros | 16-(K-1) junk]
//                       \_______________ T+K-1 valid, halo-padded ________/
//   P = (K - 1) / 2
//
// The trailing "don't-care" slack past the halo (up to a fixed 16 elements,
// independent of K) exists only so the vectorized kernel's aligned 16-wide
// loads never read past the end of the buffer; its VALUES never reach the
// output (see the tap loop below). Callers can zero it like the rest of the
// pad. See dwconv1d.py's `_pad_input` for a reference construction.
//
// Vectorized 16-outputs/iteration: two aligned loads build a 32-lane window
// (`in_pad` must be aligned; T a multiple of 16), and aie::shuffle_down_fill
// slides a 16-wide tap window across it in-register -- no unaligned loads.
// Each tap's product is aie::mac'd into one accfloat accumulator per output
// chunk (the K-sum never leaves the accumulator).
//
// K is compile-time (DWCONV_K, default 9); the shuffle window this uses is
// 16 + K - 1 wide and must fit the 32-lane two-register concat, so K <= 17.
//
// *** Why not aie::sliding_mul_ops ***
// sliding_mul is the obvious primitive for exactly this shape (a K-tap
// window MAC with the K-sum kept in one accumulator), and an earlier version
// of this kernel used aie::sliding_mul_ops<L, K, 1, 1, 1, bfloat16,
// bfloat16>::mul(). On aie2p it produces wrong output (inf/nan) for the
// bf16 x bf16 specialization even with aligned inputs and even at the
// trivial K=1 case -- a genuine aie_api defect, not a misuse of the op
// (isolated with a minimal single-core repro independent of this kernel's
// dataflow, K=1/4/9 all fail, both the pinned toolchain and the latest
// nightly at the time). This kernel routes around it with aligned-load +
// shuffle_down_fill + mac instead: same brick-layer intent (one accumulator,
// L outputs/issue, no scalar per-tap loop), a path that is bit-correct here.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

template <int K, bool BIAS>
static inline void dwconv1d_same_bf16_impl(const bfloat16 *restrict in_pad,
                                           const bfloat16 *restrict w,
                                           bfloat16 *restrict out, int32_t T) {
  static_assert(
      K >= 1 && K <= 17,
      "K taps must fit one 32-lane shuffle window (16 + K - 1 <= 32)");
  event0();
  ::aie::set_rounding(::aie::rounding_mode::conv_even);

  const float bias = BIAS ? static_cast<float>(w[K]) : 0.0f;
  const ::aie::vector<float, 16> bias_v = ::aie::broadcast<float, 16>(bias);

  // One broadcast per tap, hoisted out of the T-loop.
  ::aie::vector<bfloat16, 16> coeff[K];
  for (int p = 0; p < K; p++)
    coeff[p] = ::aie::broadcast<bfloat16, 16>(w[p]);

  for (int32_t o = 0; o < T; o += 16) {
    const ::aie::vector<bfloat16, 16> a0 = ::aie::load_v<16>(in_pad + o);
    const ::aie::vector<bfloat16, 16> a1 = ::aie::load_v<16>(in_pad + o + 16);
    ::aie::accum<accfloat, 16> acc;
    acc.from_vector(bias_v);
    for (int p = 0; p < K; p++) {
      // window(p) = in_pad[o+p .. o+p+15], built in-register from a0/a1 --
      // never an unaligned load. Only fill lanes [0, p) of a1 are selected
      // here (p <= K-1 <= 16), so the don't-care tail described above never
      // reaches the accumulator.
      const ::aie::vector<bfloat16, 16> window =
          ::aie::shuffle_down_fill(a0, a1, static_cast<unsigned>(p));
      acc = ::aie::mac(acc, window, coeff[p]);
    }
    ::aie::store_v(out + o, acc.template to_vector<bfloat16>());
  }
  event1();
}

#ifndef DWCONV_K
#define DWCONV_K 9
#endif
#ifndef DWCONV_BIAS
#define DWCONV_BIAS 1
#endif

extern "C" {

// Depthwise conv1d, 'same' padding, one channel per call. K = DWCONV_K taps
// (default 9), bias = DWCONV_BIAS (default on). See the file header for the
// in_pad layout and the -DDWCONV_K=/-DDWCONV_BIAS= compile-time selection
// (mirrors the -DINT8_ACT-style flag convention the sibling conv kernels in
// this directory use).
//   in_pad: T + 16 bf16 (halo + don't-care tail, see above)
//   w:      DWCONV_K + DWCONV_BIAS bf16 (taps [0..K-1], bias at [K] if BIAS).
//           This kernel itself only ever reads DWCONV_K + DWCONV_BIAS
//           elements of w; an IRON caller (e.g. dwconv1d.py) may pass a wider
//           buffer than that for its own reasons (aie.dma_bd needs a 4-byte-
//           aligned transfer length, which a DWCONV_BIAS=0 row of odd
//           DWCONV_K would not be); any padding past index
//           DWCONV_K + DWCONV_BIAS - 1 is never touched here.
//   out:    T bf16
//   T:      output length, must be a multiple of 16
void dwconv1d_bf16(bfloat16 *in_pad, bfloat16 *w, bfloat16 *out, int32_t T) {
  dwconv1d_same_bf16_impl<DWCONV_K, (bool)DWCONV_BIAS>(in_pad, w, out, T);
}

} // extern "C"
