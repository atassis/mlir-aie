// REAL 2-stage attention CONVEYOR kernels (heavy matmul stage -> light softmax stage).
//   stage_scores : ac = scale * Q.K^T   [TQ,DK]x[T,DK] bf16 -> [TQ,T] f32   (stage A, heavy)
//   stage_softmax: probs = row_softmax(ac)     [TQ,T] f32 -> [TQ,T] bf16     (stage B, light)
// One query tile, plain (non-relpos) attention -- the heavy->light conveyor crux from the
// research, on real math. Dims baked (like -DRELPOS_T). exp2 uses the device-proven poly
// helper from relpos_mha.cc (hw exp2 is ~2-4% off on aie2p; NOINLINE avoids the -O2 NaN bug).
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef ATTN_TQ
#define ATTN_TQ 8
#endif
#ifndef ATTN_T
#define ATTN_T 64
#endif
#ifndef ATTN_DK
#define ATTN_DK 64
#endif
#ifndef ATTN_SCALE
#define ATTN_SCALE 0.125f // 1/sqrt(64)
#endif

static constexpr float LOG2E = 1.4426950408889634f;
static constexpr int VL = 16;

// SOFTWARE f32 2^x (x<=0), device-proven (probe_floor rel-err 8.5e-5). NOINLINE is load-bearing:
// inlining into the softmax loop makes Peano -O2 miscompile to NaN. Copied from relpos_mha.cc.
static __attribute__((noinline)) aie::vector<float, VL> exp2f_vec(aie::vector<float, VL> x) {
  x = aie::max(x, aie::broadcast<float, VL>(-100.0f));
  aie::vector<int32_t, VL> ki = aie::to_fixed<int32_t>(x);
  aie::vector<float, VL> kf = aie::to_float<float>(ki);
  aie::vector<int32_t, VL> one = aie::broadcast<int32_t, VL>(1);
  aie::vector<int32_t, VL> zero = aie::broadcast<int32_t, VL>(0);
  ki = aie::sub(ki, aie::select(zero, one, aie::lt(x, kf)));
  aie::vector<float, VL> f = aie::sub(x, aie::to_float<float>(ki));
  aie::vector<float, VL> p = aie::broadcast<float, VL>(0.0013333558f);
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(0.0096181291f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(0.0555041087f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(0.2402265069f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(0.6931471805f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(1.0f));
  aie::vector<int32_t, VL> ebits =
      aie::upshift(aie::add(ki, aie::broadcast<int32_t, VL>(127)), 23);
  aie::vector<float, VL> p2k = ebits.cast_to<float>();
  return aie::mul(p, p2k).to_vector<float>();
}

// STAGE A -- scores. ac[i,j] = scale * dot(q[i,:], k[j,:]). bf16 in, f32 accumulate.
extern "C" void stage_scores(const bfloat16 *__restrict q,
                             const bfloat16 *__restrict k, float *__restrict ac) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, DK = ATTN_DK;
  constexpr float scale = ATTN_SCALE;
  event0();
  for (int i = 0; i < TQ; i++) {
    const bfloat16 *qr = q + i * DK;
    for (int j = 0; j < T; j++) {
      const bfloat16 *kr = k + j * DK;
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL)
        acc = aie::mac(acc, aie::load_v<VL>(qr + d), aie::load_v<VL>(kr + d));
      ac[i * T + j] = aie::reduce_add(acc.to_vector<float>()) * scale;
    }
  }
  event1();
}

// MONOLITH baseline: all 3 stages on ONE tile, per query tile. 2 inputs (q + kv, k|V packed) to fit
// the 2-input-channel budget. Local ac/probs scratch (stack). For the conveyor-vs-monolith perf A/B.
extern "C" void stage_mono(const bfloat16 *__restrict q, const bfloat16 *__restrict kv,
                           bfloat16 *__restrict ctx) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, DK = ATTN_DK;
  constexpr float scale = ATTN_SCALE;
  const bfloat16 *k = kv;
  const bfloat16 *V = kv + T * DK;
  // static L1 scratch (safe in the MONO single worker -- no concurrent ping-pong belt to alias, unlike
  // the pipelined softmax). Keeps ~3.3 KB off the small AIE stack (stack-alloc here -> nan/overflow).
  static float ac[TQ * T];
  static bfloat16 probs[TQ * T];
  static float srow[ATTN_T];
  aie::vector<float, VL> log2e_v = aie::broadcast<float, VL>(LOG2E);
  event0();
  // scores
  for (int i = 0; i < TQ; i++)
    for (int j = 0; j < T; j++) {
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL)
        acc = aie::mac(acc, aie::load_v<VL>(q + i * DK + d), aie::load_v<VL>(k + j * DK + d));
      ac[i * T + j] = aie::reduce_add(acc.to_vector<float>()) * scale;
    }
  // softmax
  for (int i = 0; i < TQ; i++) {
    const float *ar = ac + i * T;
    bfloat16 *pr = probs + i * T;
    float rowmax = -3.0e38f;
    for (int j = 0; j < T; j += VL) {
      aie::vector<float, VL> a = aie::load_v<VL>(ar + j);
      aie::store_v(srow + j, a);
      float cm = aie::reduce_max(a);
      if (cm > rowmax) rowmax = cm;
    }
    aie::vector<float, VL> maxv = aie::broadcast<float, VL>(rowmax);
    for (int j = 0; j < T; j += VL)
      aie::store_v(srow + j, exp2f_vec(aie::mul(aie::sub(aie::load_v<VL>(srow + j), maxv), log2e_v).to_vector<float>()));
    aie::accum<accfloat, VL> sa = aie::zeros<accfloat, VL>();
    for (int j = 0; j < T; j += VL) sa = aie::add(sa, aie::load_v<VL>(srow + j));
    float inv = 1.0f / aie::reduce_add(sa.to_vector<float>());
    aie::vector<float, VL> iv = aie::broadcast<float, VL>(inv);
    for (int j = 0; j < T; j += VL)
      aie::store_v(pr + j, aie::mul(aie::load_v<VL>(srow + j), iv).to_vector<bfloat16>());
  }
  // ctx
  for (int i = 0; i < TQ; i++)
    for (int d = 0; d < DK; d += VL) {
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int j = 0; j < T; j++)
        acc = aie::mac(acc, aie::broadcast<bfloat16, VL>(probs[i * T + j]), aie::load_v<VL>(V + j * DK + d));
      aie::store_v(ctx + i * DK + d, acc.to_vector<bfloat16>());
    }
  event1();
}

// TRIVIAL copy variants (same signatures/sizes as the real stages) -- to isolate STRUCTURE vs KERNELS
// in the N_QT>1 streaming bug. T==DK==64 so TQ*T == TQ*DK == 512; each just casts-copies straight
// through, so ctx == q. If this PASSES at N_QT>1, the real (slow) kernels are the culprit (race).
extern "C" void stage_scores_t(const bfloat16 *__restrict q, const bfloat16 *__restrict k,
                               float *__restrict ac) {
  event0();
  for (int i = 0; i < ATTN_TQ * ATTN_T; i++) ac[i] = (float)q[i];
  event1();
}
extern "C" void stage_softmax_t(const float *__restrict ac, bfloat16 *__restrict probs) {
  event0();
  for (int i = 0; i < ATTN_TQ * ATTN_T; i++) probs[i] = (bfloat16)ac[i];
  event1();
}
extern "C" void stage_ctx_t(const bfloat16 *__restrict probs, const bfloat16 *__restrict V,
                            bfloat16 *__restrict ctx) {
  event0();
  for (int i = 0; i < ATTN_TQ * ATTN_DK; i++) ctx[i] = probs[i];
  event1();
}

// STAGE C -- context. ctx[i,d] = sum_j probs[i,j] * V[j,d]. probs[TQ,T] bf16, V[T,DK] bf16 ->
// ctx[TQ,DK] bf16. Two inputs (probs from the belt + V from DDR) on ONE tile -- valid (stage A
// already takes q+k). f32 accumulate, single bf16 narrow at the end.
extern "C" void stage_ctx(const bfloat16 *__restrict probs,
                          const bfloat16 *__restrict V, bfloat16 *__restrict ctx) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, DK = ATTN_DK;
  event0();
  for (int i = 0; i < TQ; i++) {
    const bfloat16 *pr = probs + i * T;
    bfloat16 *cr = ctx + i * DK;
    for (int d = 0; d < DK; d += VL) {
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int j = 0; j < T; j++)
        acc = aie::mac(acc, aie::broadcast<bfloat16, VL>(pr[j]), aie::load_v<VL>(V + j * DK + d));
      aie::store_v(cr + d, acc.to_vector<bfloat16>());
    }
  }
  event1();
}

// STAGE B -- row softmax. probs[i,:] = softmax(ac[i,:]). f32 in, bf16 out. 3-pass (max, exp+sum,
// normalize), exp/sum SPLIT into separate loops (fusing exp2f_vec + accfloat sum -> NaN on aie2p).
extern "C" void stage_softmax(const float *__restrict ac, bfloat16 *__restrict probs) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T;
  float srow[ATTN_T];   // was static -- static may alias the odd ping-pong belt buffer in L1
  aie::vector<float, VL> log2e_v = aie::broadcast<float, VL>(LOG2E);
  event0();
  for (int i = 0; i < TQ; i++) {
    const float *ar = ac + i * T;
    bfloat16 *pr = probs + i * T;
    // pass 1: row max
    float rowmax = -3.0e38f;
    for (int j = 0; j < T; j += VL) {
      aie::vector<float, VL> a = aie::load_v<VL>(ar + j);
      aie::store_v(srow + j, a);
      float cm = aie::reduce_max(a);
      if (cm > rowmax) rowmax = cm;
    }
    // pass 2a: exp2((s-max)*log2e) into srow
    aie::vector<float, VL> maxv = aie::broadcast<float, VL>(rowmax);
    for (int j = 0; j < T; j += VL) {
      aie::vector<float, VL> sl =
          aie::mul(aie::sub(aie::load_v<VL>(srow + j), maxv), log2e_v).to_vector<float>();
      aie::store_v(srow + j, exp2f_vec(sl));
    }
    // pass 2b: sum
    aie::accum<accfloat, VL> sumacc = aie::zeros<accfloat, VL>();
    for (int j = 0; j < T; j += VL)
      sumacc = aie::add(sumacc, aie::load_v<VL>(srow + j));
    float inv_sum = 1.0f / aie::reduce_add(sumacc.to_vector<float>());
    // pass 3: normalize -> bf16
    aie::vector<float, VL> inv_v = aie::broadcast<float, VL>(inv_sum);
    for (int j = 0; j < T; j += VL) {
      aie::vector<float, VL> e = aie::load_v<VL>(srow + j);
      aie::store_v(pr + j, aie::mul(e, inv_v).to_vector<bfloat16>());
    }
  }
  event1();
}
