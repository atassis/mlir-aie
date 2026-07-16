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
#ifndef ATTN_P
#define ATTN_P (2 * ATTN_T - 1) // relative-position length (NeMo/Parakeet rel-pos)
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

// STAGE A (RELPOS) -- fused scores with on-chip AC + BD + rel_shift + scale (Parakeet/NeMo rel-pos).
//   q       : [TQ, DK] bf16          one query tile from the belt
//   kp      : [(T+P)*DK] bf16 RESIDENT -- k[T,DK] then p[P,DK] packed (2-input budget: q + kp)
//   scores  : [TQ, T] f32 (out)      = (AC + rel_shift(BD)) * inv_scale, feeds the softmax stage
//   row_off : global row index of this query tile (qt*TQ); the rel_shift base uses the GLOBAL row i.
// AC[li,j]=q[li].k[j]; BD[li,jp]=q[li].p[jp]; scores[li,j]=(AC+BD[(T-1-i)+j])*inv_scale, i=row_off+li.
// rel_shift = strided read bd + (T-1-i) (relpos_mha.cc brick 2). BD held per-row in a stack scratch ->
// stage A needs a bumped stack_size at real P (bd[343] f32 = ~1.4 KB). Scalar AC/BD dots (correctness
// first; the vectorized-unaligned path is the follow-up optimization).
extern "C" void stage_scores_relpos(const bfloat16 *__restrict q,
                                    const bfloat16 *__restrict kp,
                                    float *__restrict scores, int32_t row_off) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, DK = ATTN_DK, P = ATTN_P;
  constexpr float inv_scale = ATTN_SCALE;
  const bfloat16 *k = kp;
  const bfloat16 *p = kp + T * DK;
  float bd[ATTN_P];   // per-row BD = q[li] . p^T ; stack scratch (stage A worker gets a bumped stack_size)
  event0();
  for (int li = 0; li < TQ; li++) {
    const bfloat16 *qr = q + li * DK;
    const int i = row_off + li;                 // GLOBAL row index -> rel_shift base
    for (int jp = 0; jp < P; jp++) {
      const bfloat16 *pr = p + jp * DK;
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL)
        acc = aie::mac(acc, aie::load_v<VL>(qr + d), aie::load_v<VL>(pr + d));
      bd[jp] = aie::reduce_add(acc.to_vector<float>());
    }
    const int base = (i < T) ? (T - 1 - i) : 0;  // rel_shift base (clamp padding rows i>=T -> no OOB)
    const float *bd_row = bd + base;
    float *sc = scores + li * T;
    for (int j = 0; j < T; j++) {
      const bfloat16 *kr = k + j * DK;
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL)
        acc = aie::mac(acc, aie::load_v<VL>(qr + d), aie::load_v<VL>(kr + d));
      const float ac = aie::reduce_add(acc.to_vector<float>());
      sc[j] = (ac + bd_row[j]) * inv_scale;
    }
  }
  event1();
}

// STAGE A (RELPOS, REAL-DIMS) -- host-precomputed rel_shifted BD packed in the query belt.
// On-chip BD needs p[P,DK] resident = 88 KB at real dims (blows L1), so instead the host computes
// BD = q.p^T THEN rel_shift -> BD_shifted[TQ,T] bf16, packed AFTER q in ONE belt object. Stage A stays
// at 2 inputs (qbd belt + k resident, 44 KB) and holds NO p / NO bd scratch. rel_shift being host-side
// means there is NO row_off (global-row) dependence -> N_QT>1 needs no tile-offset wiring.
//   qbd    : [TQ*DK + TQ*T] bf16   q[TQ,DK] then BD_shifted[TQ,T]
//   k      : [T*DK] bf16 resident
//   scores : [TQ,T] f32 = (q.k^T + BD_shifted) * inv_scale
// BD carriage in the belt tail: hi-only (plain bf16, BD_SPLIT=0, byte-identical to the host BD-in-belt
// conveyor) or split-bf16 hi+lo (BD_SPLIT=1) reconstructing ~f32 for the on-chip-BD 4th-stage precision.
#ifndef BD_SPLIT
#define BD_SPLIT 0
#endif
extern "C" void stage_scores_relpos_bd(const bfloat16 *__restrict qbd,
                                       const bfloat16 *__restrict k, float *__restrict scores) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, DK = ATTN_DK;
  constexpr float inv_scale = ATTN_SCALE;
  const bfloat16 *q = qbd;
  const bfloat16 *bdhi = qbd + TQ * DK;                 // BD_hi[TQ,T] packed after q
  const bfloat16 *bdlo = bdhi + (BD_SPLIT ? TQ * T : 0); // BD_lo[TQ,T] only when split
  event0();
  for (int li = 0; li < TQ; li++) {
    const bfloat16 *qr = q + li * DK;
    const bfloat16 *hir = bdhi + li * T;
    const bfloat16 *lor = bdlo + li * T;
    float *sc = scores + li * T;
    for (int j = 0; j < T; j++) {
      const bfloat16 *kr = k + j * DK;
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL)
        acc = aie::mac(acc, aie::load_v<VL>(qr + d), aie::load_v<VL>(kr + d));
      const float ac = aie::reduce_add(acc.to_vector<float>());
      float bd = (float)hir[j];
#if BD_SPLIT
      bd += (float)lor[j];                              // reconstruct ~f32 (hi + lo residual)
#endif
      sc[j] = (ac + bd) * inv_scale;
    }
  }
  event1();
}

// Zero-scalar-arg bake wrapper (IRON kernels avoid scalar args -> bake constants). N_QT=1 validation:
// row_off = 0 (the single query tile is rows [0,TQ)). N_QT>1 needs an advancing row_off (tile-offset
// wiring, a follow-up) -- do NOT use this bake for N_QT>1.
extern "C" void stage_scores_relpos_bake(const bfloat16 *__restrict q,
                                         const bfloat16 *__restrict kp, float *__restrict scores) {
  stage_scores_relpos(q, kp, scores, 0);
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

// ==================== STAGE BD -- 4th conveyor stage (on-chip BD) ====================
// On-chip BD = rel_shift((q+pos_bias_v) @ p^T), carried to the scores stage as SPLIT-BF16 in the belt
// tail. Compute is bit-equivalent to relpos_mha.cc (bf16*bf16 -> f32 accfloat, g_bd f32); the split is a
// TRANSPORT device to cross the bf16 belt within the scores tile's 2-input budget. p=[P,DK] ~88 KB is
// L2-resident (MemTile) + streamed in BD_KB-row blocks. Spec: bd-onchip-4th-stage. Placed by the
// generator's --relpos-bd-onchip path; INERT in the current 3-stage build (no caller). g_bd is `static`
// resident scratch -- valid for the mono-style H=1 gate (single BD worker, like stage_mono's statics);
// the pipelined/streaming generator must allocate it as a resident Buffer to avoid the belt-alias hazard.
#ifndef BD_KB
#define BD_KB 39   // p key-block rows (P=2T-1=351 = 9*39, no ragged tail at real dims)
#endif

alignas(32) static float g_bd[ATTN_TQ * ATTN_P]; // resident per-query-tile f32 score scratch (~11 KB)

// COLUMN-SLICE dot: g_bd[il, j0+jj] = dot(qv[il,:], pblk[jj,:]). bf16*bf16 -> f32 accfloat (= relpos_mha.cc).
static inline void bd_dot_block(const bfloat16 *__restrict qv, const bfloat16 *__restrict pblk, int pb, int j0) {
  constexpr int TQ = ATTN_TQ, DK = ATTN_DK, P = ATTN_P;
  for (int il = 0; il < TQ; il++) {
    const bfloat16 *qr = qv + il * DK;
    float *o = g_bd + il * P + j0;
    for (int jj = 0; jj < pb; jj++) {
      const bfloat16 *pr = pblk + jj * DK;
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL)
        acc = aie::mac(acc, aie::load_v<VL>(qr + d), aie::load_v<VL>(pr + d));
      o[jj] = aie::reduce_add(acc.to_vector<float>());
    }
  }
}

// rel_shift + f32->split-bf16 emit. The window base (T-1)-(q0+il) is ~never VL-aligned; a vectorized
// aie::load_v there truncates to the 128b boundary (silent garbage, data-masked -- see the unaligned-load
// kb note). We emit SCALAR: `win[j]` is byte-addressed so it is inherently correct at any offset, and the
// emit (TQ*T ops) is trivial vs the dot (TQ*P*DK MACs). Avoids the alignment hazard by construction.
static inline void bd_relshift_emit(int q0, bfloat16 *__restrict bd_hi, bfloat16 *__restrict bd_lo) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, P = ATTN_P;
  for (int il = 0; il < TQ; il++) {
    const int base = (T - 1) - (q0 + il);           // >= 0 for real rows (q0+il < T)
    const float *win = g_bd + il * P + base;
    bfloat16 *hr = bd_hi + il * T;
    for (int j = 0; j < T; j++) {
      float x = win[j];
      bfloat16 hi = (bfloat16)x;
      hr[j] = hi;
#if BD_SPLIT
      bd_lo[il * T + j] = (bfloat16)(x - (float)hi); // lo residual ~= 8 extra mantissa bits
#endif
    }
  }
}

// STREAMING bricks for the generator Worker loop (int32 scalar ABI). g_bd is the resident scratch.
extern "C" void bd_stream_block(const bfloat16 *__restrict qv, const bfloat16 *__restrict pblk,
                                int32_t pb, int32_t j0) {
  event0(); bd_dot_block(qv, pblk, (int)pb, (int)j0); event1();
}
extern "C" void bd_stream_emit(const bfloat16 *__restrict q_pass, bfloat16 *__restrict out, int32_t q0) {
  constexpr int TQ = ATTN_TQ, DK = ATTN_DK, T = ATTN_T;
  event0();
  for (int i = 0; i < TQ * DK; i++) out[i] = q_pass[i];
  bfloat16 *bd_hi = out + TQ * DK;
  bfloat16 *bd_lo = bd_hi + (BD_SPLIT ? TQ * T : 0);
  bd_relshift_emit((int)q0, bd_hi, bd_lo);
  event1();
}

// MONOLITH bake (H=1 arithmetic gate, p kpv-resident full): dot p in-block, then rel_shift+split emit.
extern "C" void stage_bd(const bfloat16 *__restrict qv, const bfloat16 *__restrict q_pass,
                         const bfloat16 *__restrict p_resident, bfloat16 *__restrict out, int32_t q0) {
  constexpr int P = ATTN_P, DK = ATTN_DK;
  event0();
  for (int j0 = 0; j0 < P; j0 += BD_KB) {
    int pb = (P - j0 < BD_KB) ? (P - j0) : BD_KB;
    bd_dot_block(qv, p_resident + j0 * DK, pb, j0);
  }
  bd_stream_emit(q_pass, out, q0);
  event1();
}

// Zero-scalar-arg BD bake. Belt input packs q_pass[TQ,DK] then qv[TQ,DK] (2*TQ*DK, ONE streamed object);
// p is the 2nd input (resident). Out = q_pass || BD_hi [|| BD_lo] -- the belt stage_scores_relpos_bd
// consumes. ADVANCING q0: the BD worker calls this once per query tile, in order, on one core -> a static
// counter advances q0 = tile_idx*TQ; wrap % N_QT resets it per dispatch (N_QT tiles each). Solves the
// row-offset without a scalar arg / belt header. ATTN_NQT MUST be baked (-DATTN_NQT) to match the build.
#ifndef ATTN_NQT
#define ATTN_NQT 1
#endif
extern "C" void stage_bd_bake(const bfloat16 *__restrict qpv, const bfloat16 *__restrict p,
                              bfloat16 *__restrict out) {
  static int tile_idx = 0;
  const int q0 = tile_idx * ATTN_TQ;
  tile_idx = (tile_idx + 1) % ATTN_NQT;   // wrap per dispatch
  const bfloat16 *q_pass = qpv;
  const bfloat16 *qv = qpv + ATTN_TQ * ATTN_DK;
  stage_bd(qv, q_pass, p, out, q0);
}
