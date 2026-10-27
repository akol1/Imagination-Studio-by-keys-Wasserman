// ours: YuE2-3B's autoregressive decode step on CUDA (families/yue2): a Qwen3-shaped body (GQA, per-head q / k RMS
// norms, rotate-half rope) one token a branch, on music3's warp-a-row GEMVs (bf16 / int8 a scale a row / a group).
// The rounding points are the ones PyTorch's bf16 reference takes (modeling_yue2.py): RMS norm as x * rs.to(bf16) *
// w, rope cos / sin cast to bf16 before the products, each product and sum rounded to bf16.
#include "music3_common.cuh"
#include "music3_ar.cuh"

#define Y2_HD 128

// x[b] = table[tok[b]] (bf16 rows of dim) for b < blockIdx range.
extern "C" __global__ void y2_embed(const long long* __restrict__ tok, const bf16* __restrict__ table, bf16* __restrict__ x, int dim) {
  const int b = blockIdx.x;
  const uint4* src = reinterpret_cast<const uint4*>(table + (size_t)tok[b] * dim);
  uint4* dst = reinterpret_cast<uint4*>(x + (size_t)b * dim);
  for (int i = threadIdx.x; i < dim / 8; i += blockDim.x) dst[i] = src[i];
}

// y = round(round(x * round(rsqrt(mean(x^2) + eps))) * w), one row a block (PyTorch's RMSNorm in bf16).
extern "C" __global__ void y2_rmsnorm(const bf16* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ y, int dim, float eps) {
  __shared__ float sh[32];
  const bf16* xr = x + (size_t)blockIdx.x * dim;
  float ss = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = b2f(xr[i]);
    ss = fmaf(v, v, ss);
  }
  ss = block_sum(ss, sh);
  const float rs = rb(rsqrtf(ss / (float)dim + eps));
  bf16* yr = y + (size_t)blockIdx.x * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = f2b(rb(b2f(xr[i]) * rs) * b2f(w[i]));
}

// q / k: a head's RMS norm then rotate-half rope at the branch's position; v copied. qkv rows [B, (H + 2 KV) 128];
// q to q_out [B, H, 128], k / v to the caches [B, cap, KV, 128] at slot pos[b]. grid (B, H + 2 KV), block 128.
extern "C" __global__ void __launch_bounds__(Y2_HD) y2_qk_rope(const bf16* __restrict__ qkv, int heads, int kv, const bf16* __restrict__ qn, const bf16* __restrict__ kn, const float* __restrict__ inv_freq, const long long* __restrict__ pos, bf16* __restrict__ q_out, bf16* __restrict__ kc, bf16* __restrict__ vc, int cap, float eps) {
  __shared__ float sh[32];
  __shared__ float v_sh[Y2_HD];
  const int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const long long p = pos[b];
  const bf16* row = qkv + (size_t)b * (heads + 2 * kv) * Y2_HD;
  if (h >= heads + kv) {
    const int g = h - heads - kv;
    vc[(((size_t)b * cap + p) * kv + g) * Y2_HD + d] = row[(size_t)(heads + kv + g) * Y2_HD + d];
    return;
  }
  const bool isq = h < heads;
  const float x = b2f(row[(size_t)h * Y2_HD + d]);
  const float ss = block_sum(x * x, sh);
  const float rs = rb(rsqrtf(ss / (float)Y2_HD + eps));
  const float nx = rb(rb(x * rs) * b2f((isq ? qn : kn)[d]));
  v_sh[d] = nx;
  __syncthreads();
  const int i = d & (Y2_HD / 2 - 1);
  const float ang = (float)p * inv_freq[i];
  const float c = rb(cosf(ang)), s = rb(sinf(ang));
  const float o = d < Y2_HD / 2 ? rb(rb(nx * c) - rb(v_sh[d + Y2_HD / 2] * s)) : rb(rb(nx * c) + rb(v_sh[d - Y2_HD / 2] * s));
  if (isq)
    q_out[((size_t)b * heads + h) * Y2_HD + d] = f2b(o);
  else
    kc[(((size_t)b * cap + p) * kv + (h - heads)) * Y2_HD + d] = f2b(o);
}

// Split-key decode attention over slots 0..pos[b]: a block a (kv head, branch, split), a warp a q head of the group,
// four keys in flight a warp. Partials ml [B, H, S, 2] (max, sum) and acc [B, H, S, 128] (fp32).
extern "C" __global__ void y2_attn_part(const bf16* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc, const long long* __restrict__ pos, int heads, int kv, int cap, int S, float scale, float* __restrict__ ml, float* __restrict__ acc) {
  const int g = blockIdx.x, b = blockIdx.y, sp = blockIdx.z;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int h = g * (heads / kv) + w;
  const int L = (int)pos[b] + 1;
  const int chunk = (L + S - 1) / S;
  const int k0 = sp * chunk, k1 = min(L, k0 + chunk);
  const uint2 qv = *reinterpret_cast<const uint2*>(q + ((size_t)b * heads + h) * Y2_HD + lane * 4);
  const float q0 = bfu(qv.x & 0xFFFF) * scale, q1 = bfu(qv.x >> 16) * scale, q2 = bfu(qv.y & 0xFFFF) * scale, q3 = bfu(qv.y >> 16) * scale;
  const size_t stride = (size_t)kv * Y2_HD;  // one slot to the next
  const bf16* kb = kc + ((size_t)b * cap * kv + g) * Y2_HD + lane * 4;
  const bf16* vb = vc + ((size_t)b * cap * kv + g) * Y2_HD + lane * 4;
  float m = -INFINITY, l = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  int j = k0;
  for (; j + 4 <= k1; j += 4) {
    uint2 kk[4], vv[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      kk[u] = __ldg(reinterpret_cast<const uint2*>(kb + (size_t)(j + u) * stride));
      vv[u] = __ldg(reinterpret_cast<const uint2*>(vb + (size_t)(j + u) * stride));
    }
    float s[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) s[u] = q0 * bfu(kk[u].x & 0xFFFF) + q1 * bfu(kk[u].x >> 16) + q2 * bfu(kk[u].y & 0xFFFF) + q3 * bfu(kk[u].y >> 16);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int u = 0; u < 4; ++u) s[u] += __shfl_xor_sync(FULL, s[u], o);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float mn = fmaxf(m, s[u]);
      const float cor = __expf(m - mn), e = __expf(s[u] - mn);
      l = l * cor + e;
      a0 = a0 * cor + e * bfu(vv[u].x & 0xFFFF);
      a1 = a1 * cor + e * bfu(vv[u].x >> 16);
      a2 = a2 * cor + e * bfu(vv[u].y & 0xFFFF);
      a3 = a3 * cor + e * bfu(vv[u].y >> 16);
      m = mn;
    }
  }
  for (; j < k1; ++j) {
    const uint2 kv2 = __ldg(reinterpret_cast<const uint2*>(kb + (size_t)j * stride));
    float s = q0 * bfu(kv2.x & 0xFFFF) + q1 * bfu(kv2.x >> 16) + q2 * bfu(kv2.y & 0xFFFF) + q3 * bfu(kv2.y >> 16);
    s = warp_sum(s);
    const uint2 vv = __ldg(reinterpret_cast<const uint2*>(vb + (size_t)j * stride));
    const float mn = fmaxf(m, s);
    const float cor = __expf(m - mn), e = __expf(s - mn);
    l = l * cor + e;
    a0 = a0 * cor + e * bfu(vv.x & 0xFFFF);
    a1 = a1 * cor + e * bfu(vv.x >> 16);
    a2 = a2 * cor + e * bfu(vv.y & 0xFFFF);
    a3 = a3 * cor + e * bfu(vv.y >> 16);
    m = mn;
  }
  const size_t o = ((size_t)b * heads + h) * S + sp;
  if (lane == 0) {
    ml[o * 2] = m;
    ml[o * 2 + 1] = l;
  }
  *reinterpret_cast<float4*>(acc + o * Y2_HD + lane * 4) = make_float4(a0, a1, a2, a3);
}

// The S partials of a (branch, head) into out [B, H * 128] bf16. grid (B, H), block 128.
extern "C" __global__ void y2_attn_merge(const float* __restrict__ ml, const float* __restrict__ acc, int heads, int S, bf16* __restrict__ out) {
  const int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const size_t base = ((size_t)b * heads + h) * S;
  float m = -INFINITY;
  for (int s = 0; s < S; ++s) m = fmaxf(m, ml[(base + s) * 2]);
  float l = 0.f, a = 0.f;
  for (int s = 0; s < S; ++s) {
    const float ms = ml[(base + s) * 2];
    if (ms == -INFINITY) continue;
    const float c = __expf(ms - m);
    l += ml[(base + s) * 2 + 1] * c;
    a += acc[(base + s) * Y2_HD + d] * c;
  }
  out[((size_t)b * heads + h) * Y2_HD + d] = f2b(a / l);
}

// pos[b] += 1 for b < B: the step's token is in the cache.
extern "C" __global__ void y2_advance(long long* __restrict__ pos, int B) {
  if (threadIdx.x < B) pos[threadIdx.x] += 1;
}

// The decode GEMVs for one branch (M = 1) and a CFG pair (M = 2): music3's gemv template, bf16 / int8 a scale a row /
// int8 a scale a group of 128 inputs, plain or gate | up SwiGLU.
#define Y2_GEMV(NAME, M, I8, PAIR, GRP)                                                                                \
  extern "C" __global__ void __launch_bounds__(256) NAME(const void* __restrict__ w, const float* __restrict__ sc, const bf16* __restrict__ x, int xs, int K, int N, bf16* y, int ys, float* __restrict__ yf, const bf16* res, int res_div) { \
    gemv<M, I8, PAIR, GRP>(w, sc, x, xs, K, N, y, ys, yf, res, res_div);                                                     \
  }
Y2_GEMV(y2_gemv_bf16_1, 1, false, false, false)
Y2_GEMV(y2_gemv_bf16_2, 2, false, false, false)
Y2_GEMV(y2_gemv_i8_1, 1, true, false, false)
Y2_GEMV(y2_gemv_i8_2, 2, true, false, false)
Y2_GEMV(y2_gemv_g8_1, 1, true, false, true)
Y2_GEMV(y2_gemv_g8_2, 2, true, false, true)
Y2_GEMV(y2_glu_bf16_1, 1, false, true, false)
Y2_GEMV(y2_glu_bf16_2, 2, false, true, false)
Y2_GEMV(y2_glu_i8_1, 1, true, true, false)
Y2_GEMV(y2_glu_i8_2, 2, true, true, false)
Y2_GEMV(y2_glu_g8_1, 1, true, true, true)
Y2_GEMV(y2_glu_g8_2, 2, true, true, true)
