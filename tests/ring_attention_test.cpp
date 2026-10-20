#include "strata/ring/attention.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int fails = 0;
void check(bool ok, const char *text) {
  std::printf("  %-60s %s\n", text, ok ? "ok" : "FAIL");
  if (!ok)
    ++fails;
}
bool near(float a, float b, float tol = 2e-5f) {
  return std::abs(a - b) <= tol * (1.0f + std::max(std::abs(a), std::abs(b)));
}
bool near_abs(float a, float b, float tol = 2e-6f) {
  return std::abs(a - b) <= tol;
}
uint32_t bits(float value) {
  uint32_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
bool near_ulp(float a, float b, uint32_t max_ulps = 4) {
  if (!(a >= 0.0f) || !(b >= 0.0f))
    return false;
  const uint32_t ia = bits(a), ib = bits(b);
  const uint32_t distance = ia >= ib ? ia - ib : ib - ia;
  return distance <= max_ulps;
}
void reference(const strata::ring::AttentionConfig &c, const float *q,
               const float *k, const float *v, int64_t nt, int64_t pos,
               const float *qw, const float *kw,
               const strata::ring::KvCache &old, float *out) {
  const int64_t group = c.n_head / c.n_head_kv;
  const float scale = 1.0f / std::sqrt(float(c.head_dim));
  const size_t kvn = size_t(nt) * c.n_head_kv * c.head_dim;
  std::vector<float> sk(kvn);
  auto norm = [&](const float *x, float *y, const float *w) {
    float ss = 0;
    for (int64_t d = 0; d < c.head_dim; ++d)
      ss += x[d] * x[d];
    const float inv = 1.0f / std::sqrt(ss / float(c.head_dim) + c.rms_eps);
    for (int64_t d = 0; d < c.head_dim; ++d)
      y[d] = x[d] * inv * (w ? w[d] : 1.0f);
  };
  auto rot = [&](const float *x, float *y, int64_t p) {
    std::copy(x, x + c.head_dim, y);
    const int64_t half = c.rope_dim / 2;
    for (int64_t d = 0; d < half; ++d) {
      const float exponent = 2.0f * float(d) / float(c.rope_dim);
      const float z = 1.0f / std::pow(c.rope_theta, exponent);
      const float co = std::cos(float(p) * z), si = std::sin(float(p) * z);
      const float a = x[d], b = x[d + half];
      y[d] = a * co - b * si;
      y[d + half] = b * co + a * si;
    }
  };
  std::vector<float> tmp(c.head_dim), rr(c.head_dim);
  for (int64_t t = 0; t < nt; ++t)
    for (int64_t h = 0; h < c.n_head_kv; ++h) {
      norm(k + (size_t(t) * c.n_head_kv + h) * c.head_dim, tmp.data(), kw);
      rot(tmp.data(), sk.data() + (size_t(t) * c.n_head_kv + h) * c.head_dim,
          pos + t);
    }
  for (int64_t t = 0; t < nt; ++t)
    for (int64_t h = 0; h < c.n_head; ++h) {
      norm(q + (size_t(t) * c.n_head + h) * c.head_dim, tmp.data(), qw);
      rot(tmp.data(), rr.data(), pos + t);
      const int64_t total = pos + t + 1, kh = h / group;
      std::vector<float> scores(total);
      float mx = -INFINITY;
      for (int64_t j = 0; j < total; ++j) {
        const float *kk =
            j < pos
                ? old.key_data() + (size_t(j) * c.n_head_kv + kh) * c.head_dim
                : sk.data() + (size_t(j - pos) * c.n_head_kv + kh) * c.head_dim;
        float dot = 0;
        for (int64_t d = 0; d < c.head_dim; ++d)
          dot += rr[d] * kk[d];
        scores[j] = dot * scale;
        mx = std::max(mx, scores[j]);
      }
      float den = 0;
      for (float &z : scores) {
        z = std::exp(z - mx);
        den += z;
      }
      float *dst = out + (size_t(t) * c.n_head + h) * c.head_dim;
      std::fill(dst, dst + c.head_dim, 0);
      for (int64_t j = 0; j < total; ++j) {
        const float *vv =
            j < pos
                ? old.value_data() + (size_t(j) * c.n_head_kv + kh) * c.head_dim
                : v + (size_t(j - pos) * c.n_head_kv + kh) * c.head_dim;
        for (int64_t d = 0; d < c.head_dim; ++d)
          dst[d] += scores[j] / den * vv[d];
      }
    }
}
} // namespace

int main() {
  // Values independently captured from the authoritative Ring-mini-2.0
  // reference (theta=600000, rope_dim=64), using Float32 inv_freq and
  // positions from the qualification inputs.
  struct RopeFixture {
    int position;
    float angle;
    float cosine;
    float sine;
  };
  const RopeFixture rope_fixture[] = {
      {1, 0.6598311067f, 0.7900957465f, 0.6129834056f},
      {172, 113.4909515f, 0.9235282540f, 0.3835304081f},
      {1024, 675.6670532f, -0.9748759866f, -0.2227483094f},
      {4095, 2702.008301f, 0.9716653824f, 0.2363606840f},
  };
  const float reference_inv_freq = 0.6598311067f;
  check(near_ulp(strata::ring::rope_inverse_frequency(600000.0f, 64, 0),
                 1.0f),
        "RoPE index zero inverse frequency is one");
  check(near_ulp(strata::ring::rope_inverse_frequency(600000.0f, 64, 1),
                 reference_inv_freq),
        "RoPE inverse frequency matches pinned Float32 reference");
  check(near_ulp(strata::ring::rope_inverse_frequency(600000.0f, 64, 31),
                 2.525898935e-6f),
        "RoPE high index inverse frequency matches pinned reference");
  // Index 2 is deliberately bit-pinned to the reciprocal-pow operation;
  // pow(theta, -exponent) produces the adjacent Float32 value here.
  check(bits(strata::ring::rope_inverse_frequency(600000.0f, 64, 2)) ==
            UINT32_C(0x3edee9bd),
        "RoPE uses reciprocal-pow operation for index two");
  const uint32_t original_rope_bits[] = {
      0x3f800000, 0x3f28eab1, 0x3edee9bf, 0x3e9315c1,
      0x3e421a3b, 0x3e001329, 0x3da903f9, 0x3d5f0b1c,
      0x3d132bc4, 0x3cc23749, 0x3c802654, 0x3c291d46,
      0x3bdf2c7e, 0x3b9341cc, 0x3b42545a, 0x3b003983,
      0x3aa93696, 0x3a5f4de6, 0x3a1357d7, 0x39c27171,
      0x39804cb4, 0x39294fea, 0x38df6f53, 0x38936de4,
      0x38428e8c, 0x38005fe9, 0x37a96942, 0x375f90c4,
      0x371383f5, 0x36c2abac, 0x36807320, 0x3629829d};
  bool original_rope_exact = true;
  for (int i = 0; i < 32; ++i)
    original_rope_exact &=
        bits(strata::ring::rope_inverse_frequency(600000.0f, 64, i, true)) ==
        original_rope_bits[i];
  check(original_rope_exact, "original Ring RoPE frequency table is bit exact");
  const float index_two_inv =
      strata::ring::rope_inverse_frequency(600000.0f, 64, 2);
  const float index_two_angle = 4095.0f * index_two_inv;
  check(near_abs(index_two_angle, 1782.868896f) &&
            near_abs(std::cos(index_two_angle), 0.01506500226f) &&
            near_abs(std::sin(index_two_angle), -0.9998865128f),
        "RoPE reciprocal-pow angle remains stable at position 4095");
  for (const RopeFixture &fixture : rope_fixture) {
    const float got_inv =
        strata::ring::rope_inverse_frequency(600000.0f, 64, 1);
    const float got_angle = float(fixture.position) * got_inv;
    check(near_abs(got_angle, fixture.angle),
          "RoPE angle matches pinned Float32 reference");
    check(near_abs(std::cos(got_angle), fixture.cosine) &&
              near_abs(std::sin(got_angle), fixture.sine),
          "RoPE sin/cos match pinned Float32 reference");
  }
  check(std::isnan(strata::ring::rope_inverse_frequency(600000.0f, 64, -1)),
        "invalid RoPE index fails closed");
  check(std::isnan(strata::ring::rope_inverse_frequency(
                    std::numeric_limits<float>::infinity(), 64, 31)),
        "non-finite RoPE input fails closed");
  check(std::isnan(strata::ring::rope_inverse_frequency(500000.0f, 64, 2, true)),
        "original RoPE table rejects unsupported theta");
  check(strata::ring::AttentionConfig{}.rope_dim == 64,
        "Ring default partial RoPE is 64");
  strata::ring::AttentionConfig c;
  c.n_head = 4;
  c.n_head_kv = 2;
  c.head_dim = 4;
  c.rope_dim = 2;
  c.rope_theta = 100.0f;
  const float qw[] = {1.0f, .9f, 1.1f, .8f}, kw[] = {.8f, 1.2f, .7f, 1.1f};
  strata::ring::KvCache cache(8, 2, 4);
  const int64_t nt = 3;
  std::vector<float> q(nt * 4 * 4), k(nt * 2 * 4), v(nt * 2 * 4), got(q.size()),
      want(q.size());
  for (size_t i = 0; i < q.size(); ++i)
    q[i] = std::sin(float(i + 1) * .17f);
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = std::cos(float(i + 2) * .23f);
    v[i] = std::sin(float(i + 3) * .31f);
  }
  std::string err;
  reference(c, q.data(), k.data(), v.data(), nt, 0, qw, kw, cache, want.data());
  check(strata::ring::attention_forward(c, q.data(), k.data(), v.data(), nt, 0,
                                        qw, kw, cache, got.data(), err),
        "prefill succeeds");
  bool same = true;
  for (size_t i = 0; i < got.size(); ++i)
    same &= near(got[i], want[i]);
  check(same, "prefill matches independent golden");
  check(cache.length() == 3, "prefill commits three positions");
  const auto saved_keys = cache.live_keys(), saved_values = cache.live_values();
  strata::ring::KvCache forked(8, 2, 4);
  check(forked.restore_live(3, saved_keys, saved_values, err),
        "KV snapshot restores a fork");
  check(forked.length() == 3 && forked.live_keys() == saved_keys &&
            forked.live_values() == saved_values,
        "KV fork has exactly committed payload");
  auto bad_keys = saved_keys; bad_keys[0] = std::numeric_limits<float>::quiet_NaN();
  check(!forked.restore_live(3, bad_keys, saved_values, err) &&
            forked.live_keys() == saved_keys,
        "nonfinite KV restore is transactional");
  check(!forked.restore_live(2, saved_keys, saved_values, err) &&
            forked.live_keys() == saved_keys,
        "wrong KV restore shape is transactional");
  check(!forked.restore_live(9, saved_keys, saved_values, err) &&
            forked.length() == 3 && forked.live_values() == saved_values,
        "out-of-capacity KV restore is transactional");
  strata::ring::KvCache empty(8, 2, 4);
  check(empty.restore_live(0, {}, {}, err) && empty.length() == 0,
        "empty KV snapshot restores");

  strata::ring::KvCache normalized_cache(8, 2, 4);
  std::vector<float> normalized(q.size());
  check(strata::ring::attention_forward_order(
            c, q.data(), k.data(), v.data(), nt, 0, qw, kw, normalized_cache,
            normalized.data(), err,
            strata::ring::AttentionValueOrder::NormalizeWeightsFirst),
        "original-order normalized-value prefill succeeds");
  bool normalized_same = true;
  for (size_t i = 0; i < normalized.size(); ++i)
    normalized_same &= near(normalized[i], want[i]);
  check(normalized_same, "original-order prefill matches independent golden");
  check(normalized_cache.length() == 3,
        "original-order prefill commits three positions");

  strata::ring::KvCache trace_cache(8, 2, 4);
  std::vector<float> trace_out(16);
  strata::ring::AttentionTrace trace;
  check(strata::ring::attention_forward(
            c, q.data(), k.data(), v.data(), 1, 0, qw, kw, trace_cache,
            trace_out.data(), err, &trace),
        "single-token attention trace succeeds");
  check(trace.n_heads == 4 && trace.n_head_kv == 2 && trace.head_dim == 4 &&
            trace.n_keys == 1 &&
            trace.raw_q.size() == 16 && trace.raw_k.size() == 8 &&
            trace.normalized_q.size() == 16 &&
            trace.rotated_k.size() == 8 && trace.cache_k.size() == 8 &&
            trace.cache_v.size() == 8 && trace.qk_dots.size() == 4 &&
            trace.scores.size() == 4 &&
            trace.probabilities.size() == 4 && trace.output.size() == 16,
        "trace reports documented tensor shapes");
  bool trace_probs = true;
  for (int h = 0; h < 4; ++h)
    trace_probs &= near_abs(trace.probabilities[size_t(h)], 1.0f);
  check(trace_probs, "trace probabilities are normalized");
  check(strata::ring::attention_forward(
            c, q.data() + 16, k.data() + 8, v.data() + 8, 1, 1, qw, kw,
            trace_cache, trace_out.data(), err, &trace),
        "default trace cache replay succeeds");
  trace_probs = true;
  for (int h = 0; h < 4; ++h) {
    float sum = 0.0f;
    for (int64_t j = 0; j < trace.n_keys; ++j)
      sum += trace.probabilities[size_t(h) * size_t(trace.n_keys) + size_t(j)];
    trace_probs &= near_abs(sum, 1.0f);
  }
  check(trace_probs, "default trace probabilities sum to one");
  strata::ring::KvCache divide_cache(8, 2, 4);
  std::vector<float> divide_out(16);
  check(strata::ring::attention_forward_order(
            c, q.data(), k.data(), v.data(), 1, 0, qw, kw, divide_cache,
            divide_out.data(), err,
            strata::ring::AttentionValueOrder::NormalizeWeightsFirst, nullptr,
            strata::ring::AttentionScoreScale::DivideBySqrt),
        "original divide-by-sqrt score scale succeeds");
  strata::ring::AttentionTrace rejected_trace;
  std::vector<float> rejected_out(q.size());
  check(!strata::ring::attention_forward(
            c, q.data(), k.data(), v.data(), nt, 0, qw, kw, trace_cache,
            rejected_out.data(), err, &rejected_trace) &&
            err.find("single-token") != std::string::npos &&
            rejected_trace.output.empty(),
        "multi-token trace request fails closed");

  strata::ring::KvCache old(8, 2, 4);
  std::vector<float> ignored(got.size());
  check(strata::ring::attention_forward(c, q.data(), k.data(), v.data(), nt, 0,
                                        qw, kw, old, ignored.data(), err),
        "independent cache setup");
  const float q1[] = {.2f, .3f,  .4f,  .5f,  .6f,  .7f,  .8f,  .9f,
                      1.f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f};
  const float k1[] = {.3f, .4f, .5f, .6f, .7f, .8f, .9f, 1.f},
              v1[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float o1[16] = {}, w1[16] = {};
  reference(c, q1, k1, v1, 1, 3, qw, kw, old, w1);
  check(strata::ring::attention_forward(c, q1, k1, v1, 1, 3, qw, kw, cache, o1,
                                        err),
        "incremental succeeds");
  same = true;
  for (int i = 0; i < 16; ++i)
    same &= near(o1[i], w1[i]);
  check(same, "incremental matches independent golden");
  check(cache.length() == 4, "incremental appends one position");

  const int64_t before = cache.length();
  std::vector<float> snapshot(cache.key_data(),
                              cache.key_data() + before * 2 * 4),
      badout(16, 77);
  check(!strata::ring::attention_forward(c, q1, k1, v1, 1, 0, qw, kw, cache,
                                         badout.data(), err),
        "non-contiguous position rejected");
  check(cache.length() == before &&
            std::equal(snapshot.begin(), snapshot.end(), cache.key_data()),
        "failed validation preserves cache");

  strata::ring::AttentionConfig one = c;
  one.n_head = one.n_head_kv = 1;
  strata::ring::KvCache tiny(2, 1, 4), single(2, 1, 4);
  const float fq[] = {1, 0, 0, 0, 0, 1, 0, 0}, fk[] = {1, 0, 0, 0, 0, 1, 0, 0},
              fv[] = {3, 4, 5, 6, 7, 8, 9, 10};
  float fo[8] = {}, so[4] = {};
  check(strata::ring::attention_forward(one, fq, fk, fv, 2, 0, nullptr, nullptr,
                                        tiny, fo, err),
        "causal setup");
  check(strata::ring::attention_forward(one, fq, fk, fv, 1, 0, nullptr, nullptr,
                                        single, so, err),
        "single-token setup");
  check(std::equal(fo, fo + 4, so, [](float a, float b) { return near(a, b); }),
        "first token cannot see future");
  check(tiny.length() == 2, "causal prefill commits both positions");

  // A native prefill and its one-token cache replay must agree; this isolates
  // cache staging/causal indexing from the original Torch batched arithmetic.
  strata::ring::KvCache batched_cache(8, 2, 4), replay_cache(8, 2, 4);
  std::vector<float> batched(q.size()), replay(q.size());
  check(strata::ring::attention_forward(c, q.data(), k.data(), v.data(), nt, 0,
                                        qw, kw, batched_cache, batched.data(), err),
        "batched cache replay setup");
  bool replay_ok = true;
  for (int64_t t = 0; t < nt; ++t) {
    std::vector<float> one_out(16);
    replay_ok &= strata::ring::attention_forward(
        c, q.data() + size_t(t) * 16, k.data() + size_t(t) * 8,
        v.data() + size_t(t) * 8, 1, t, qw, kw, replay_cache, one_out.data(),
        err);
    std::copy(one_out.begin(), one_out.end(), replay.begin() + size_t(t) * 16);
  }
  check(replay_ok, "one-token cache replay succeeds");
  bool replay_same = replay_ok;
  for (size_t i = 0; i < replay.size(); ++i)
    replay_same &= near(replay[i], batched[i]);
  check(replay_same, "batched and one-token cache attention agree");
  check(batched_cache.length() == nt && replay_cache.length() == nt,
        "batched and one-token cache lengths agree");

  strata::ring::KvCache partial(2, 1, 4);
  const float pk[] = {4, 3, 2, 1}, pv[] = {1, 2, 3, 4}, pq[] = {1, 2, 3, 4};
  float po[4] = {};
  one.rope_dim = 2;
  check(strata::ring::attention_forward(one, pq, pk, pv, 1, 0, nullptr, nullptr,
                                        partial, po, err),
        "partial-rope setup");
  const float norm = std::sqrt(7.5f);
  check(near(partial.key_data()[2], 2.0f / norm) &&
            near(partial.key_data()[3], 1.0f / norm),
        "rotary tail untouched");

  const float huge_q[] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX},
              normal_k[] = {1, 0, 0, 0}, normal_v[] = {1, 2, 3, 4};
  strata::ring::KvCache numeric(2, 1, 4);
  float numeric_out[4] = {9, 9, 9, 9};
  check(!strata::ring::attention_forward(one, huge_q, normal_k, normal_v, 1, 0,
                                         nullptr, nullptr, numeric, numeric_out,
                                         err),
        "RMS overflow rejected");
  check(numeric.length() == 0 && numeric_out[0] == 9,
        "RMS overflow preserves cache and output");
  const float huge_weight[] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
  check(!strata::ring::attention_forward(one, pq, pk, pv, 1, 0, huge_weight,
                                         nullptr, numeric, numeric_out, err),
        "normalized query overflow rejected");
  check(numeric.length() == 0 && numeric_out[0] == 9,
        "normalized overflow preserves cache and output");

  strata::ring::AttentionConfig bad = one;
  bad.n_head = 3;
  bad.n_head_kv = 2;
  strata::ring::KvCache invalid(2, 1, 4);
  std::vector<float> invout(4, 9);
  check(!strata::ring::attention_forward(bad, pq, pk, pv, 1, 0, nullptr,
                                         nullptr, invalid, invout.data(), err),
        "invalid GQA rejected");
  check(invalid.length() == 0, "invalid geometry leaves empty cache");
  strata::ring::AttentionConfig unsupported = one;
  unsupported.original_rope_frequency = true;
  unsupported.rope_dim = 2;
  check(!strata::ring::attention_forward(
            unsupported, fq, fk, fv, 1, 0, nullptr, nullptr, single, so, err) &&
            err.find("original Ring RoPE frequency") != std::string::npos,
        "original RoPE mode rejects unsupported geometry");
  bool overflow_rejected = false;
  try {
    strata::ring::KvCache impossible(std::numeric_limits<int64_t>::max(), 2, 2);
  } catch (const std::invalid_argument &) {
    overflow_rejected = true;
  }
  check(overflow_rejected, "KV allocation overflow rejected");
  // Pre-normalized Q/K must be equivalent to the ordinary path with
  // nontrivial per-dimension gamma, and must not normalize a second time.
  strata::ring::AttentionConfig pre_cfg = one;
  pre_cfg.n_head = 2; pre_cfg.n_head_kv = 1; pre_cfg.head_dim = 4;
  pre_cfg.rope_dim = 2; pre_cfg.rope_theta = 100.0f;
  const float pre_q[] = {1.2f, -0.7f, 2.1f, 0.4f, -0.3f, 1.7f, 0.8f, -2.2f};
  const float pre_k[] = {0.9f, -1.4f, 0.6f, 2.0f};
  const float pre_v[] = {1.0f, 2.0f, -1.0f, 0.5f};
  const float pre_qw[] = {0.7f, 1.3f, -0.8f, 1.1f};
  const float pre_kw[] = {1.4f, -0.6f, 0.9f, 1.2f};
  auto manual_norm = [](const float *x, float *y, int n, const float *w) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(ss / float(n) + 1e-6f);
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
  };
  float pre_qn[8], pre_kn[4];
  manual_norm(pre_q, pre_qn, 4, pre_qw);
  manual_norm(pre_q + 4, pre_qn + 4, 4, pre_qw);
  manual_norm(pre_k, pre_kn, 4, pre_kw);
  strata::ring::KvCache ordinary_cache(2, 1, 4), normalized_cache2(2, 1, 4);
  float ordinary_out[8] = {}, normalized_out[8] = {};
  strata::ring::AttentionTrace ordinary_trace, normalized_trace;
  std::string pre_err;
  const bool ordinary_ok = strata::ring::attention_forward_order(
      pre_cfg, pre_q, pre_k, pre_v, 1, 0, pre_qw, pre_kw, ordinary_cache,
      ordinary_out, pre_err, strata::ring::AttentionValueOrder::AccumulateUnnormalized,
      &ordinary_trace);
  const bool normalized_ok = strata::ring::attention_forward_order(
      pre_cfg, pre_qn, pre_kn, pre_v, 1, 0, nullptr, nullptr,
      normalized_cache2, normalized_out, pre_err,
      strata::ring::AttentionValueOrder::AccumulateUnnormalized,
      &normalized_trace, strata::ring::AttentionScoreScale::ReciprocalSqrt,
      true);
  bool pre_same = ordinary_ok && normalized_ok;
  for (int i = 0; i < 8; ++i) pre_same &= bits(ordinary_out[i]) == bits(normalized_out[i]);
  for (size_t i = 0; i < ordinary_trace.normalized_q.size() && i < normalized_trace.normalized_q.size(); ++i)
    pre_same &= bits(ordinary_trace.normalized_q[i]) == bits(normalized_trace.normalized_q[i]);
  for (size_t i = 0; i < ordinary_trace.normalized_k.size() && i < normalized_trace.normalized_k.size(); ++i)
    pre_same &= bits(ordinary_trace.normalized_k[i]) == bits(normalized_trace.normalized_k[i]);
  check(pre_same, "pre-normalized Q/K skips duplicate RMSNorm bit exactly");
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
