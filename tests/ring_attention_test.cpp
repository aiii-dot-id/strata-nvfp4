#include "strata/ring/attention.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
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
      const float z =
          std::pow(c.rope_theta, -2.0f * float(d) / float(c.rope_dim));
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
  bool overflow_rejected = false;
  try {
    strata::ring::KvCache impossible(std::numeric_limits<int64_t>::max(), 2, 2);
  } catch (const std::invalid_argument &) {
    overflow_rejected = true;
  }
  check(overflow_rejected, "KV allocation overflow rejected");
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
