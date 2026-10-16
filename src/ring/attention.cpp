#include "strata/ring/attention.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace strata::ring {
namespace {
bool positive(int64_t x) { return x > 0; }
bool valid_geometry(const AttentionConfig &c, std::string &e) {
  if (!positive(c.n_head) || !positive(c.n_head_kv) || !positive(c.head_dim)) {
    e = "attention head counts and head_dim must be positive";
    return false;
  }
  if (c.n_head % c.n_head_kv) {
    e = "n_head must be divisible by n_head_kv";
    return false;
  }
  if (c.rope_dim < 0 || c.rope_dim > c.head_dim || (c.rope_dim & 1)) {
    e = "rope_dim must be even and within head_dim";
    return false;
  }
  if (!(c.rms_eps > 0.0f) || !(c.rope_theta > 0.0f) ||
      !std::isfinite(c.rms_eps) || !std::isfinite(c.rope_theta)) {
    e = "rms_eps and rope_theta must be finite and positive";
    return false;
  }
  return true;
}
bool finite_array(const float *p, size_t n) {
  if (!p)
    return false;
  for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(p[i]))
      return false;
  return true;
}
bool rms_norm(const float *x, float *y, int64_t dim, float eps,
              const float *w) {
  float sum = 0.0f;
  for (int64_t i = 0; i < dim; ++i) {
    const float term = x[i] * x[i];
    if (!std::isfinite(term) || !std::isfinite(sum += term))
      return false;
  }
  const float inv = 1.0f / std::sqrt(sum / float(dim) + eps);
  if (!std::isfinite(inv))
    return false;
  for (int64_t i = 0; i < dim; ++i) {
    y[i] = x[i] * inv * (w ? w[i] : 1.0f);
    if (!std::isfinite(y[i]))
      return false;
  }
  return true;
}
bool rope(const float *x, float *y, int64_t dim, int64_t rope_dim,
          int64_t position, float theta) {
  std::copy(x, x + dim, y);
  const int64_t half = rope_dim / 2;
  for (int64_t i = 0; i < half; ++i) {
    const float inv = std::pow(theta, -2.0f * float(i) / float(rope_dim));
    const float angle = float(position) * inv;
    const float co = std::cos(angle), si = std::sin(angle);
    const float a = x[i], b = x[i + half];
    y[i] = a * co - b * si;
    y[i + half] = b * co + a * si;
    if (!std::isfinite(y[i]) || !std::isfinite(y[i + half]))
      return false;
  }
  for (int64_t i = 0; i < dim; ++i)
    if (!std::isfinite(y[i]))
      return false;
  return true;
}
} // namespace

KvCache::KvCache(int64_t capacity, int64_t n_head_kv, int64_t head_dim)
    : capacity_(capacity), n_head_kv_(n_head_kv), head_dim_(head_dim) {
  if (!positive(capacity) || !positive(n_head_kv) || !positive(head_dim))
    throw std::invalid_argument("KvCache dimensions must be positive");
  const size_t a = size_t(capacity), b = size_t(n_head_kv),
               d = size_t(head_dim);
  if (a > std::numeric_limits<size_t>::max() / b ||
      a * b > std::numeric_limits<size_t>::max() / d)
    throw std::invalid_argument("KvCache allocation size overflows");
  const size_t n = a * b * d;
  keys_.resize(n);
  values_.resize(n);
}

bool attention_forward(const AttentionConfig &c, const float *q, const float *k,
                       const float *v, int64_t n_tokens, int64_t position,
                       const float *q_norm, const float *k_norm, KvCache &cache,
                       float *out, std::string &error) {
  error.clear();
  if (!valid_geometry(c, error))
    return false;
  if (!positive(n_tokens) || position < 0) {
    error = "n_tokens must be positive and position must be non-negative";
    return false;
  }
  if (position != cache.length()) {
    error = "position must equal the current cache length";
    return false;
  }
  if (n_tokens > cache.capacity() - cache.length()) {
    error = "attention call exceeds KV cache capacity";
    return false;
  }
  if (cache.n_head_kv() != c.n_head_kv || cache.head_dim() != c.head_dim) {
    error = "KV cache geometry does not match attention geometry";
    return false;
  }
  if (!q || !k || !v || !out) {
    error = "attention input and output pointers must be non-null";
    return false;
  }
  const size_t nt = size_t(n_tokens), nh = size_t(c.n_head),
               nk = size_t(c.n_head_kv), d = size_t(c.head_dim);
  if (nt > std::numeric_limits<size_t>::max() / nh ||
      nt * nh > std::numeric_limits<size_t>::max() / d ||
      nt > std::numeric_limits<size_t>::max() / nk ||
      nt * nk > std::numeric_limits<size_t>::max() / d) {
    error = "attention tensor size overflows";
    return false;
  }
  const size_t q_count = nt * nh * d;
  const size_t kv_count = nt * nk * d;
  if (!finite_array(q, q_count) || !finite_array(k, kv_count) ||
      !finite_array(v, kv_count)) {
    error = "attention inputs must be finite";
    return false;
  }
  if (q_norm && !finite_array(q_norm, size_t(c.head_dim))) {
    error = "q_norm contains a non-finite value";
    return false;
  }
  if (k_norm && !finite_array(k_norm, size_t(c.head_dim))) {
    error = "k_norm contains a non-finite value";
    return false;
  }

  std::vector<float> staged_k(kv_count), staged_v(kv_count), result(q_count);
  std::vector<float> qn(size_t(c.head_dim)), kn(size_t(c.head_dim)),
      qr(size_t(c.head_dim)), kr(size_t(c.head_dim));
  const int64_t group = c.n_head / c.n_head_kv;
  const float scale = 1.0f / std::sqrt(float(c.head_dim));

  // Stage every new K/V row before computing any query, so a prefill query
  // can attend to prior rows in the same call but never future rows.
  for (int64_t t = 0; t < n_tokens; ++t) {
    const int64_t pos = position + t;
    for (int64_t h = 0; h < c.n_head_kv; ++h) {
      const float *krow = k + (size_t(t) * c.n_head_kv + h) * c.head_dim;
      if (!rms_norm(krow, kn.data(), c.head_dim, c.rms_eps, k_norm) ||
          !rope(kn.data(), kr.data(), c.head_dim, c.rope_dim, pos,
                c.rope_theta)) {
        error = "non-finite transformed key";
        return false;
      }
      std::copy(kr.data(), kr.data() + c.head_dim,
                staged_k.data() + (size_t(t) * c.n_head_kv + h) * c.head_dim);
      std::copy(v + (size_t(t) * c.n_head_kv + h) * c.head_dim,
                v + (size_t(t) * c.n_head_kv + h + 1) * c.head_dim,
                staged_v.data() + (size_t(t) * c.n_head_kv + h) * c.head_dim);
    }
  }

  for (int64_t t = 0; t < n_tokens; ++t) {
    const int64_t pos = position + t;
    for (int64_t h = 0; h < c.n_head; ++h) {
      const float *qrow = q + (size_t(t) * c.n_head + h) * c.head_dim;
      if (!rms_norm(qrow, qn.data(), c.head_dim, c.rms_eps, q_norm) ||
          !rope(qn.data(), qr.data(), c.head_dim, c.rope_dim, pos,
                c.rope_theta)) {
        error = "non-finite transformed query";
        return false;
      }
      const int64_t kh = h / group;
      const int64_t total = position + t + 1;
      float *dst = result.data() + (size_t(t) * c.n_head + h) * c.head_dim;
      float max_score = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j < total; ++j) {
        const float *kk =
            j < position
                ? cache.key_data() + (size_t(j) * c.n_head_kv + kh) * c.head_dim
                : staged_k.data() +
                      (size_t(j - position) * c.n_head_kv + kh) * c.head_dim;
        float dot = 0.0f;
        for (int64_t d = 0; d < c.head_dim; ++d) {
          dot += qr[d] * kk[d];
          if (!std::isfinite(dot)) {
            error = "non-finite attention score";
            return false;
          }
        }
        const float score = dot * scale;
        if (!std::isfinite(score)) {
          error = "non-finite attention score";
          return false;
        }
        max_score = std::max(max_score, score);
      }
      if (!std::isfinite(max_score)) {
        error = "non-finite attention score";
        return false;
      }
      std::fill(dst, dst + c.head_dim, 0.0f);
      float denom = 0.0f;
      for (int64_t j = 0; j < total; ++j) {
        const float *kk;
        const float *vv;
        if (j < position) {
          kk = cache.key_data() + (size_t(j) * c.n_head_kv + kh) * c.head_dim;
          vv = cache.value_data() + (size_t(j) * c.n_head_kv + kh) * c.head_dim;
        } else {
          kk = staged_k.data() +
               (size_t(j - position) * c.n_head_kv + kh) * c.head_dim;
          vv = staged_v.data() +
               (size_t(j - position) * c.n_head_kv + kh) * c.head_dim;
        }
        float dot = 0.0f;
        for (int64_t d = 0; d < c.head_dim; ++d)
          dot += qr[d] * kk[d];
        const float score = dot * scale;
        if (!std::isfinite(score)) {
          error = "non-finite attention score";
          return false;
        }
        const float a = std::exp(score - max_score);
        if (!std::isfinite(a) || !std::isfinite(denom += a)) {
          error = "non-finite attention softmax";
          return false;
        }
        for (int64_t d = 0; d < c.head_dim; ++d) {
          dst[d] += a * vv[d];
          if (!std::isfinite(dst[d])) {
            error = "non-finite attention output";
            return false;
          }
        }
      }
      if (!(denom > 0.0f) || !std::isfinite(denom)) {
        error = "invalid attention softmax denominator";
        return false;
      }
      for (int64_t d = 0; d < c.head_dim; ++d) {
        dst[d] /= denom;
        if (!std::isfinite(dst[d])) {
          error = "non-finite attention output";
          return false;
        }
      }
    }
  }

  const size_t off =
      size_t(cache.length()) * size_t(c.n_head_kv) * size_t(c.head_dim);
  std::copy(staged_k.begin(), staged_k.end(), cache.keys_.begin() + off);
  std::copy(staged_v.begin(), staged_v.end(), cache.values_.begin() + off);
  cache.length_ += n_tokens;
  std::copy(result.begin(), result.end(), out);
  return true;
}
} // namespace strata::ring
