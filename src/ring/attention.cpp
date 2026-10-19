#include "strata/ring/attention.hpp"
#include "strata/ring/numerics.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::ring {

float rope_inverse_frequency(float theta, int64_t rope_dim, int64_t index,
                            bool original_frequency) noexcept {
  if (!(theta > 0.0f) || !std::isfinite(theta) || rope_dim <= 0 ||
      (rope_dim & 1) || index < 0 || index >= rope_dim / 2)
    return std::numeric_limits<float>::quiet_NaN();
  if (original_frequency) {
    static constexpr uint32_t exact_bits[32] = {
        0x3f800000, 0x3f28eab1, 0x3edee9bf, 0x3e9315c1,
        0x3e421a3b, 0x3e001329, 0x3da903f9, 0x3d5f0b1c,
        0x3d132bc4, 0x3cc23749, 0x3c802654, 0x3c291d46,
        0x3bdf2c7e, 0x3b9341cc, 0x3b42545a, 0x3b003983,
        0x3aa93696, 0x3a5f4de6, 0x3a1357d7, 0x39c27171,
        0x39804cb4, 0x39294fea, 0x38df6f53, 0x38936de4,
        0x38428e8c, 0x38005fe9, 0x37a96942, 0x375f90c4,
        0x371383f5, 0x36c2abac, 0x36807320, 0x3629829d};
    if (theta != 600000.0f || rope_dim != 64 || index >= 32)
      return std::numeric_limits<float>::quiet_NaN();
    float value;
    const uint32_t bits = exact_bits[index];
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  const float exponent = 2.0f * float(index) / float(rope_dim);
  const float denominator = std::pow(theta, exponent);
  if (!(denominator > 0.0f) || !std::isfinite(denominator))
    return std::numeric_limits<float>::quiet_NaN();
  const float inverse = 1.0f / denominator;
  return std::isfinite(inverse) ? inverse
                                 : std::numeric_limits<float>::quiet_NaN();
}

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
  if (c.original_rope_frequency &&
      (c.rope_theta != 600000.0f || c.rope_dim != 64)) {
    e = "original Ring RoPE frequency requires theta=600000 and rope_dim=64";
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
  if (!x || !y || dim <= 0)
    return false;
  float sum = 0.0f;
  try {
    sum = balanced_sum_squares(x, size_t(dim));
  } catch (const std::exception &) {
    return false;
  }
  const float variance = sum / float(dim) + eps;
  const float inv = 1.0f / std::sqrt(variance);
  if (!std::isfinite(variance) || !std::isfinite(inv))
    return false;
  for (int64_t i = 0; i < dim; ++i) {
    y[i] = x[i] * inv * (w ? w[i] : 1.0f);
    if (!std::isfinite(y[i]))
      return false;
  }
  return true;
}
bool rope(const float *x, float *y, int64_t dim, int64_t rope_dim,
          int64_t position, float theta, bool original_frequency) {
  std::copy(x, x + dim, y);
  const int64_t half = rope_dim / 2;
  for (int64_t i = 0; i < half; ++i) {
    const float inv =
        rope_inverse_frequency(theta, rope_dim, i, original_frequency);
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
                       float *out, std::string &error, AttentionTrace *trace) {
  return attention_forward_order(
      c, q, k, v, n_tokens, position, q_norm, k_norm, cache, out, error,
      AttentionValueOrder::AccumulateUnnormalized, trace);
}

bool attention_forward_order(const AttentionConfig &c, const float *q,
                             const float *k, const float *v,
                             int64_t n_tokens, int64_t position,
                             const float *q_norm, const float *k_norm,
                             KvCache &cache, float *out, std::string &error,
                             AttentionValueOrder value_order,
                             AttentionTrace *trace,
                             AttentionScoreScale score_scale, bool pre_normalized_qk,
                             AttentionBackend *backend) {
  error.clear();
  if (trace)
    trace->clear();
  if (!valid_geometry(c, error))
    return false;
  if (!positive(n_tokens) || position < 0) {
    error = "n_tokens must be positive and position must be non-negative";
    return false;
  }
  if (trace && n_tokens != 1) {
    error = "attention trace requires a single-token call";
    return false;
  }
  if (score_scale != AttentionScoreScale::ReciprocalSqrt &&
      score_scale != AttentionScoreScale::DivideBySqrt) {
    error = "invalid attention score scale";
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
  const float sqrt_dim = std::sqrt(float(c.head_dim));
  const float scale = 1.0f / sqrt_dim;
  const auto scale_score = [&](float dot) {
    return score_scale == AttentionScoreScale::DivideBySqrt
               ? dot / sqrt_dim
               : dot * scale;
  };
  if (trace) {
    trace->position = position;
    trace->n_heads = c.n_head;
    trace->n_head_kv = c.n_head_kv;
    trace->head_dim = c.head_dim;
    trace->n_keys = position + 1;
    trace->raw_q.assign(q, q + size_t(c.n_head) * d);
    trace->raw_k.assign(k, k + size_t(c.n_head_kv) * d);
    trace->normalized_q.resize(size_t(c.n_head) * d);
    trace->rotated_q.resize(size_t(c.n_head) * d);
    trace->normalized_k.resize(size_t(c.n_head_kv) * d);
    trace->rotated_k.resize(size_t(c.n_head_kv) * d);
    trace->cache_k.resize(size_t(trace->n_keys) * nk * d);
    trace->cache_v.resize(size_t(trace->n_keys) * nk * d);
    trace->qk_dots.resize(size_t(c.n_head) * size_t(trace->n_keys));
    trace->scores.resize(size_t(c.n_head) * size_t(trace->n_keys));
    trace->probabilities.resize(size_t(c.n_head) * size_t(trace->n_keys));
    trace->output.resize(size_t(c.n_head) * d);
  }

  // Stage every new K/V row before computing any query, so a prefill query
  // can attend to prior rows in the same call but never future rows.
  for (int64_t t = 0; t < n_tokens; ++t) {
    const int64_t pos = position + t;
    for (int64_t h = 0; h < c.n_head_kv; ++h) {
      const float *krow = k + (size_t(t) * c.n_head_kv + h) * c.head_dim;
      if (!((pre_normalized_qk ? (std::copy(krow, krow + c.head_dim, kn.data()), finite_array(kn.data(), size_t(c.head_dim))) : rms_norm(krow, kn.data(), c.head_dim, c.rms_eps, k_norm)) &&
          rope(kn.data(), kr.data(), c.head_dim, c.rope_dim, pos,
                c.rope_theta, c.original_rope_frequency))) {
        error = "non-finite transformed key";
        return false;
      }
      std::copy(kr.data(), kr.data() + c.head_dim,
                staged_k.data() + (size_t(t) * c.n_head_kv + h) * c.head_dim);
      if (trace) {
        std::copy(kn.begin(), kn.end(), trace->normalized_k.data() + size_t(h) * d);
        std::copy(kr.begin(), kr.end(), trace->rotated_k.data() + size_t(h) * d);
      }
      std::copy(v + (size_t(t) * c.n_head_kv + h) * c.head_dim,
                v + (size_t(t) * c.n_head_kv + h + 1) * c.head_dim,
                staged_v.data() + (size_t(t) * c.n_head_kv + h) * c.head_dim);
    }
  }

  // The opt-in backend preserves the native cache and Q/K construction while
  // replacing only the reference expression Q @ K.T / sqrt(d), softmax, P @ V.
  // It intentionally accepts one token at a time: the Ring executor is causal
  // tokenwise, and that fixes the original CUDA operator shapes.
  if (backend) {
    if (n_tokens != 1) {
      error = "attention backend requires a one-token causal call";
      return false;
    }
    const int64_t total = position + 1;
    std::vector<float> transformed_q(size_t(c.n_head) * d);
    for (int64_t h = 0; h < c.n_head; ++h) {
      const float *qrow = q + size_t(h) * c.head_dim;
      float *qdst = transformed_q.data() + size_t(h) * d;
      if (!((pre_normalized_qk
                 ? (std::copy(qrow, qrow + c.head_dim, qn.data()),
                    finite_array(qn.data(), size_t(c.head_dim)))
                 : rms_norm(qrow, qn.data(), c.head_dim, c.rms_eps, q_norm)) &&
            rope(qn.data(), qdst, c.head_dim, c.rope_dim, position,
                 c.rope_theta, c.original_rope_frequency))) {
        error = "non-finite transformed query";
        return false;
      }
      if (trace) {
        std::copy(qn.begin(), qn.end(), trace->normalized_q.data() + size_t(h) * d);
        std::copy(qdst, qdst + c.head_dim,
                  trace->rotated_q.data() + size_t(h) * d);
      }
    }
    std::vector<float> all_k(size_t(total) * nk * d),
        all_v(size_t(total) * nk * d), backend_context, backend_scores,
        backend_probabilities;
    const size_t old_count = size_t(position) * nk * d;
    std::copy(cache.key_data(), cache.key_data() + old_count, all_k.begin());
    std::copy(cache.value_data(), cache.value_data() + old_count, all_v.begin());
    std::copy(staged_k.begin(), staged_k.end(), all_k.begin() + old_count);
    std::copy(staged_v.begin(), staged_v.end(), all_v.begin() + old_count);
    if (!backend->forward(transformed_q.data(), all_k.data(), all_v.data(),
                          c.n_head, c.n_head_kv, c.head_dim, total,
                          backend_context, backend_scores,
                          backend_probabilities, error))
      return false;
    if (backend_context.size() != q_count ||
        backend_scores.size() != size_t(c.n_head) * size_t(total) ||
        backend_probabilities.size() != backend_scores.size() ||
        !finite_array(backend_context.data(), backend_context.size()) ||
        !finite_array(backend_scores.data(), backend_scores.size()) ||
        !finite_array(backend_probabilities.data(), backend_probabilities.size())) {
      error = "attention backend returned invalid output geometry";
      return false;
    }
    result = std::move(backend_context);
    if (trace) {
      trace->scores = std::move(backend_scores);
      trace->probabilities = std::move(backend_probabilities);
    }
  } else {

  for (int64_t t = 0; t < n_tokens; ++t) {
    const int64_t pos = position + t;
    for (int64_t h = 0; h < c.n_head; ++h) {
      const float *qrow = q + (size_t(t) * c.n_head + h) * c.head_dim;
      if (!((pre_normalized_qk ? (std::copy(qrow, qrow + c.head_dim, qn.data()), finite_array(qn.data(), size_t(c.head_dim))) : rms_norm(qrow, qn.data(), c.head_dim, c.rms_eps, q_norm)) &&
          rope(qn.data(), qr.data(), c.head_dim, c.rope_dim, pos,
                c.rope_theta, c.original_rope_frequency))) {
        error = "non-finite transformed query";
        return false;
      }
      if (trace) {
        std::copy(qn.begin(), qn.end(), trace->normalized_q.data() + size_t(h) * d);
        std::copy(qr.begin(), qr.end(), trace->rotated_q.data() + size_t(h) * d);
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
        const float score = scale_score(dot);
        if (!std::isfinite(score)) {
          error = "non-finite attention score";
          return false;
        }
        max_score = std::max(max_score, score);
        if (trace) {
          trace->qk_dots[size_t(h) * size_t(total) + size_t(j)] = dot;
          trace->scores[size_t(h) * size_t(total) + size_t(j)] = score;
        }
      }
      if (!std::isfinite(max_score)) {
        error = "non-finite attention score";
        return false;
      }
      std::fill(dst, dst + c.head_dim, 0.0f);
      if (value_order == AttentionValueOrder::NormalizeWeightsFirst) {
        // Match the original implementation: normalize each probability
        // before multiplying it by V.  Keep this separate so the default
        // unnormalized accumulation remains byte-identical.
        std::vector<float> weights(static_cast<size_t>(total));
        float denom = 0.0f;
        for (int64_t j = 0; j < total; ++j) {
          const float *kk;
          if (j < position)
            kk = cache.key_data() +
                 (size_t(j) * c.n_head_kv + kh) * c.head_dim;
          else
            kk = staged_k.data() +
                 (size_t(j - position) * c.n_head_kv + kh) * c.head_dim;
          float dot = 0.0f;
          for (int64_t d = 0; d < c.head_dim; ++d)
            dot += qr[d] * kk[d];
          const float score = scale_score(dot);
          const float a = std::exp(score - max_score);
          if (!std::isfinite(score) || !std::isfinite(a) ||
              !std::isfinite(denom += a)) {
            error = "non-finite attention softmax";
            return false;
          }
          weights[size_t(j)] = a;
        }
        if (!(denom > 0.0f) || !std::isfinite(denom)) {
          error = "invalid attention softmax denominator";
          return false;
        }
        for (int64_t j = 0; j < total; ++j) {
          const float *vv;
          if (j < position)
            vv = cache.value_data() +
                 (size_t(j) * c.n_head_kv + kh) * c.head_dim;
          else
            vv = staged_v.data() +
                 (size_t(j - position) * c.n_head_kv + kh) * c.head_dim;
          const float a = weights[size_t(j)] / denom;
          if (trace)
            trace->probabilities[size_t(h) * size_t(total) + size_t(j)] = a;
          for (int64_t d = 0; d < c.head_dim; ++d) {
            dst[d] += a * vv[d];
            if (!std::isfinite(dst[d])) {
              error = "non-finite attention output";
              return false;
            }
          }
        }
      } else {
        float denom = 0.0f;
        for (int64_t j = 0; j < total; ++j) {
          const float *kk;
          const float *vv;
          if (j < position) {
            kk = cache.key_data() +
                 (size_t(j) * c.n_head_kv + kh) * c.head_dim;
            vv = cache.value_data() +
                 (size_t(j) * c.n_head_kv + kh) * c.head_dim;
          } else {
            kk = staged_k.data() +
                 (size_t(j - position) * c.n_head_kv + kh) * c.head_dim;
            vv = staged_v.data() +
                 (size_t(j - position) * c.n_head_kv + kh) * c.head_dim;
          }
          float dot = 0.0f;
          for (int64_t d = 0; d < c.head_dim; ++d)
            dot += qr[d] * kk[d];
          const float score = scale_score(dot);
          if (!std::isfinite(score)) {
            error = "non-finite attention score";
            return false;
          }
          const float a = std::exp(score - max_score);
          if (!std::isfinite(a) || !std::isfinite(denom += a)) {
            error = "non-finite attention softmax";
            return false;
          }
          if (trace)
            trace->probabilities[size_t(h) * size_t(total) + size_t(j)] = a;
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
        if (trace) {
          for (int64_t j = 0; j < total; ++j)
            trace->probabilities[size_t(h) * size_t(total) + size_t(j)] /=
                denom;
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
  }
  }

  const size_t off =
      size_t(cache.length()) * size_t(c.n_head_kv) * size_t(c.head_dim);
  std::copy(staged_k.begin(), staged_k.end(), cache.keys_.begin() + off);
  std::copy(staged_v.begin(), staged_v.end(), cache.values_.begin() + off);
  cache.length_ += n_tokens;
  std::copy(result.begin(), result.end(), out);
  if (trace) {
    const size_t cache_count = size_t(trace->n_keys) * nk * d;
    std::copy(cache.key_data(), cache.key_data() + cache_count,
              trace->cache_k.begin());
    std::copy(cache.value_data(), cache.value_data() + cache_count,
              trace->cache_v.begin());
    std::copy(result.begin(), result.end(), trace->output.begin());
  }
  return true;
}
} // namespace strata::ring
