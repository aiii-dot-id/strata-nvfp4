#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace strata::ring {

/// Returns the Float32 inverse frequency used by the original Ring RoPE
/// construction. Invalid arguments return NaN.
float rope_inverse_frequency(float theta, int64_t rope_dim,
                            int64_t index) noexcept;

struct AttentionConfig {
  int64_t n_head = 16;
  int64_t n_head_kv = 4;
  int64_t head_dim = 128;
  int64_t rope_dim = 64;
  float rms_eps = 1e-6f;
  float rope_theta = 600000.0f;
};

class KvCache {
public:
  KvCache() = default;
  KvCache(int64_t capacity, int64_t n_head_kv, int64_t head_dim);
  KvCache(const KvCache &) = delete;
  KvCache &operator=(const KvCache &) = delete;
  KvCache(KvCache &&) noexcept = default;
  KvCache &operator=(KvCache &&) noexcept = default;

  int64_t capacity() const noexcept { return capacity_; }
  int64_t length() const noexcept { return length_; }
  int64_t n_head_kv() const noexcept { return n_head_kv_; }
  int64_t head_dim() const noexcept { return head_dim_; }
  void clear() noexcept { length_ = 0; }
  const float *key_data() const noexcept { return keys_.data(); }
  const float *value_data() const noexcept { return values_.data(); }

private:
  friend bool attention_forward(const AttentionConfig &, const float *,
                                const float *, const float *, int64_t, int64_t,
                                const float *, const float *, KvCache &,
                                float *, std::string &);
  int64_t capacity_ = 0, length_ = 0, n_head_kv_ = 0, head_dim_ = 0;
  std::vector<float> keys_, values_;
};

/// Inputs are q [tokens, heads, dim], k/v [tokens, kv_heads, dim], and out
/// [tokens, heads, dim]. Position must equal cache.length(). q_norm/k_norm
/// are optional per-dimension RMSNorm weights; nullptr means all ones.
/// Norm precedes partial RoPE. A failed validation leaves cache unchanged.
bool attention_forward(const AttentionConfig &config, const float *q,
                       const float *k, const float *v, int64_t n_tokens,
                       int64_t position, const float *q_norm,
                       const float *k_norm, KvCache &cache, float *out,
                       std::string &error);

} // namespace strata::ring
