#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace strata::ring {

/// Returns the Float32 inverse frequency used by the original Ring RoPE
/// construction. Invalid arguments return NaN.
float rope_inverse_frequency(float theta, int64_t rope_dim,
                            int64_t index,
                            bool original_frequency = false) noexcept;

struct AttentionConfig {
  int64_t n_head = 16;
  int64_t n_head_kv = 4;
  int64_t head_dim = 128;
  int64_t rope_dim = 64;
  float rms_eps = 1e-6f;
  float rope_theta = 600000.0f;
  // Opt-in exact float32 frequency table emitted by the original Ring
  // RotaryEmbedding for theta=600000 and rope_dim=64.
  bool original_rope_frequency = false;
};

// The default preserves the existing native accumulation order.  The
// normalized-weight order mirrors the original Ring implementation's
// probability-before-value-matmul sequence and is opt-in for diagnostics.
enum class AttentionValueOrder {
  AccumulateUnnormalized,
  NormalizeWeightsFirst,
};

/// Selects the final QK scaling operation. The reciprocal form is the
/// existing native default; division is an opt-in diagnostic matching the
/// original reference expression.
enum class AttentionScoreScale {
  ReciprocalSqrt,
  DivideBySqrt,
};

/// Optional diagnostics for one-token calls. Tensor vectors use row-major
/// shapes documented by the fields below; no payload is captured unless the
/// caller supplies this object.
struct AttentionTrace {
  int64_t position = 0;
  int64_t n_heads = 0, n_head_kv = 0, head_dim = 0, n_keys = 0;
  std::vector<float> raw_q, normalized_q, rotated_q; // [heads, dim]
  std::vector<float> raw_k, normalized_k, rotated_k; // [kv_heads, dim]
  std::vector<float> cache_k, cache_v;               // [keys, kv_heads, dim]
  std::vector<float> qk_dots, scores, probabilities; // [heads, keys]
  std::vector<float> output;                         // [heads, dim]
  void clear() noexcept {
    position = n_heads = n_head_kv = head_dim = n_keys = 0;
    raw_q.clear(); normalized_q.clear(); rotated_q.clear();
    raw_k.clear(); normalized_k.clear(); rotated_k.clear();
    cache_k.clear(); cache_v.clear(); qk_dots.clear(); scores.clear();
    probabilities.clear();
    output.clear();
  }
};

/// Optional executor for the QK/softmax/AV portion of a one-token attention
/// call.  Inputs have already received Q/K RMSNorm and RoPE.  The key and
/// value arrays are [keys, kv_heads, dim], and output arrays are [heads, ...].
/// The native executor remains the default; this interface lets a diagnostic
/// backend reproduce a separately qualified operator sequence.
class AttentionBackend {
public:
  virtual ~AttentionBackend() = default;
  virtual bool forward(const float *query, const float *keys,
                       const float *values, int64_t heads,
                       int64_t kv_heads, int64_t dim, int64_t key_count,
                       std::vector<float> &context,
                       std::vector<float> &scores,
                       std::vector<float> &probabilities,
                       std::string &error) = 0;
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
  // Copies exactly the committed rows, never spare capacity.
  std::vector<float> live_keys() const;
  std::vector<float> live_values() const;
  // Validates a complete replacement before mutating this cache.
  bool restore_live(int64_t length, const std::vector<float> &keys,
                    const std::vector<float> &values, std::string &error);

private:
  friend bool attention_forward_order(const AttentionConfig &, const float *,
                                      const float *, const float *, int64_t,
                                      int64_t, const float *, const float *,
                                      KvCache &, float *, std::string &,
                                      AttentionValueOrder, AttentionTrace *,
                                      AttentionScoreScale, bool,
                                      AttentionBackend *);
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
                       std::string &error, AttentionTrace *trace = nullptr);

/// Same operation with an explicit value accumulation order.  The cache is
/// committed only after the complete call succeeds.
bool attention_forward_order(
    const AttentionConfig &config, const float *q, const float *k,
    const float *v, int64_t n_tokens, int64_t position, const float *q_norm,
    const float *k_norm, KvCache &cache, float *out, std::string &error,
    AttentionValueOrder value_order, AttentionTrace *trace = nullptr,
    AttentionScoreScale score_scale = AttentionScoreScale::ReciprocalSqrt,
    bool pre_normalized_qk = false, AttentionBackend *backend = nullptr);

} // namespace strata::ring
