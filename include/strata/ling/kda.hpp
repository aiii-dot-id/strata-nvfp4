#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::ling {

// AI Identity Incorported <james@aiii.id>
struct KdaConfig {
  int64_t num_heads = 16;
  int64_t value_heads = 16;
  int64_t key_dim = 128;
  int64_t value_dim = 128;
  int64_t conv_kernel = 4;
  float qk_epsilon = 1.0e-6f;
  float gate_lower_bound = -5.0f;
  bool safe_gate = true;
};

struct KdaWeights {
  const float *q_conv = nullptr;
  const float *k_conv = nullptr;
  const float *v_conv = nullptr;
  const float *q_bias = nullptr;
  const float *k_bias = nullptr;
  const float *v_bias = nullptr;
  const float *a_log = nullptr;
  const float *dt_bias = nullptr;
};

class KdaState {
public:
  KdaState() = default;
  explicit KdaState(const KdaConfig &config);

  KdaState(const KdaState &) = default;
  KdaState &operator=(const KdaState &) = default;
  KdaState(KdaState &&) noexcept = default;
  KdaState &operator=(KdaState &&) noexcept = default;

  int64_t num_heads() const noexcept { return num_heads_; }
  int64_t value_heads() const noexcept { return value_heads_; }
  int64_t key_dim() const noexcept { return key_dim_; }
  int64_t value_dim() const noexcept { return value_dim_; }
  int64_t conv_kernel() const noexcept { return conv_kernel_; }
  // Logical float payload for recurrent and three convolution-state vectors.
  // The KDA state is allocated at construction and does not vary by position.
  size_t logical_payload_bytes() const noexcept;
  const float *recurrent_data() const noexcept { return recurrent_.data(); }
  const float *q_conv_data() const noexcept { return q_conv_state_.data(); }
  const float *k_conv_data() const noexcept { return k_conv_state_.data(); }
  const float *v_conv_data() const noexcept { return v_conv_state_.data(); }
  void clear() noexcept;

private:
  int64_t num_heads_ = 0;
  int64_t value_heads_ = 0;
  int64_t key_dim_ = 0;
  int64_t value_dim_ = 0;
  int64_t conv_kernel_ = 0;
  std::vector<float> recurrent_;
  std::vector<float> q_conv_state_;
  std::vector<float> k_conv_state_;
  std::vector<float> v_conv_state_;

  friend bool kda_forward(const KdaConfig &, const float *, const float *,
                          const float *, const float *, const float *,
                          int64_t, const KdaWeights &, KdaState &, float *,
                          std::string &);
};

// Process projected Ling KDA inputs. q and k are [tokens, H, K], v is
// [tokens, HV, V], gate is [tokens, HV, K], and beta is [tokens, HV] in
// post-sigmoid space. A_log and dt_bias are per-head parameters. The
// convolution weights are depthwise [channels, conv_kernel]. State is
// committed only after the complete call succeeds.
bool kda_forward(const KdaConfig &config, const float *q, const float *k,
                 const float *v, const float *gate, const float *beta,
                 int64_t n_tokens, const KdaWeights &weights, KdaState &state,
                 float *out, std::string &error);

} // namespace strata::ling
