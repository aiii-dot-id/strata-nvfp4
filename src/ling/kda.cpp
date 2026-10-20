#include "strata/ling/kda.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace strata::ling {
namespace {

bool positive(int64_t value) { return value > 0; }

bool checked_mul(size_t a, size_t b, size_t &result) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    return false;
  result = a * b;
  return true;
}

bool valid_config(const KdaConfig &c, std::string &error) {
  if (!positive(c.num_heads) || !positive(c.value_heads) ||
      !positive(c.key_dim) || !positive(c.value_dim) ||
      !positive(c.conv_kernel)) {
    error = "KDA geometry must be positive";
    return false;
  }
  if (c.value_heads % c.num_heads != 0) {
    error = "value_heads must be divisible by num_heads";
    return false;
  }
  if (!(c.qk_epsilon > 0.0f) || !std::isfinite(c.qk_epsilon) ||
      !std::isfinite(c.gate_lower_bound)) {
    error = "KDA epsilon and gate bound must be finite and valid";
    return false;
  }
  if (c.safe_gate && !(c.gate_lower_bound <= 0.0f)) {
    error = "safe gate lower bound must be non-positive";
    return false;
  }
  return true;
}

bool same_geometry(const KdaConfig &c, const KdaState &s) {
  return c.num_heads == s.num_heads() && c.value_heads == s.value_heads() &&
         c.key_dim == s.key_dim() && c.value_dim == s.value_dim() &&
         c.conv_kernel == s.conv_kernel();
}

bool finite_values(const float *values, size_t count) {
  if (!values)
    return false;
  for (size_t i = 0; i < count; ++i)
    if (!std::isfinite(values[i]))
      return false;
  return true;
}

bool finite_vector(const std::vector<float> &values) {
  return finite_values(values.data(), values.size());
}

float sigmoid(float value) {
  if (value >= 0.0f) {
    const float e = std::exp(-value);
    return 1.0f / (1.0f + e);
  }
  const float e = std::exp(value);
  return e / (1.0f + e);
}

float softplus(float value) {
  if (value > 0.0f)
    return value + std::log1p(std::exp(-value));
  return std::log1p(std::exp(value));
}

bool conv_token(const float *input, const float *weight, const float *bias,
                int64_t channels, int64_t kernel, std::vector<float> &history,
                std::vector<float> &output) {
  output.resize(size_t(channels));
  std::vector<float> next(history.size());
  for (int64_t channel = 0; channel < channels; ++channel) {
    const size_t base = size_t(channel) * size_t(kernel);
    for (int64_t w = 0; w + 1 < kernel; ++w)
      next[base + size_t(w)] = history[base + size_t(w + 1)];
    next[base + size_t(kernel - 1)] = input[channel];
    float value = bias ? bias[channel] : 0.0f;
    for (int64_t w = 0; w < kernel; ++w) {
      const float term = next[base + size_t(w)] * weight[base + size_t(w)];
      if (!std::isfinite(term) || !std::isfinite(value += term))
        return false;
    }
    value = value * sigmoid(value);
    if (!std::isfinite(value))
      return false;
    output[size_t(channel)] = value;
  }
  history.swap(next);
  return true;
}

bool l2_normalize(const float *input, int64_t dim, float epsilon,
                  std::vector<float> &output) {
  float sum = 0.0f;
  output.resize(size_t(dim));
  for (int64_t i = 0; i < dim; ++i) {
    const float term = input[i] * input[i];
    if (!std::isfinite(term) || !std::isfinite(sum += term))
      return false;
  }
  const float variance = sum + epsilon;
  if (!(variance > 0.0f) || !std::isfinite(variance))
    return false;
  const float inv = 1.0f / std::sqrt(variance);
  if (!std::isfinite(inv))
    return false;
  for (int64_t i = 0; i < dim; ++i) {
    output[size_t(i)] = input[i] * inv;
    if (!std::isfinite(output[size_t(i)]))
      return false;
  }
  return true;
}

} // namespace

KdaState::KdaState(const KdaConfig &config) {
  std::string error;
  if (!valid_config(config, error))
    throw std::invalid_argument(error);
  num_heads_ = config.num_heads;
  value_heads_ = config.value_heads;
  key_dim_ = config.key_dim;
  value_dim_ = config.value_dim;
  conv_kernel_ = config.conv_kernel;
  const size_t h = size_t(value_heads_);
  const size_t qh = size_t(num_heads_);
  const size_t k = size_t(key_dim_);
  const size_t v = size_t(value_dim_);
  const size_t w = size_t(conv_kernel_);
  size_t hk = 0, hkv = 0, qkw = 0, vvw = 0;
  if (!checked_mul(h, k, hk) || !checked_mul(hk, v, hkv) ||
      !checked_mul(qh, k, qkw) || !checked_mul(qkw, w, qkw) ||
      !checked_mul(h, v, vvw) || !checked_mul(vvw, w, vvw))
    throw std::invalid_argument("KDA state size overflows");
  recurrent_.assign(hkv, 0.0f);
  q_conv_state_.assign(qkw, 0.0f);
  k_conv_state_.assign(qkw, 0.0f);
  v_conv_state_.assign(vvw, 0.0f);
}

size_t KdaState::logical_payload_bytes() const noexcept {
  return (recurrent_.size() + q_conv_state_.size() + k_conv_state_.size() +
          v_conv_state_.size()) * sizeof(float);
}

void KdaState::clear() noexcept {
  std::fill(recurrent_.begin(), recurrent_.end(), 0.0f);
  std::fill(q_conv_state_.begin(), q_conv_state_.end(), 0.0f);
  std::fill(k_conv_state_.begin(), k_conv_state_.end(), 0.0f);
  std::fill(v_conv_state_.begin(), v_conv_state_.end(), 0.0f);
}

bool kda_forward(const KdaConfig &c, const float *q, const float *k,
                 const float *v, const float *gate, const float *beta,
                 int64_t n_tokens, const KdaWeights &weights, KdaState &state,
                 float *out, std::string &error) {
  error.clear();
  if (!valid_config(c, error))
    return false;
  if (n_tokens <= 0 || !same_geometry(c, state)) {
    error = n_tokens <= 0 ? "KDA token count must be positive"
                          : "KDA state geometry does not match config";
    return false;
  }
  const size_t h = size_t(c.num_heads), hv = size_t(c.value_heads);
  const size_t kd = size_t(c.key_dim), vd = size_t(c.value_dim),
               w = size_t(c.conv_kernel), tokens = size_t(n_tokens);
  size_t q_channels = 0, v_channels = 0, q_count = 0, v_count = 0,
         g_channels = 0, g_count = 0, b_count = 0, q_weight_count = 0,
         v_weight_count = 0, dt_count = 0;
  const bool sizes_ok =
      checked_mul(h, kd, q_channels) && checked_mul(hv, vd, v_channels) &&
      checked_mul(tokens, q_channels, q_count) &&
      checked_mul(tokens, v_channels, v_count) && checked_mul(hv, kd, g_channels) &&
      checked_mul(tokens, g_channels, g_count) && checked_mul(tokens, hv, b_count) &&
      checked_mul(q_channels, w, q_weight_count) &&
      checked_mul(v_channels, w, v_weight_count) && checked_mul(hv, kd, dt_count);
  if (!sizes_ok || !q || !k || !v || !gate || !beta || !out || !weights.q_conv ||
      !weights.k_conv || !weights.v_conv || !weights.a_log ||
      !weights.dt_bias || !finite_values(q, q_count) ||
      !finite_values(k, q_count) || !finite_values(v, v_count) ||
      !finite_values(gate, g_count) || !finite_values(beta, b_count) ||
      !finite_values(weights.q_conv, q_weight_count) ||
      !finite_values(weights.k_conv, q_weight_count) ||
      !finite_values(weights.v_conv, v_weight_count) ||
      !finite_values(weights.a_log, hv) ||
      !finite_values(weights.dt_bias, dt_count) ||
      (weights.q_bias && !finite_values(weights.q_bias, q_channels)) ||
      (weights.k_bias && !finite_values(weights.k_bias, q_channels)) ||
      (weights.v_bias && !finite_values(weights.v_bias, v_channels)) ||
      !finite_vector(state.recurrent_) || !finite_vector(state.q_conv_state_) ||
      !finite_vector(state.k_conv_state_) || !finite_vector(state.v_conv_state_)) {
    error = "KDA inputs, weights, and state must be finite and non-null";
    return false;
  }
  for (size_t i = 0; i < b_count; ++i) {
    if (beta[i] < 0.0f || beta[i] > 1.0f) {
      error = "KDA beta must be post-sigmoid in [0, 1]";
      return false;
    }
  }

  std::vector<float> recurrent = state.recurrent_;
  std::vector<float> q_history = state.q_conv_state_;
  std::vector<float> k_history = state.k_conv_state_;
  std::vector<float> v_history = state.v_conv_state_;
  std::vector<float> q_conv, k_conv, v_conv, q_norm, k_norm;
  std::vector<float> result(v_count);
  const float q_scale = 1.0f / std::sqrt(float(c.key_dim));
  if (!std::isfinite(q_scale)) {
    error = "KDA query scale is non-finite";
    return false;
  }
  for (int64_t token = 0; token < n_tokens; ++token) {
    const float *q_token = q + size_t(token) * q_channels;
    const float *k_token = k + size_t(token) * q_channels;
    const float *v_token = v + size_t(token) * v_channels;
    if (!conv_token(q_token, weights.q_conv, weights.q_bias, int64_t(q_channels),
                    c.conv_kernel, q_history, q_conv) ||
        !conv_token(k_token, weights.k_conv, weights.k_bias, int64_t(q_channels),
                    c.conv_kernel, k_history, k_conv) ||
        !conv_token(v_token, weights.v_conv, weights.v_bias, int64_t(v_channels),
                    c.conv_kernel, v_history, v_conv)) {
      error = "KDA short convolution produced a non-finite value";
      return false;
    }
    for (size_t value_head = 0; value_head < hv; ++value_head) {
      const float *q_head = q_conv.data() +
                            (value_head / (hv / h)) * kd;
      const float *k_head = k_conv.data() +
                            (value_head / (hv / h)) * kd;
      if (!l2_normalize(q_head, c.key_dim, c.qk_epsilon, q_norm) ||
          !l2_normalize(k_head, c.key_dim, c.qk_epsilon, k_norm)) {
        error = "KDA q/k normalization produced a non-finite value";
        return false;
      }
      const float *gate_head = gate + (size_t(token) * hv + value_head) * kd;
      const float *dt_head = weights.dt_bias + value_head * kd;
      const float beta_value = beta[size_t(token) * hv + value_head];
      float *state_head = recurrent.data() + value_head * kd * vd;
      const float *value_head_input = v_conv.data() + value_head * vd;
      const float a = std::exp(weights.a_log[value_head]);
      if (!std::isfinite(a)) {
        error = "KDA gate scale is non-finite";
        return false;
      }
      for (size_t key_index = 0; key_index < kd; ++key_index) {
        float decay = 0.0f;
        if (c.safe_gate) {
          const float z = a * (gate_head[key_index] + dt_head[key_index]);
          if (!std::isfinite(z)) {
            error = "KDA gate logit is non-finite";
            return false;
          }
          decay = c.gate_lower_bound * sigmoid(z);
        } else {
          const float gate_input = gate_head[key_index] + dt_head[key_index];
          if (!std::isfinite(gate_input)) {
            error = "KDA gate input is non-finite";
            return false;
          }
          decay = -a * softplus(gate_input);
        }
        if (!std::isfinite(decay)) {
          error = "KDA decay logit is non-finite";
          return false;
        }
        const float factor = std::exp(decay);
        if (!std::isfinite(factor)) {
          error = "KDA decay is non-finite";
          return false;
        }
        float *row = state_head + key_index * vd;
        for (size_t value_index = 0; value_index < vd; ++value_index) {
          row[value_index] *= factor;
          if (!std::isfinite(row[value_index])) {
            error = "KDA recurrent decay is non-finite";
            return false;
          }
        }
      }
      std::vector<float> delta(vd);
      for (size_t value_index = 0; value_index < vd; ++value_index) {
        float dot = 0.0f;
        for (size_t key_index = 0; key_index < kd; ++key_index) {
          const float term = state_head[key_index * vd + value_index] *
                             k_norm[key_index];
          if (!std::isfinite(term) || !std::isfinite(dot += term)) {
            error = "KDA state-key product is non-finite";
            return false;
          }
        }
        delta[value_index] = (value_head_input[value_index] - dot) * beta_value;
        if (!std::isfinite(delta[value_index])) {
          error = "KDA delta is non-finite";
          return false;
        }
      }
      for (size_t key_index = 0; key_index < kd; ++key_index) {
        float *row = state_head + key_index * vd;
        for (size_t value_index = 0; value_index < vd; ++value_index) {
          row[value_index] += k_norm[key_index] * delta[value_index];
          if (!std::isfinite(row[value_index])) {
            error = "KDA recurrent update is non-finite";
            return false;
          }
        }
      }
      float *output_head = result.data() +
                           (size_t(token) * hv + value_head) * vd;
      for (size_t value_index = 0; value_index < vd; ++value_index) {
        float sum = 0.0f;
        for (size_t key_index = 0; key_index < kd; ++key_index) {
          const float term = state_head[key_index * vd + value_index] *
                             (q_norm[key_index] * q_scale);
          if (!std::isfinite(term) || !std::isfinite(sum += term)) {
            error = "KDA output reduction is non-finite";
            return false;
          }
        }
        output_head[value_index] = sum;
      }
    }
  }
  state.recurrent_.swap(recurrent);
  state.q_conv_state_.swap(q_history);
  state.k_conv_state_.swap(k_history);
  state.v_conv_state_.swap(v_history);
  std::copy(result.begin(), result.end(), out);
  return true;
}

} // namespace strata::ling
