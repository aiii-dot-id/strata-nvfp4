// Copyright (c) AI Identity Incorported <james@aiii.id>.
#pragma once
#include <memory>
#include <strata/ring/attention.hpp>

namespace strata::torch_router {

// Executes the original eager Ring attention expression with ATen.  It is
// intentionally a diagnostic backend: callers must opt in and provide the
// already normalized and RoPE-rotated FP32 inputs.
class Attention final : public ring::AttentionBackend {
public:
  explicit Attention(const char *device);
  ~Attention() override;
  Attention(const Attention &) = delete;
  Attention &operator=(const Attention &) = delete;

  bool forward(const float *query, const float *keys, const float *values,
               int64_t heads, int64_t kv_heads, int64_t dim,
               int64_t key_count, std::vector<float> &context,
               std::vector<float> &scores,
               std::vector<float> &probabilities,
               std::string &error) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace strata::torch_router
