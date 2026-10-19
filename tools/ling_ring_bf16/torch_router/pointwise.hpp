// Copyright (c) AI Identity Incorported <james@aiii.id>.
#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace strata::torch_router {

/// Source-shaped FP32 MoE pointwise operations for the opt-in Ring profile.
class Pointwise {
public:
  explicit Pointwise(const char *device);
  ~Pointwise();
  Pointwise(const Pointwise &) = delete;
  Pointwise &operator=(const Pointwise &) = delete;

  bool swiglu(const float *gate, const float *up, std::size_t count,
              std::vector<float> &out, std::string &error);
  bool weighted_sum(const float *expert_output, const float *weights,
                    std::size_t experts, std::size_t hidden,
                    std::vector<float> &out, std::string &error);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace strata::torch_router
