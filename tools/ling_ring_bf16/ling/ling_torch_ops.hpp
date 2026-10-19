// Copyright (c) AI Identity Incorported <james@aiii.id>.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace strata::ling_torch {

// Isolated, opt-in ATen pointwise/reduction adapter. Matrix multiplication and
// the default Ling path remain independent of Torch.
class Ops {
 public:
  explicit Ops(const char *device);
  ~Ops();
  Ops(const Ops &) = delete;
  bool sigmoid(const std::vector<float> &, std::vector<float> &, std::string &) const;
  bool rms_norm(const std::vector<float> &, const std::vector<float> &,
                std::vector<float> &, std::string &) const;
  bool swiglu(const std::vector<float> &, const std::vector<float> &,
              std::vector<float> &, std::string &) const;
  bool kda_output_norm_gate(const std::vector<float> &, const std::vector<float> &,
                            const std::vector<float> &, std::size_t heads,
                            std::size_t width, std::vector<float> &,
                            std::string &) const;
  bool weighted_sum(const std::vector<std::vector<float>> &,
                    const std::vector<float> &, std::vector<float> &,
                    std::string &) const;
  bool route(const std::vector<float> &, const std::vector<float> &,
             std::vector<std::size_t> &, std::vector<float> &,
             std::vector<float> &, std::vector<float> &, std::string &) const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace strata::ling_torch
