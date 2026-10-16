// Copyright (c) AI Identity Incorported <james@aiii.id>.
#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
namespace strata::torch_router {
class Rms {
 public:
  explicit Rms(const char *device);
  ~Rms();
  Rms(const Rms&) = delete;
  bool normalize(const float *x, const float *weight, std::size_t rows,
                 std::size_t dim, float eps, std::vector<float> &out,
                 std::string &error);
 private:
  struct Impl; std::unique_ptr<Impl> impl_;
};
}
