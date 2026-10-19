// Copyright (c) AI Identity Incorported <james@aiii.id>.
#include "pointwise.hpp"

#include <ATen/Context.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::torch_router {

struct Pointwise::Impl {
  c10::Device device;
  explicit Impl(const char *name) : device(c10::kCPU) {
    if (!name)
      throw std::invalid_argument("pointwise device is null");
    const std::string value(name);
    if (value == "cuda") {
      device = c10::kCUDA;
      at::globalContext().setAllowTF32CuBLAS(false);
      at::globalContext().setAllowTF32CuDNN(false);
    } else if (value != "cpu") {
      throw std::invalid_argument("pointwise device must be cpu or cuda");
    }
  }
};

Pointwise::Pointwise(const char *device) : impl_(std::make_unique<Impl>(device)) {}
Pointwise::~Pointwise() = default;

namespace {
void require_finite(const float *values, std::size_t count, const char *name) {
  if (!values)
    throw std::invalid_argument(std::string("null pointwise ") + name);
  for (std::size_t i = 0; i < count; ++i)
    if (!std::isfinite(values[i]))
      throw std::invalid_argument(std::string("nonfinite pointwise ") + name);
}
} // namespace

bool Pointwise::swiglu(const float *gate, const float *up, std::size_t count,
                       std::vector<float> &out, std::string &error) {
  out.clear();
  try {
    if (!count)
      throw std::invalid_argument("invalid pointwise SwiGLU size");
    require_finite(gate, count, "gate");
    require_finite(up, count, "up");
    const auto options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto g = torch::from_blob(const_cast<float *>(gate), {static_cast<long>(count)}, options)
                       .clone().to(impl_->device);
    const auto u = torch::from_blob(const_cast<float *>(up), {static_cast<long>(count)}, options)
                       .clone().to(impl_->device);
    const auto result = (torch::silu(g) * u).to(torch::kCPU).contiguous();
    const auto data = result.data_ptr<float>();
    std::vector<float> next(data, data + count);
    require_finite(next.data(), next.size(), "SwiGLU output");
    out.swap(next);
    return true;
  } catch (const std::exception &exception) {
    error = exception.what();
    return false;
  }
}

bool Pointwise::weighted_sum(const float *expert_output, const float *weights,
                             std::size_t experts, std::size_t hidden,
                             std::vector<float> &out, std::string &error) {
  out.clear();
  try {
    if (!experts || !hidden || experts > std::numeric_limits<std::size_t>::max() / hidden)
      throw std::invalid_argument("invalid pointwise weighted-sum geometry");
    require_finite(expert_output, experts * hidden, "expert output");
    require_finite(weights, experts, "weights");
    const auto options = torch::TensorOptions().dtype(torch::kFloat32);
    // [1,8,hidden] * [1,8,1], then reduce the expert dimension exactly once.
    const auto y = torch::from_blob(const_cast<float *>(expert_output),
                                    {1, static_cast<long>(experts), static_cast<long>(hidden)}, options)
                       .clone().to(impl_->device);
    const auto w = torch::from_blob(const_cast<float *>(weights),
                                    {1, static_cast<long>(experts), 1}, options)
                       .clone().to(impl_->device);
    const auto result = (y * w).sum(1).to(torch::kCPU).contiguous();
    const auto data = result.data_ptr<float>();
    std::vector<float> next(data, data + hidden);
    require_finite(next.data(), next.size(), "weighted-sum output");
    out.swap(next);
    return true;
  } catch (const std::exception &exception) {
    error = exception.what();
    return false;
  }
}

} // namespace strata::torch_router
