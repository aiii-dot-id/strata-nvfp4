// Copyright (c) AI Identity Incorported <james@aiii.id>.
#include "attention.hpp"

#include <ATen/Context.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::torch_router {

struct Attention::Impl {
  c10::Device device;
  explicit Impl(const char *name) : device(c10::kCPU) {
    if (!name)
      throw std::invalid_argument("attention device is null");
    const std::string value(name);
    if (value == "cuda") {
      device = c10::kCUDA;
      // The captured reference explicitly disables TF32 before its matmuls.
      at::globalContext().setAllowTF32CuBLAS(false);
      at::globalContext().setAllowTF32CuDNN(false);
    } else if (value != "cpu") {
      throw std::invalid_argument("attention device must be cpu or cuda");
    }
  }
};

Attention::Attention(const char *device) : impl_(std::make_unique<Impl>(device)) {}
Attention::~Attention() = default;

bool Attention::forward(const float *query, const float *keys,
                        const float *values, int64_t heads,
                        int64_t kv_heads, int64_t dim, int64_t key_count,
                        std::vector<float> &context,
                        std::vector<float> &scores,
                        std::vector<float> &probabilities,
                        std::string &error) {
  context.clear();
  scores.clear();
  probabilities.clear();
  try {
    if (!query || !keys || !values || heads <= 0 || kv_heads <= 0 ||
        dim <= 0 || key_count <= 0 || heads % kv_heads)
      throw std::invalid_argument("invalid Torch attention geometry");
    const auto h = static_cast<std::size_t>(heads);
    const auto kh = static_cast<std::size_t>(kv_heads);
    const auto d = static_cast<std::size_t>(dim);
    const auto n = static_cast<std::size_t>(key_count);
    if (h > std::numeric_limits<std::size_t>::max() / d ||
        n > std::numeric_limits<std::size_t>::max() / kh ||
        n * kh > std::numeric_limits<std::size_t>::max() / d)
      throw std::invalid_argument("Torch attention geometry overflows");
    for (std::size_t i = 0; i < h * d; ++i)
      if (!std::isfinite(query[i]))
        throw std::invalid_argument("nonfinite Torch attention query");
    for (std::size_t i = 0; i < n * kh * d; ++i)
      if (!std::isfinite(keys[i]) || !std::isfinite(values[i]))
        throw std::invalid_argument("nonfinite Torch attention cache");

    const auto options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto q = torch::from_blob(const_cast<float *>(query), {heads, dim}, options)
                       .clone().to(impl_->device);
    const auto k = torch::from_blob(const_cast<float *>(keys),
                                    {key_count, kv_heads, dim}, options)
                       .clone().to(impl_->device);
    const auto v = torch::from_blob(const_cast<float *>(values),
                                    {key_count, kv_heads, dim}, options)
                       .clone().to(impl_->device);
    const auto group = heads / kv_heads;
    // This is the source expression used by the retained original replay:
    // [H,1,D] @ [H,D,K], FP32 softmax, then [H,1,K] @ [H,K,D].
    const auto repeated_k = k.repeat_interleave(group, 1).permute({1, 0, 2});
    const auto repeated_v = v.repeat_interleave(group, 1).permute({1, 0, 2});
    const auto result_scores = torch::matmul(q.unsqueeze(1),
                                              repeated_k.transpose(1, 2))
                                   .squeeze(1) /
                               std::sqrt(static_cast<double>(dim));
    const auto result_probabilities =
        torch::softmax(result_scores, -1, torch::kFloat32).to(result_scores.dtype());
    const auto result_context = torch::matmul(result_probabilities.unsqueeze(1),
                                               repeated_v).squeeze(1);
    if (!torch::isfinite(result_scores).all().item<bool>() ||
        !torch::isfinite(result_probabilities).all().item<bool>() ||
        !torch::isfinite(result_context).all().item<bool>())
      throw std::runtime_error("nonfinite Torch attention output");

    const auto score_cpu = result_scores.to(torch::kCPU).contiguous();
    const auto probability_cpu = result_probabilities.to(torch::kCPU).contiguous();
    const auto context_cpu = result_context.to(torch::kCPU).contiguous();
    const auto score_data = score_cpu.data_ptr<float>();
    const auto probability_data = probability_cpu.data_ptr<float>();
    const auto context_data = context_cpu.data_ptr<float>();
    std::vector<float> next_scores(score_data, score_data + h * n);
    std::vector<float> next_probabilities(probability_data,
                                          probability_data + h * n);
    std::vector<float> next_context(context_data, context_data + h * d);
    scores.swap(next_scores);
    probabilities.swap(next_probabilities);
    context.swap(next_context);
    return true;
  } catch (const std::exception &exception) {
    error = exception.what();
    return false;
  }
}

} // namespace strata::torch_router
