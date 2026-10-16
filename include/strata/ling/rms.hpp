// AI Identity Incorported <james@aiii.id>
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "strata/ring/numerics.hpp"

namespace strata::ling {
enum class RmsMode { Native, TorchCudaOrder };

// This models the source-defined CUDA mean path used by PyTorch's
// Reduce.cuh/ReduceMomentKernel.cu for float RMS inputs: four-value vector
// loads, geometry-derived lanes, then the full descending block tree (including offset
// 32).  It is kept as a CPU-order model so the opt-in Ling path does not
// depend on a device reduction implementation or alter KDA's fused
// normalization.
inline std::vector<float> rms_norm(const std::vector<float> &x,
                                   const std::vector<float> &weights,
                                   RmsMode mode, float epsilon = 1e-6f) {
  if (x.empty() || x.size() != weights.size() || !std::isfinite(epsilon) || epsilon <= 0)
    throw std::invalid_argument("invalid Ling RMS normalization geometry/epsilon");
  if (mode == RmsMode::Native)
    return strata::ring::rms_norm(x, weights, epsilon);
  constexpr std::size_t kMaxThreads = 512, kVector = 4;
  if (x.size() % kVector != 0)
    throw std::invalid_argument("Torch CUDA RMS normalization requires four-element vectors");
  const std::size_t vectors = x.size() / kVector;
  // Reduce.cuh chooses the largest power-of-two block width no greater than
  // the vectorized input count (up to the float reduction limit of 512).
  if (vectors == 0 || vectors > kMaxThreads)
    throw std::invalid_argument("Torch CUDA RMS normalization geometry exceeds one reduction block");
  std::size_t threads = 1;
  while (threads * 2 <= vectors)
    threads *= 2;
  std::array<std::array<float, kVector>, kMaxThreads> lane{};
  for (std::size_t thread = 0; thread < threads; ++thread) {
    for (std::size_t vector = thread; vector < vectors; vector += threads) {
      for (std::size_t j = 0; j < kVector; ++j) {
        const float value = x[vector * kVector + j];
        if (!std::isfinite(value))
          throw std::runtime_error("Ling RMS normalization nonfinite input");
        lane[thread][j] += value * value;
        if (!std::isfinite(lane[thread][j]))
          throw std::runtime_error("Ling RMS normalization overflow/nonfinite input");
      }
    }
    for (std::size_t j = 1; j < kVector; ++j) {
      lane[thread][0] += lane[thread][j];
      if (!std::isfinite(lane[thread][0]))
        throw std::runtime_error("Ling RMS normalization overflow/nonfinite input");
    }
  }
  for (std::size_t offset = threads / 2; offset > 0; offset >>= 1)
    for (std::size_t thread = 0; thread < offset; ++thread)
      lane[thread][0] = lane[thread][0] + lane[thread + offset][0];
  const float sum = lane[0][0];
  const float factor = 1.0f / float(x.size());
  const float variance = sum * factor + epsilon;
  const float inverse = 1.0f / std::sqrt(variance);
  if (!std::isfinite(variance) || !std::isfinite(inverse) || inverse <= 0)
    throw std::runtime_error("Ling RMS normalization invalid scale");
  std::vector<float> output(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (!std::isfinite(weights[i]))
      throw std::runtime_error("Ling RMS normalization nonfinite weight");
    // Matches PyTorch's vectorized kernel: gamma * (rstd * x).
    output[i] = weights[i] * (inverse * x[i]);
    if (!std::isfinite(output[i]))
      throw std::runtime_error("Ling RMS normalization nonfinite output");
  }
  return output;
}
} // namespace strata::ling
