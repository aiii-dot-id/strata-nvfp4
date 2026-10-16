// AI Identity Incorported <james@aiii.id>
#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace strata::ring {

// Sum squared FP32 terms through a fixed balanced tree.  The products and
// additions intentionally remain FP32 so the reduction is deterministic and
// matches the arithmetic type used by the native path, while avoiding the
// order-dependent loss of small terms after a large partial sum.
inline float balanced_sum_squares(const float *values, std::size_t count) {
  if (!values || count == 0)
    throw std::invalid_argument("invalid squared-sum input");

  const auto reduce = [](const auto &self, const float *p,
                         std::size_t n) -> float {
    if (n == 1) {
      const float f = p[0];
      if (!std::isfinite(f))
        throw std::runtime_error("RMS normalization nonfinite input");
      const float term = f * f;
      if (!std::isfinite(term))
        throw std::runtime_error("RMS normalization overflow/nonfinite input");
      return term;
    }
    const std::size_t left_count = n / 2;
    const float left = self(self, p, left_count);
    const float right = self(self, p + left_count, n - left_count);
    const float sum = left + right;
    if (!std::isfinite(sum))
      throw std::runtime_error("RMS normalization overflow/nonfinite input");
    return sum;
  };
  return reduce(reduce, values, count);
}

// RMS normalization keeps the FP32 product/reduction arithmetic explicit and
// fails closed on overflow or non-finite input.
inline std::vector<float> rms_norm(const std::vector<float> &x,
                                   const std::vector<float> &weights,
                                   float epsilon = 1e-6f) {
  if (x.empty() || x.size() != weights.size() || !std::isfinite(epsilon) ||
      epsilon <= 0)
    throw std::invalid_argument("invalid RMS normalization geometry/epsilon");
  const float sum = balanced_sum_squares(x.data(), x.size());
  const float variance = sum / float(x.size()) + epsilon;
  const float inverse = 1 / std::sqrt(variance);
  if (!std::isfinite(variance) || !std::isfinite(inverse) || inverse <= 0)
    throw std::runtime_error("invalid RMS normalization scale");
  std::vector<float> output(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    output[i] = x[i] * inverse * weights[i];
    if (!std::isfinite(weights[i]) || !std::isfinite(output[i]))
      throw std::runtime_error("nonfinite RMS normalization output");
  }
  return output;
}
} // namespace strata::ring
