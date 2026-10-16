// AI Identity Incorported <james@aiii.id>
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

namespace strata::ring {
// Preserve the FP32 evaluation order; fail on overflow rather than silently
// turning an infinite variance into a finite, all-zero normalized state.
inline std::vector<float> rms_norm(const std::vector<float> &x,
                                   const std::vector<float> &weights,
                                   float epsilon = 1e-6f) {
  if (x.empty() || x.size() != weights.size() || !std::isfinite(epsilon) ||
      epsilon <= 0)
    throw std::invalid_argument("invalid RMS normalization geometry/epsilon");
  float sum = 0;
  for (float f : x) {
    const float term = f * f;
    sum += term;
    if (!std::isfinite(f) || !std::isfinite(term) || !std::isfinite(sum))
      throw std::runtime_error("RMS normalization overflow/nonfinite input");
  }
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
