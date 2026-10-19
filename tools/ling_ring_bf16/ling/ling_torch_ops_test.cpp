#include "ling_torch_ops.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {
void near(float actual, float expected) { assert(std::fabs(actual - expected) < 1e-5f); }
}

int main() {
  bool bad_device = false;
  try { strata::ling_torch::Ops invalid("bogus"); } catch (const std::invalid_argument &) { bad_device = true; }
  assert(bad_device);
  strata::ling_torch::Ops ops("cpu");
  std::string error;
  std::vector<float> output{99.0f};
  assert(ops.sigmoid({-20.0f, 0.0f, 20.0f}, output, error));
  assert(output.size() == 3); assert(output[0] < 1e-8f); near(output[1], 0.5f); assert(output[2] > 0.9999999f);
  assert(ops.rms_norm({3.0f, 4.0f}, {2.0f, 1.0f}, output, error));
  const float inverse = 1.0f / std::sqrt(12.5f + 1e-6f);
  near(output[0], 2.0f * inverse * 3.0f); near(output[1], inverse * 4.0f);
  assert(ops.swiglu({0.0f, -2.0f, 2.0f}, {9.0f, 1.0f, -1.0f}, output, error));
  near(output[0], 0.0f); assert(output[1] < 0.0f); assert(output[2] < 0.0f);
  std::vector<std::vector<float>> rows(8, std::vector<float>(2)); std::vector<float> weights(8);
  for (std::size_t i = 0; i < 8; ++i) { rows[i][0] = float(i); rows[i][1] = float(2 * i); weights[i] = float(i + 1); }
  assert(ops.weighted_sum(rows, weights, output, error)); near(output[0], 168.0f); near(output[1], 336.0f);
  std::vector<float> logits(128), bias(128, 0.0f), scores, corrected, route_weights; std::vector<std::size_t> ids;
  for (std::size_t i = 0; i < logits.size(); ++i) logits[i] = float(i) * 0.1f;
  assert(ops.route(logits, bias, ids, route_weights, scores, corrected, error));
  assert(ids.size() == 8 && route_weights.size() == 8 && scores.size() == 128 && corrected.size() == 128);
  for (std::size_t i = 0; i < 8; ++i) assert(ids[i] == 127 - i);
  assert(!ops.rms_norm({1.0f}, {1.0f, 2.0f}, output, error)); assert(output.empty());
  assert(!ops.swiglu({1.0f}, {}, output, error)); assert(output.empty());
  assert(!ops.weighted_sum({{1.0f}, {2.0f, 3.0f}}, {1.0f, 1.0f}, output, error)); assert(output.empty());
  logits[0] = std::numeric_limits<float>::quiet_NaN();
  assert(!ops.route(logits, bias, ids, route_weights, scores, corrected, error));
  assert(ids.empty() && route_weights.empty() && scores.empty() && corrected.empty());
}
