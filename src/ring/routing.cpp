#include "strata/ring/routing.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strata::ring {
namespace {

float sigmoid(float x) {
  if (x >= 0.0f) {
    const float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(x);
  return z / (1.0f + z);
}

void validate(const float *logits, const float *bias, std::size_t count,
              const RouterConfig &c) {
  if (logits == nullptr || bias == nullptr)
    throw std::invalid_argument("router inputs must be non-null");
  if (c.num_experts == 0 || c.n_group == 0 || c.num_experts % c.n_group != 0)
    throw std::invalid_argument(
        "num_experts must be divisible by nonzero n_group");
  if (c.topk_group == 0 || c.topk_group > c.n_group || c.top_k == 0 ||
      c.top_k > c.num_experts)
    throw std::invalid_argument("top-k geometry out of range");
  if (c.top_k > c.topk_group * (c.num_experts / c.n_group))
    throw std::invalid_argument("top-k exceeds selected-group capacity");
  if (!std::isfinite(c.routed_scaling_factor))
    throw std::invalid_argument("nonfinite routed scaling");
  if (count != c.num_experts)
    throw std::invalid_argument("router input count mismatch");
  for (std::size_t i = 0; i < count; ++i)
    if (!std::isfinite(logits[i]) || !std::isfinite(bias[i]))
      throw std::invalid_argument("nonfinite router input");
}

} // namespace

RoutingResult route(const float *logits, const float *expert_bias,
                    std::size_t count, const RouterConfig &c) {
  validate(logits, expert_bias, count, c);
  const std::size_t per_group = c.num_experts / c.n_group;
  std::vector<float> scores(count), routed(count);
  for (std::size_t i = 0; i < count; ++i) {
    scores[i] = sigmoid(logits[i]);
    routed[i] = scores[i] + expert_bias[i];
    if (!std::isfinite(routed[i])) throw std::invalid_argument("router score overflow");
  }
  std::vector<std::size_t> groups(c.n_group);
  for (std::size_t g = 0; g < c.n_group; ++g)
    groups[g] = g;
  std::sort(groups.begin(), groups.end(), [&](std::size_t a, std::size_t b) {
    auto score = [&](std::size_t group) {
      float first = -std::numeric_limits<float>::infinity(), second = first;
      for (std::size_t i = 0; i < per_group; ++i) {
        const float x = routed[group * per_group + i];
        if (x >= first) {
          second = first;
          first = x;
        } else if (x > second)
          second = x;
      }
      const float total = first + second;
      if (!std::isfinite(total)) throw std::invalid_argument("router group score overflow");
      return total;
    };
    const float av = score(a), bv = score(b);
    return av != bv ? av > bv : a < b;
  });
  groups.resize(c.topk_group);
  std::vector<std::size_t> candidates;
  for (std::size_t g : groups)
    for (std::size_t i = 0; i < per_group; ++i)
      candidates.push_back(g * per_group + i);
  std::sort(candidates.begin(), candidates.end(),
            [&](std::size_t a, std::size_t b) {
              return routed[a] != routed[b] ? routed[a] > routed[b] : a < b;
            });
  candidates.resize(c.top_k);
  RoutingResult out;
  out.expert_ids = candidates;
  out.weights.resize(c.top_k);
  float sum = 0.0f;
  for (std::size_t i = 0; i < c.top_k; ++i) {
    out.weights[i] = scores[candidates[i]];
    sum += out.weights[i];
  }
  if (c.top_k > 1)
    for (float &weight : out.weights)
      weight /= (sum + 1e-20f);
  for (float &weight : out.weights)
    weight *= c.routed_scaling_factor;
  return out;
}

} // namespace strata::ring
