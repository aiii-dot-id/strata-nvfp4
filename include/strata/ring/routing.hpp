#pragma once

#include <cstddef>
#include <vector>

namespace strata::ring {

struct RouterConfig {
  std::size_t num_experts;
  std::size_t n_group;
  std::size_t topk_group;
  std::size_t top_k;
  float routed_scaling_factor;
};

struct RoutingResult {
  std::vector<std::size_t> expert_ids;
  std::vector<float> weights;
};

// Route one token from already-computed FP32 router logits and expert bias.
// Bias participates only in group/expert selection. Returned weights are the
// bias-free sigmoid scores, normalized (top_k > 1), then scaled.
// Ties are deterministic: higher score first, then lower expert id. This
// explicit tie rule replaces torch.topk's unspecified equal-value ordering.
RoutingResult route(const float *logits, const float *expert_bias,
                    std::size_t count, const RouterConfig &config);

} // namespace strata::ring
