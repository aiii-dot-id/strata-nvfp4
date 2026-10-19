// Copyright (c) AI Identity Incorported <james@aiii.id>.
#include "attention.hpp"
#include "pointwise.hpp"
#include "router.hpp"
#include "rms.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using strata::torch_router::Router;
static void require_empty(const std::vector<std::size_t>& ids, const std::vector<float>& w,
                          const std::vector<float>& s, const std::vector<float>& c) {
  assert(ids.empty()); assert(w.empty()); assert(s.empty()); assert(c.empty());
}
int main() {
  std::vector<std::size_t> ids{99}; std::vector<float> weights{1.0f}, scores{1.0f}, corrected{1.0f};
  std::string error;
  Router bad_groups(8, 0, 1, 1, 1.0f, "cpu"); float zero[8] = {};
  assert(!bad_groups.route(zero, zero, 8, ids, weights, scores, corrected, error));
  require_empty(ids, weights, scores, corrected);
  Router r(8, 2, 1, 2, 1.0f, "cpu"); float logits[8] = {0,1,2,3,4,5,6,7}; float bias[8] = {};
  assert(r.route(logits, bias, 8, ids, weights, scores, corrected, error));
  assert(ids.size() == 2 && weights.size() == 2);
  for (float x : weights) assert(std::isfinite(x));
  for (float x : scores) assert(std::isfinite(x));
  for (float x : corrected) assert(std::isfinite(x));
  float bad[8] = {}; bad[3] = std::numeric_limits<float>::quiet_NaN();
  assert(!r.route(bad, bias, 8, ids, weights, scores, corrected, error));
  require_empty(ids, weights, scores, corrected);
  bool bad_device = false;
  try { strata::torch_router::Rms bad("bogus"); } catch (const std::invalid_argument &) { bad_device = true; }
  assert(bad_device);
  bool null_device = false;
  try { strata::torch_router::Rms bad(nullptr); } catch (const std::invalid_argument &) { null_device = true; }
  assert(null_device);
  strata::torch_router::Rms rms("cpu");
  float rv[4] = {1, 2, 3, 4}; float rw[4] = {1, 1, 1, 1};
  std::vector<float> normed;
  assert(rms.normalize(rv, rw, 1, 4, 1e-6f, normed, error));
  assert(normed.size() == 4); for (float v : normed) assert(std::isfinite(v));
  float bad_rv[4] = {1, 2, std::numeric_limits<float>::quiet_NaN(), 4};
  assert(!rms.normalize(bad_rv, rw, 1, 4, 1e-6f, normed, error));
  assert(normed.empty());
  assert(!rms.normalize(rv, rw, 0, 4, 1e-6f, normed, error));
  assert(normed.empty());
  // CPU control for the source-shaped eager attention expression.  A single
  // key makes softmax exact while the two KV heads prove GQA replication.
  strata::torch_router::Attention attention("cpu");
  const float query[16] = {1, 0, 0, 0, 0, 1, 0, 0,
                            0, 0, 1, 0, 0, 0, 0, 1};
  const float keys[8] = {1, 0, 0, 0, 0, 1, 0, 0};
  const float values[8] = {1, 2, 3, 4, -1, -2, -3, -4};
  std::vector<float> context, attention_scores, probabilities;
  assert(attention.forward(query, keys, values, 4, 2, 4, 1, context,
                           attention_scores, probabilities, error));
  assert(context.size() == 16 && attention_scores.size() == 4 &&
         probabilities.size() == 4);
  for (std::size_t h = 0; h < 4; ++h) {
    const float *expected = values + (h / 2) * 4;
    for (std::size_t d = 0; d < 4; ++d)
      assert(context[h * 4 + d] == expected[d]);
    assert(probabilities[h] == 1.0f);
  }
  assert(!attention.forward(nullptr, keys, values, 4, 2, 4, 1, context,
                            attention_scores, probabilities, error));
  assert(context.empty() && attention_scores.empty() && probabilities.empty());
  strata::torch_router::Pointwise pointwise("cpu");
  const float gate[] = {0.0f, 1.0f, -1.0f, 2.0f};
  const float up[] = {2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> swiglu;
  assert(pointwise.swiglu(gate, up, 4, swiglu, error));
  assert(swiglu.size() == 4 && swiglu[0] == 0.0f);
  const float expert_output[] = {1, 2, 3, 4, -1, -2, -3, -4};
  const float expert_weights[] = {1.0f, 0.0f};
  std::vector<float> reduced;
  assert(pointwise.weighted_sum(expert_output, expert_weights, 2, 4, reduced, error));
  assert(reduced == std::vector<float>({1, 2, 3, 4}));
  assert(!pointwise.weighted_sum(expert_output, expert_weights, 0, 4, reduced, error));
  assert(reduced.empty());
  return 0;
}
