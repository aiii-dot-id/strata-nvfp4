// Copyright (c) AI Identity Incorported <james@aiii.id>.
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
  return 0;
}
