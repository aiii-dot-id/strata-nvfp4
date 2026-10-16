#include "strata/ring/routing.hpp"

#include <cassert>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using strata::ring::RouterConfig;

static void expect_throw(const float *l, const float *b, std::size_t n,
                         RouterConfig c) {
  bool threw = false;
  try {
    (void)strata::ring::route(l, b, n, c);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);
}

int main() {
  const RouterConfig c{8, 2, 1, 2, 2.5f};
  const float bias[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const float logits[8] = {-4, -3, -2, -1, 1, 2, 3, 4};
  auto normal = strata::ring::route(logits, bias, 8, c);
  assert(normal.expert_ids[0] == 7 && normal.expert_ids[1] == 6);
  assert(std::fabs(normal.weights[0] + normal.weights[1] - 2.5f) < 1e-6f);
  const float negative[8] = {-8, -7, -6, -5, -4, -3, -2, -1};
  auto neg = strata::ring::route(negative, bias, 8, c);
  assert(neg.expert_ids[0] == 7 && neg.expert_ids[1] == 6);
  const float tie[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  auto tied = strata::ring::route(tie, bias, 8, c);
  assert(tied.expert_ids[0] == 0 && tied.expert_ids[1] == 1);
  const float biased[8] = {0, 0, 0, 0, 100, 0, 0, 0};
  auto route_bias = strata::ring::route(tie, biased, 8, c);
  assert(route_bias.expert_ids[0] == 4);
  RouterConfig bad{8, 3, 1, 2, 2.5f};
  expect_throw(logits, bias, 8, bad);
  bad = {8, 2, 3, 2, 2.5f};
  expect_throw(logits, bias, 8, bad);
  bad = {8, 2, 1, 9, 2.5f};
  expect_throw(logits, bias, 8, bad);
  bad = {8, 2, 1, 5, 2.5f};
  expect_throw(logits, bias, 8, bad);
  float nan[8] = {};
  nan[0] = NAN;
  expect_throw(nan, bias, 8, c);
  float huge_bias[8] = {};
  huge_bias[0] = FLT_MAX;
  huge_bias[1] = FLT_MAX;
  expect_throw(logits, huge_bias, 8, c);
  std::cout << "ring_routing_test: ok\n";
}
