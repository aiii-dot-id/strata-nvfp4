// AI Identity Incorported <james@aiii.id>
#include <cassert>
#include <iostream>
#include <limits>
#include <strata/ring/generation.hpp>
template <class F> void rejects(F f) {
  bool caught = false;
  try {
    f();
  } catch (const std::exception &) {
    caught = true;
  }
  assert(caught);
}
int main() {
  using namespace strata::ring;
  GreedyGeneration g({1, 2}, 3, {5, 6}, 8, 16);
  assert(!g.sampling_position(0) && g.sampling_position(1));
  std::vector<float> scores(8, 0);
  scores[3] = 2;
  assert(g.sample(1, scores) == 3 &&
         g.evaluated() == std::vector<int>({1, 2, 3}));
  assert(g.sample(2, scores) == 3 && g.evaluated().size() == 4);
  assert(g.sample(3, scores) == 3 && g.finish_reason() == "length" &&
         g.evaluated().size() == 4);
  rejects([&] { g.sample(3, scores); });
  GreedyGeneration eos({1}, 4, {5, 6}, 8, 16);
  scores[6] = 3;
  assert(eos.sample(0, scores) == 6 && eos.finish_reason() == "eos" &&
         eos.evaluated().size() == 1);
  assert(eos.generated() == std::vector<int>({6}));
  GreedyGeneration tie({1}, 1, {7}, 8, 16);
  scores.assign(8, 0);
  assert(tie.sample(0, scores) == 0 && tie.finish_reason() == "length");
  GreedyGeneration bad({1}, 2, {7}, 8, 16);
  scores[2] = std::numeric_limits<float>::infinity();
  rejects([&] { bad.sample(0, scores); });
  assert(bad.generated().empty() && bad.evaluated().size() == 1);
  rejects([&] { bad.receipt(); });
  rejects([] { GreedyGeneration g({}, 1, {1}); });
  rejects([] { GreedyGeneration g({1}, 1, {}); });
  rejects([] { GreedyGeneration g({1}, 1, {2, 2}); });
  rejects([] { GreedyGeneration g({1}, 16, {2}, 8, 16); });
  rejects([] { GreedyGeneration g({8}, 1, {2}, 8, 16); });
  assert(vocabulary_ids("0,7", 8) == std::vector<int>({0, 7}));
  for (auto text : {"", "1,", ",1", "1,,2", "-1", "1x", "1e2", "8", " 1"})
    rejects([&] { vocabulary_ids(text, 8); });
  assert(json_ids({1, 2}) == "[1,2]");
  std::cout << "ring_generation_test: ok\n";
}
