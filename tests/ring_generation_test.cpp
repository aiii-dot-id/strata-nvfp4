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
  rejects([] { positive_finite_decimal("nan"); });
  rejects([] { positive_finite_decimal("inf"); });
  rejects([] { positive_finite_decimal("0"); });
  rejects([] { positive_finite_decimal("-1"); });
  assert(uint64_decimal("18446744073709551615") ==
         std::numeric_limits<uint64_t>::max());
  rejects([] { uint64_decimal("18446744073709551616"); });

  std::vector<float> flat(256, 0.0f);
  GreedyGeneration stochastic_a({1}, 8, {255}, 256, 16,
                                 std::make_pair(1.0, uint64_t(61021)));
  GreedyGeneration stochastic_b({1}, 8, {255}, 256, 16,
                                 std::make_pair(1.0, uint64_t(61021)));
  for (size_t i = 0; i < 8; ++i) {
    assert(stochastic_a.sample(i, flat) == stochastic_b.sample(i, flat));
    if (!stochastic_a.finish_reason().empty())
      break;
  }
  assert(stochastic_a.receipt() == stochastic_b.receipt());
  assert(stochastic_a.receipt().find("\"method\":\"categorical_full_vocab\"") !=
         std::string::npos);
  assert(stochastic_a.receipt().find("\"temperature\":1") != std::string::npos);
  assert(stochastic_a.receipt().find("\"temperature\":[") == std::string::npos);
  assert(stochastic_a.receipt().find("\"sampled_logprobs\":[") !=
         std::string::npos);
  assert(stochastic_a.receipt().find("\"uniform_draws\":[") !=
         std::string::npos);

  GreedyGeneration stochastic_c({1}, 8, {255}, 256, 16,
                                 std::make_pair(1.0, uint64_t(61022)));
  for (size_t i = 0; i < 8; ++i) {
    stochastic_c.sample(i, flat);
    if (!stochastic_c.finish_reason().empty())
      break;
  }
  assert(stochastic_c.receipt() != stochastic_a.receipt());

  std::vector<float> sharp(8, -100.0f);
  sharp[3] = 100.0f;
  GreedyGeneration sharp_generation({1}, 3, {7}, 8, 16,
                                    std::make_pair(0.0001, uint64_t(1)));
  assert(sharp_generation.sample(0, sharp) == 3);
  assert(sharp_generation.sample(1, sharp) == 3);
  assert(sharp_generation.sample(2, sharp) == 3);
  assert(sharp_generation.finish_reason() == "length");

  std::vector<float> extreme = {-std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max(),
                                0.0f, -1.0f};
  GreedyGeneration extreme_generation({1}, 1, {3}, 4, 8,
                                      std::make_pair(1e-300, uint64_t(2)));
  assert(extreme_generation.sample(0, extreme) == 1);
  assert(extreme_generation.finish_reason() == "length");
  std::cout << "ring_generation_test: ok\n";
}
