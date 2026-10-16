// AI Identity Incorported <james@aiii.id>
#include <cassert>
#include <cfloat>
#include <iostream>
#include <limits>
#include <strata/ring/numerics.hpp>

template <class Fn> void rejects(Fn fn) {
  bool threw = false;
  try {
    fn();
  } catch (const std::exception &) {
    threw = true;
  }
  assert(threw);
}
int main() {
  using strata::ring::rms_norm;
  const auto v = rms_norm({3.f, 4.f}, {1.f, 2.f});
  const float inv = 1.f / std::sqrt(12.5f + 1e-6f);
  assert(v[0] == 3.f * inv && v[1] == 8.f * inv);
  assert(rms_norm({0.f, 0.f}, {1.f, 1.f}) == std::vector<float>({0, 0}));
  rejects([&] { rms_norm({FLT_MAX}, {1.f}); });
  rejects([&] { rms_norm({1.5e19f, 1.5e19f}, {1.f, 1.f}); });
  rejects([&] { rms_norm({std::numeric_limits<float>::quiet_NaN()}, {1.f}); });
  rejects([&] { rms_norm({0.f}, {std::numeric_limits<float>::infinity()}); });
  rejects([&] { rms_norm({1.f, 0.f}, {FLT_MAX, 1.f}); });
  rejects([&] { rms_norm({}, {}); });
  rejects([&] { rms_norm({1.f}, {1.f, 1.f}); });
  rejects([&] { rms_norm({1.f}, {1.f}, 0.f); });
  std::cout << "ring_numerics_test: ok\n";
}
