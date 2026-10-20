// AI Identity Incorported <james@aiii.id>
#include "strata/ling/mla.hpp"
#include "strata/ling/rms.hpp"
#include "strata/ring/generation.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iostream>

int main() {
  using namespace strata::ling;
  strata::ring::GreedyGeneration bounded({1,2,3}, 4, {9}, 32, 8);
  MlaCache generation_capacity(MlaConfig{}, bounded.kv_capacity());
  assert(bounded.kv_capacity() == 7);
  assert(generation_capacity.capacity() == bounded.kv_capacity());
  MlaConfig c{2,2,4,2,100.0f};
  MlaCache cache(c,3);
  // MLA reserves key/value slots for the configured capacity, but only the
  // committed positions form its live logical payload.
  assert(cache.live_payload_bytes() == 0);
  assert(cache.capacity_payload_bytes() == 192);
  std::vector<float> q(12),kn(4),kr(4),v{2,4,6,8},g(2);
  auto first=cache.step(q,kn,kr,v,g,0);
  assert(cache.live_payload_bytes() == 64);
  assert(cache.live_payload_bytes() < cache.capacity_payload_bytes());
  assert((first==std::vector<float>{1,2,3,4}));
  auto checkpoint=cache;
  auto second=cache.step(q,kn,kr,{4,8,10,12},g,1);
  assert((second==std::vector<float>{1.5f,3,4,5}));
  assert(cache.live_payload_bytes() == 128);
  assert(checkpoint.step(q,kn,kr,{4,8,10,12},g,1)==second);
  assert(checkpoint.keys()==cache.keys() && checkpoint.values()==cache.values());
  const auto keys=cache.keys(),values=cache.values();
  try { auto bad=q;bad[0]=std::numeric_limits<float>::quiet_NaN();cache.step(bad,kn,kr,v,g,2);assert(false); }
  catch(const std::exception &) {}
  assert(cache.length()==2 && cache.keys()==keys && cache.values()==values);
  try { cache.step(q,kn,kr,v,g,1);assert(false); } catch(const std::exception &) {}
  assert(cache.length()==2 && cache.keys()==keys && cache.values()==values);
  auto third=cache.step(q,kn,kr,v,g,2);
  (void)third;
  assert(cache.live_payload_bytes() == cache.capacity_payload_bytes());
  try { MlaCache invalid(MlaConfig{},0); (void)invalid; assert(false); }
  catch(const std::invalid_argument &) {}
  try {
    MlaConfig overflow{}; overflow.heads=std::numeric_limits<size_t>::max();
    MlaCache invalid(overflow,2); (void)invalid; assert(false);
  } catch(const std::invalid_argument &) {}
  // Explicit one-head two-position oracle, evaluated in double precision.
  MlaConfig d{1,1,2,1,100.0f};MlaCache nonzero(d,2);
  nonzero.step({0.2f,0.3f,-0.4f},{0.7f},{0.1f,0.9f},{2.0f},{0.0f},0);
  const auto output=nonzero.step({-0.2f,0.6f,0.5f},{-0.8f},{0.4f,-0.3f},{5.0f},{0.7f},1);
  const double co=std::cos(1.0),si=std::sin(1.0);
  const double qa=0.6*co-0.5*si,qb=0.5*co+0.6*si;
  const double ka=0.4*co+0.3*si,kb=-0.3*co+0.4*si;
  const double s0=(-0.2*0.7+qa*0.1+qb*0.9)/std::sqrt(3.0);
  const double s1=(-0.2*-0.8+qa*ka+qb*kb)/std::sqrt(3.0);
  const double expected=(std::exp(s0)*2+std::exp(s1)*5)/(std::exp(s0)+std::exp(s1))/(1+std::exp(-0.7));
  assert(std::abs(output[0]-expected)<2e-6);
  // Exact source layout: interleaved inputs become half-packed rotated keys.
  assert(std::abs(nonzero.keys()[4]-ka)<2e-7);
  assert(std::abs(nonzero.keys()[5]-kb)<2e-7);
  MlaCache real(MlaConfig{},2);
  auto r=real.step(std::vector<float>(16*192),std::vector<float>(16*128),
                  std::vector<float>(64),std::vector<float>(16*128,2),
                  std::vector<float>(16),0);
  for(float x:r) assert(x==1);
  // The opt-in original mode uses the exact float32 inv_freq buffer emitted
  // by transformers for Ling-3.0-tiny, rather than recomputing it.
  const uint32_t original_bits[] = {
      0x3f800000, 0x3f1d3070, 0x3ec108e6, 0x3e6d0df6,
      0x3e118e6a, 0x3db2bfc4, 0x3d5b82da, 0x3d06c8c9,
      0x3ca58548, 0x3c4b441d, 0x3bf99e78, 0x3b99456b,
      0x3b3c3926, 0x3ae72552, 0x3a8deda0, 0x3a2e4b2b,
      0x39d60a28, 0x39836cbb, 0x39216519, 0x38c63314,
      0x387365a7, 0x38157366, 0x37b78819, 0x37616263,
      0x370a63fc, 0x36a9f301, 0x3650b460, 0x3600261c,
      0x359d5f3e, 0x3541425e, 0x34ed548b, 0x3491b9c1};
  MlaConfig original{};
  original.rope_frequency = MlaRopeFrequency::Original;
  for (size_t i = 0; i < 32; ++i) {
    uint32_t actual;
    const float frequency = MlaCache::inverse_frequency(original, i);
    std::memcpy(&actual, &frequency, sizeof(actual));
    assert(actual == original_bits[i]);
  }
  try {
    MlaConfig bad{16,128,32,128,6000000.0f,MlaRopeFrequency::Original};
    (void)MlaCache::inverse_frequency(bad, 0);
    assert(false);
  } catch (const std::exception &) {}
  // Independent CPU model of PyTorch's vectorized CUDA RMS path: four-value
  // loads, 256 lanes, descending warp reduction, then descending 8-warp
  // reduction.  This is deliberately separate from the helper implementation.
  std::vector<float> rms_input(1536), rms_weights(1536, 1.0f);
  uint32_t state = 123456789u;
  for (size_t i = 0; i < rms_input.size(); ++i) {
    state = 1664525u * state + 1013904223u;
    const float unit = float((state >> 5) & 0xffffu) * (1.0f / 65536.0f) - 0.5f;
    rms_input[i] = unit * 1000000.0f;
  }
  const size_t vectors = rms_input.size() / 4;
  size_t threads = 1;
  while (threads * 2 <= vectors)
    threads *= 2;
  std::array<std::array<float, 4>, 512> lane{};
  for (size_t thread = 0; thread < threads; ++thread) {
    for (size_t vector = thread; vector < vectors; vector += threads)
      for (size_t j = 0; j < 4; ++j)
        lane[thread][j] += rms_input[vector * 4 + j] * rms_input[vector * 4 + j];
    for (size_t j = 1; j < 4; ++j)
      lane[thread][0] += lane[thread][j];
  }
  for (size_t offset = threads / 2; offset; offset >>= 1)
    for (size_t thread = 0; thread < offset; ++thread)
      lane[thread][0] = lane[thread][0] + lane[thread + offset][0];
  const float expected_scale = 1.0f / std::sqrt(lane[0][0] * (1.0f / 1536.0f) + 1e-6f);
  const auto torch_order = rms_norm(rms_input, rms_weights, RmsMode::TorchCudaOrder);
  const auto native_order = rms_norm(rms_input, rms_weights, RmsMode::Native);
  for (float value : torch_order)
    assert(value == value && std::isfinite(value));
  for (size_t i = 0; i < torch_order.size(); ++i)
    assert(torch_order[i] == rms_input[i] * expected_scale);
  bool reduction_order_changed = false;
  for (size_t i = 0; i < torch_order.size(); ++i)
    reduction_order_changed |= torch_order[i] != native_order[i];
  assert(reduction_order_changed);
  for (size_t size : {192u, 512u}) {
    std::vector<float> small(size), weights(size, 1.0f);
    for (size_t i = 0; i < size; ++i)
      small[i] = float((i * 37u) % 101u) - 50.0f;
    const auto out = rms_norm(small, weights, RmsMode::TorchCudaOrder);
    for (float value : out)
      assert(std::isfinite(value));
  }
  try {
    (void)rms_norm({1.0f, 2.0f, 3.0f}, {1.0f, 1.0f, 1.0f}, RmsMode::TorchCudaOrder);
    assert(false);
  } catch (const std::exception &) {}
  std::cout<<"Ling MLA CPU controls passed\n";
}
