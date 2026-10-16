// AI Identity Incorported <james@aiii.id>
#include "strata/ling/mla.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>

int main() {
  using namespace strata::ling;
  MlaConfig c{2,2,4,2,100.0f};
  MlaCache cache(c,3);
  std::vector<float> q(12),kn(4),kr(4),v{2,4,6,8},g(2);
  auto first=cache.step(q,kn,kr,v,g,0);
  assert((first==std::vector<float>{1,2,3,4}));
  auto checkpoint=cache;
  auto second=cache.step(q,kn,kr,{4,8,10,12},g,1);
  assert((second==std::vector<float>{1.5f,3,4,5}));
  assert(checkpoint.step(q,kn,kr,{4,8,10,12},g,1)==second);
  assert(checkpoint.keys()==cache.keys() && checkpoint.values()==cache.values());
  const auto keys=cache.keys(),values=cache.values();
  try { auto bad=q;bad[0]=std::numeric_limits<float>::quiet_NaN();cache.step(bad,kn,kr,v,g,2);assert(false); }
  catch(const std::exception &) {}
  assert(cache.length()==2 && cache.keys()==keys && cache.values()==values);
  try { cache.step(q,kn,kr,v,g,1);assert(false); } catch(const std::exception &) {}
  assert(cache.length()==2 && cache.keys()==keys && cache.values()==values);
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
  std::cout<<"Ling MLA CPU controls passed\n";
}
