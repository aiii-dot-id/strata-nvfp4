// AI Identity Incorported <james@aiii.id>
#include "strata/ling/mla.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strata::ling {
namespace {
size_t product(size_t a, size_t b) {
  if (!a || !b || a > std::numeric_limits<size_t>::max() / b)
    throw std::invalid_argument("MLA geometry overflows");
  return a*b;
}
void require(const std::vector<float> &v, size_t n) {
  if (v.size() != n) throw std::invalid_argument("MLA input shape mismatch");
  for (float x : v)
    if (!std::isfinite(x)) throw std::invalid_argument("nonfinite MLA input");
}
float sum(const float *v, size_t n) {
  if (n == 1) return *v;
  const size_t half = n/2;
  return sum(v, half) + sum(v+half, n-half);
}
float sigmoid(float x) {
  if (x >= 0) return 1.0f/(1.0f+std::exp(-x));
  float e = std::exp(x); return e/(1.0f+e);
}
}
MlaCache::MlaCache(MlaConfig c, size_t capacity)
    : config_(c), capacity_(capacity) {
  if (!c.heads || !c.nope || !c.rope || (c.rope&1) || !c.value ||
      !capacity || !(c.theta > 0) || !std::isfinite(c.theta) ||
      c.nope > std::numeric_limits<size_t>::max()-c.rope)
    throw std::invalid_argument("invalid MLA geometry");
  keys_.resize(product(product(capacity,c.heads),c.nope+c.rope));
  values_.resize(product(product(capacity,c.heads),c.value));
}
std::vector<float> MlaCache::step(const std::vector<float> &query,
                                 const std::vector<float> &key_nope,
                                 const std::vector<float> &key_rope,
                                 const std::vector<float> &value,
                                 const std::vector<float> &gate_logits,
                                 size_t position) {
  const auto &c = config_;
  const size_t kd = c.nope+c.rope, half = c.rope/2;
  if (position != length_ || position >= capacity_)
    throw std::invalid_argument("MLA cache position mismatch or capacity exceeded");
  require(query,c.heads*kd); require(key_nope,c.heads*c.nope);
  require(key_rope,c.rope); require(value,c.heads*c.value);
  require(gate_logits,c.heads);
  std::vector<float> q(query), k(c.heads*kd), kr(c.rope);
  for (size_t i=0;i<half;++i) {
    const float inv = 1.0f/std::pow(c.theta,2.0f*float(i)/float(c.rope));
    const float angle = float(position)*inv;
    const float co=std::cos(angle), si=std::sin(angle);
    const float a=key_rope[2*i], b=key_rope[2*i+1];
    kr[i]=a*co-b*si; kr[half+i]=b*co+a*si;
    for (size_t h=0;h<c.heads;++h) {
      const size_t base=h*kd+c.nope;
      const float qa=query[base+2*i], qb=query[base+2*i+1];
      q[base+i]=qa*co-qb*si; q[base+half+i]=qb*co+qa*si;
    }
  }
  for (size_t h=0;h<c.heads;++h) {
    std::copy_n(key_nope.data()+h*c.nope,c.nope,k.data()+h*kd);
    std::copy(kr.begin(),kr.end(),k.begin()+h*kd+c.nope);
  }
  require(q,q.size()); require(k,k.size());
  std::vector<float> out(c.heads*c.value), scores(length_+1), terms(kd);
  const float scale=std::pow(float(kd),-0.5f);
  for (size_t h=0;h<c.heads;++h) {
    for (size_t t=0;t<=length_;++t) {
      const float *key = t==length_ ? k.data()+h*kd : keys_.data()+(t*c.heads+h)*kd;
      for (size_t d=0;d<kd;++d) terms[d]=q[h*kd+d]*key[d];
      scores[t]=sum(terms.data(),kd)*scale;
    }
    require(scores,scores.size());
    const float maximum=*std::max_element(scores.begin(),scores.end());
    for (float &x:scores) x=std::exp(x-maximum);
    const float denominator=sum(scores.data(),scores.size());
    if (!(denominator>0) || !std::isfinite(denominator))
      throw std::runtime_error("invalid MLA softmax");
    for (float &x:scores) x/=denominator;
    for (size_t t=0;t<=length_;++t) {
      const float *v=t==length_ ? value.data()+h*c.value : values_.data()+(t*c.heads+h)*c.value;
      for (size_t d=0;d<c.value;++d) out[h*c.value+d]+=scores[t]*v[d];
    }
    const float gate=sigmoid(gate_logits[h]);
    for (size_t d=0;d<c.value;++d) out[h*c.value+d]*=gate;
  }
  require(out,out.size());
  std::copy(k.begin(),k.end(),keys_.begin()+length_*c.heads*kd);
  std::copy(value.begin(),value.end(),values_.begin()+length_*c.heads*c.value);
  ++length_;
  return out;
}
}
