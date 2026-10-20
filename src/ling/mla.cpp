// AI Identity Incorported <james@aiii.id>
#include "strata/ling/mla.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
float original_inverse_frequency(size_t index) {
  // Exact float32 values emitted by transformers 4.55.4's
  // _compute_default_rope_parameters for Ling-3.0-tiny (theta=6000000,
  // head_dim=rope_dim=64). The table is part of the original buffer metadata.
  static constexpr uint32_t bits[32] = {
      0x3f800000, 0x3f1d3070, 0x3ec108e6, 0x3e6d0df6,
      0x3e118e6a, 0x3db2bfc4, 0x3d5b82da, 0x3d06c8c9,
      0x3ca58548, 0x3c4b441d, 0x3bf99e78, 0x3b99456b,
      0x3b3c3926, 0x3ae72552, 0x3a8deda0, 0x3a2e4b2b,
      0x39d60a28, 0x39836cbb, 0x39216519, 0x38c63314,
      0x387365a7, 0x38157366, 0x37b78819, 0x37616263,
      0x370a63fc, 0x36a9f301, 0x3650b460, 0x3600261c,
      0x359d5f3e, 0x3541425e, 0x34ed548b, 0x3491b9c1};
  if (index >= 32) throw std::invalid_argument("original RoPE index out of range");
  float value;
  std::memcpy(&value, &bits[index], sizeof(value));
  return value;
}
}
float MlaCache::inverse_frequency(const MlaConfig &c, size_t index) {
  if (index >= c.rope / 2) throw std::invalid_argument("RoPE index out of range");
  if (c.rope_frequency == MlaRopeFrequency::Original) {
    if (c.rope != 64 || c.theta != 6000000.0f)
      throw std::invalid_argument("original Ling RoPE metadata requires theta=6000000 and rope=64");
    return original_inverse_frequency(index);
  }
  const float exponent = 2.0f * float(index) / float(c.rope);
  const float value = 1.0f / std::pow(c.theta, exponent);
  if (!std::isfinite(value)) throw std::invalid_argument("nonfinite RoPE frequency");
  return value;
}
MlaCache::MlaCache(MlaConfig c, size_t capacity)
    : config_(c), capacity_(capacity) {
  if (!c.heads || !c.nope || !c.rope || (c.rope&1) || !c.value ||
      !capacity || !(c.theta > 0) || !std::isfinite(c.theta) ||
      c.nope > std::numeric_limits<size_t>::max()-c.rope)
    throw std::invalid_argument("invalid MLA geometry");
  if (c.rope_frequency == MlaRopeFrequency::Original &&
      (c.rope != 64 || c.theta != 6000000.0f))
    throw std::invalid_argument("original Ling RoPE metadata requires theta=6000000 and rope=64");
  keys_.resize(product(product(capacity,c.heads),c.nope+c.rope));
  values_.resize(product(product(capacity,c.heads),c.value));
}
size_t MlaCache::live_payload_bytes() const noexcept {
  const size_t per_position = config_.heads * (config_.nope + config_.rope + config_.value);
  return length_ * per_position * sizeof(float);
}
size_t MlaCache::capacity_payload_bytes() const noexcept {
  return (keys_.size() + values_.size()) * sizeof(float);
}
std::vector<float> MlaCache::live_keys() const {
  const size_t per = config_.heads * (config_.nope + config_.rope);
  return std::vector<float>(keys_.begin(), keys_.begin() + length_ * per);
}
std::vector<float> MlaCache::live_values() const {
  const size_t per = config_.heads * config_.value;
  return std::vector<float>(values_.begin(), values_.begin() + length_ * per);
}
bool MlaCache::restore_live(size_t length, const std::vector<float> &keys,
                            const std::vector<float> &values,
                            std::string &error) {
  error.clear();
  const size_t key_per = config_.heads * (config_.nope + config_.rope);
  const size_t value_per = config_.heads * config_.value;
  if (length > capacity_ || keys.size() != length * key_per ||
      values.size() != length * value_per) {
    error = "MLA continuation state geometry mismatch";
    return false;
  }
  const auto finite = [](const std::vector<float> &v) {
    return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
  };
  if (!finite(keys) || !finite(values)) {
    error = "MLA continuation state must be finite";
    return false;
  }
  // Stage complete capacity arrays so neither vector nor length changes on an
  // invalid input or allocation failure.
  std::vector<float> next_keys(keys_.size(), 0.0f);
  std::vector<float> next_values(values_.size(), 0.0f);
  std::copy(keys.begin(), keys.end(), next_keys.begin());
  std::copy(values.begin(), values.end(), next_values.begin());
  keys_.swap(next_keys);
  values_.swap(next_values);
  length_ = length;
  return true;
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
    const float inv = inverse_frequency(c, i);
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
