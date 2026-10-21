// AI Identity Incorported <james@aiii.id>
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::ring {
inline size_t unsigned_decimal(const std::string &text) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("expected an unsigned decimal integer");
  size_t end = 0;
  auto n = std::stoull(text, &end);
  if (end != text.size() || static_cast<size_t>(n) != n)
    throw std::invalid_argument("integer out of range");
  return static_cast<size_t>(n);
}
inline uint64_t uint64_decimal(const std::string &text) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("expected an unsigned decimal integer");
  size_t end = 0;
  unsigned long long n = 0;
  try {
    n = std::stoull(text, &end);
  } catch (...) {
    throw std::invalid_argument("integer out of range");
  }
  if (end != text.size())
    throw std::invalid_argument("integer out of range");
  return static_cast<uint64_t>(n);
}
inline double positive_finite_decimal(const std::string &text) {
  size_t end = 0;
  double value = 0;
  try {
    value = std::stod(text, &end);
  } catch (...) {
    throw std::invalid_argument("expected a positive finite temperature");
  }
  if (end != text.size() || !std::isfinite(value) || value <= 0)
    throw std::invalid_argument("expected a positive finite temperature");
  return value;
}
inline std::vector<int> vocabulary_ids(const std::string &text, int vocab) {
  if (text.empty() || text.back() == ',')
    throw std::invalid_argument("empty token ID");
  std::vector<int> ids;
  size_t pos = 0;
  while (pos < text.size()) {
    auto end = text.find(',', pos);
    if (end == std::string::npos)
      end = text.size();
    auto n = unsigned_decimal(text.substr(pos, end - pos));
    if (n >= static_cast<size_t>(vocab))
      throw std::invalid_argument("token outside vocabulary");
    ids.push_back(static_cast<int>(n));
    pos = end + 1;
  }
  return ids;
}
inline std::string json_ids(const std::vector<int> &ids) {
  std::string out = "[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i)
      out += ',';
    out += std::to_string(ids[i]);
  }
  return out + ']';
}
class GreedyGeneration {
  struct CategoricalSampler {
    double temperature;
    uint64_t seed;
    std::mt19937_64 rng;
    std::vector<double> logprobs;
    std::vector<double> uniforms;

    CategoricalSampler(double t, uint64_t s) : temperature(t), seed(s), rng(s) {}

    int sample(const std::vector<float> &logits) {
      const float max_logit = *std::max_element(logits.begin(), logits.end());
      std::vector<double> weights(logits.size());
      double total = 0.0;
      for (size_t i = 0; i < logits.size(); ++i) {
        const double scaled =
            (static_cast<double>(logits[i]) - static_cast<double>(max_logit)) /
            temperature;
        // The maximum-shifted exponent is never above one.  Extremely small
        // temperatures may overflow the division for a non-maximum value;
        // that value has probability zero.
        const double weight = std::isfinite(scaled) && scaled > -745.0
                                  ? std::exp(scaled)
                                  : 0.0;
        weights[i] = weight;
        total += weight;
      }
      if (!(total > 0.0) || !std::isfinite(total))
        throw std::runtime_error("invalid categorical probability mass");
      // Convert 53 bits directly so the draw is reproducible independently of
      // implementation-specific uniform_real_distribution details.
      const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53;
      uniforms.push_back(u);
      const double target = u * total;
      double cumulative = 0.0;
      size_t chosen = weights.size() - 1;
      for (size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i];
        if (target < cumulative) {
          chosen = i;
          break;
        }
      }
      logprobs.push_back(
          (static_cast<double>(logits[chosen]) -
           static_cast<double>(max_logit)) /
              temperature -
          std::log(total));
      return static_cast<int>(chosen);
    }
  };

  std::vector<int> prompt_, evaluated_, generated_, eos_;
  size_t limit_, capacity_;
  int vocab_;
  std::string reason_;
  std::optional<CategoricalSampler> sampler_;

public:
  GreedyGeneration(std::vector<int> prompt, size_t limit, std::vector<int> eos,
                   int vocab = 157184, size_t capacity = 4096,
                   std::optional<std::pair<double, uint64_t>> sampling = std::nullopt)
      : prompt_(prompt), evaluated_(std::move(prompt)), eos_(std::move(eos)),
        limit_(limit), capacity_(capacity), vocab_(vocab) {
    if (vocab_ <= 0 || prompt_.empty() || prompt_.size() > capacity_ ||
        limit_ > capacity_ - prompt_.size())
      throw std::invalid_argument("prompt and generation exceed context limit");
    for (int id : prompt_)
      if (id < 0 || id >= vocab_)
        throw std::invalid_argument("invalid prompt token");
    for (int id : eos_)
      if (id < 0 || id >= vocab_)
        throw std::invalid_argument("invalid EOS token");
    auto sorted = eos_;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end() ||
        (limit_ && eos_.empty()))
      throw std::invalid_argument("generation needs unique EOS IDs");
    if (sampling) {
      if (!std::isfinite(sampling->first) || sampling->first <= 0)
        throw std::invalid_argument("expected a positive finite temperature");
      sampler_.emplace(sampling->first, sampling->second);
    }
  }
  const std::vector<int> &prompt() const { return prompt_; }
  const std::vector<int> &evaluated() const { return evaluated_; }
  const std::vector<int> &generated() const { return generated_; }
  const std::vector<int> &eos() const { return eos_; }
  const std::string &finish_reason() const { return reason_; }
  size_t limit() const { return limit_; }
  size_t kv_capacity() const { return prompt_.size() + limit_; }
  bool sampling_position(size_t pos) const {
    return limit_ && reason_.empty() && pos + 1 == evaluated_.size();
  }
  int sample(size_t pos, const std::vector<float> &logits) {
    if (!sampling_position(pos) || logits.size() != static_cast<size_t>(vocab_))
      throw std::logic_error("invalid generation sampling position or logits");
    for (float x : logits)
      if (!std::isfinite(x))
        throw std::runtime_error("nonfinite generation logits");
    const int token = sampler_ ? sampler_->sample(logits)
                               : static_cast<int>(std::distance(
                                     logits.begin(),
                                     std::max_element(logits.begin(), logits.end())));
    generated_.push_back(token);
    if (std::find(eos_.begin(), eos_.end(), token) != eos_.end())
      reason_ = "eos";
    else if (generated_.size() == limit_)
      reason_ = "length";
    else
      evaluated_.push_back(token);
    return token;
  }
  // Resume before sampling the token following the last committed position.
  // Cache restore is the caller's responsibility and must commit atomically.
  void resume_prefix(const std::vector<int>& prefix,
                     const std::vector<float>& last_logits) {
    if (sampler_)
      throw std::invalid_argument("stochastic sampling is incompatible with continuation");
    if (prefix.size() > kv_capacity())
      throw std::invalid_argument("resumed prefix exceeds generation capacity");
    for(int id:prefix) if(id<0 || id>=vocab_)
      throw std::invalid_argument("invalid resumed prefix token");
    const size_t common=std::min(prefix.size(),prompt_.size());
    if(!std::equal(prefix.begin(),prefix.begin()+common,prompt_.begin()))
      throw std::invalid_argument("resumed prompt prefix mismatch");
    GreedyGeneration staged(*this);
    if(prefix.size()>=prompt_.size()) {
      if(!limit_) {
        if(prefix.size()!=prompt_.size())
          throw std::invalid_argument("teacher-forced prefix exceeds prompt");
      } else {
        staged.generated_.assign(prefix.begin()+prompt_.size(),prefix.end());
        if(staged.generated_.size()>=limit_)
          throw std::invalid_argument("resumed generation exceeds budget");
        for(int id:staged.generated_)
          if(std::find(eos_.begin(),eos_.end(),id)!=eos_.end())
            throw std::invalid_argument("resumed prefix contains committed EOS");
        staged.evaluated_=prefix;staged.reason_.clear();
        staged.sample(prefix.size()-1,last_logits);
      }
    }
    *this=std::move(staged);
  }
  std::string receipt(const std::string &schema = "strata-ring-generation-v1") const {
    if (!limit_ || reason_.empty())
      throw std::logic_error("generation not complete");
    if (schema != "strata-ring-generation-v1" && schema != "strata-ling-generation-v1")
      throw std::invalid_argument("unsupported generation receipt schema");
    std::string out = "{\"schema\":\"" + schema + "\",\"prompt_token_ids\":" +
           json_ids(prompt_) +
           ",\"generated_token_ids\":" + json_ids(generated_) +
           ",\"eos_token_ids\":" + json_ids(eos_) +
           ",\"max_new_tokens\":" + std::to_string(limit_) +
           ",\"finish_reason\":\"" + reason_ + "\"";
    if (sampler_) {
      auto json_doubles = [](const std::vector<double> &values) {
        std::ostringstream stream;
        stream << '[' << std::setprecision(17);
        for (size_t i = 0; i < values.size(); ++i) {
          if (i)
            stream << ',';
          stream << values[i];
        }
        stream << ']';
        return stream.str();
      };
      std::ostringstream temperature;
      temperature << std::setprecision(17) << sampler_->temperature;
      out += ",\"sampling\":{\"method\":\"categorical_full_vocab\""
             ",\"temperature\":" + temperature.str() +
             ",\"seed\":" + std::to_string(sampler_->seed) +
             ",\"algorithm\":\"mt19937_64_double_max_shift\""
             ",\"sampled_logprobs\":" + json_doubles(sampler_->logprobs) +
             ",\"uniform_draws\":" + json_doubles(sampler_->uniforms) + '}';
    }
    return out + "}\n";
  }
};
} // namespace strata::ring
