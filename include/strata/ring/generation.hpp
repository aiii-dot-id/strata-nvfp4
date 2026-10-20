// AI Identity Incorported <james@aiii.id>
#pragma once
#include <algorithm>
#include <cmath>
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
  std::vector<int> prompt_, evaluated_, generated_, eos_;
  size_t limit_, capacity_;
  int vocab_;
  std::string reason_;

public:
  GreedyGeneration(std::vector<int> prompt, size_t limit, std::vector<int> eos,
                   int vocab = 157184, size_t capacity = 4096)
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
    const int token = static_cast<int>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
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
    return "{\"schema\":\"" + schema + "\",\"prompt_token_ids\":" +
           json_ids(prompt_) +
           ",\"generated_token_ids\":" + json_ids(generated_) +
           ",\"eos_token_ids\":" + json_ids(eos_) +
           ",\"max_new_tokens\":" + std::to_string(limit_) +
           ",\"finish_reason\":\"" + reason_ + "\"}\n";
  }
};
} // namespace strata::ring
