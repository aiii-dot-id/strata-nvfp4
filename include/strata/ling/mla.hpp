// AI Identity Incorported <james@aiii.id>
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace strata::ling {
enum class MlaRopeFrequency { Native, Original };
struct MlaConfig {
  size_t heads = 16, nope = 128, rope = 64, value = 128;
  float theta = 6000000.0f;
  MlaRopeFrequency rope_frequency = MlaRopeFrequency::Native;
};
class MlaCache {
  MlaConfig config_;
  size_t capacity_, length_ = 0;
  std::vector<float> keys_, values_;
public:
  MlaCache(MlaConfig config, size_t capacity);
  size_t length() const { return length_; }
  size_t capacity() const { return capacity_; }
  const MlaConfig &config() const noexcept { return config_; }
  // Logical payload only. Vectors reserve these slots at construction; live
  // bytes count just the positions committed by step().
  size_t live_payload_bytes() const noexcept;
  size_t capacity_payload_bytes() const noexcept;
  static float inverse_frequency(const MlaConfig &config, size_t index);
  const std::vector<float> &keys() const { return keys_; }
  const std::vector<float> &values() const { return values_; }
  // The continuation representation contains committed slots only; unused
  // capacity is deliberately not serialized.
  std::vector<float> live_keys() const;
  std::vector<float> live_values() const;
  // Transactional restore of committed key/value slots. Invalid length,
  // geometry, or values leave this cache unchanged.
  bool restore_live(size_t length, const std::vector<float> &keys,
                    const std::vector<float> &values, std::string &error);
  // Inputs are projected Q, per-head nonrotary K, shared rotary K, V,
  // and one gate logit per head. Projection/norm weights live in the engine.
  // A failed call leaves both cache arrays and length unchanged.
  std::vector<float> step(const std::vector<float> &query,
                          const std::vector<float> &key_nope,
                          const std::vector<float> &key_rope,
                          const std::vector<float> &value,
                          const std::vector<float> &gate_logits,
                          size_t position);
};
}
