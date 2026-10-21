// AI Identity Incorported <james@aiii.id>
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "STRALOR1 requires a little-endian host");
#endif

namespace strata::ring {

struct TerminalAdapter {
  uint32_t modelcode = 0, expert = 0, rank = 0;
  float alpha = 0.0f;
  size_t hidden = 0;
  std::vector<float> gate_a, gate_b, up_a, up_b, down_a, down_b;
  bool gate_b_zero = true, up_b_zero = true, down_b_zero = true;

  static TerminalAdapter load(const std::string &path, uint32_t expected_model,
                              size_t expected_hidden, uint32_t max_expert) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("terminal adapter open failed");
    char magic[8]; in.read(magic, 8);
    if (std::memcmp(magic, "STRALOR1", 8) != 0)
      throw std::runtime_error("terminal adapter magic mismatch");
    auto u32 = [&]() { uint32_t v[1]{}; in.read(reinterpret_cast<char *>(v), 4); if (!in) throw std::runtime_error("terminal adapter truncated"); return v[0]; };
    auto f32 = [&]() { float v[1]{}; in.read(reinterpret_cast<char *>(v), 4); if (!in || !std::isfinite(v[0])) throw std::runtime_error("terminal adapter nonfinite or truncated"); return v[0]; };
    TerminalAdapter a; a.modelcode = u32(); a.expert = u32(); a.rank = u32(); a.alpha = f32(); a.hidden = expected_hidden;
    if (a.modelcode != expected_model || a.expert >= max_expert || a.rank == 0 || a.rank > 32)
      throw std::runtime_error("terminal adapter bounds mismatch");
    auto read = [&](size_t n) { std::vector<float> v(n); in.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(n * 4)); if (!in) throw std::runtime_error("terminal adapter truncated"); for (float x : v) if (!std::isfinite(x)) throw std::runtime_error("terminal adapter nonfinite"); return v; };
    a.gate_a = read(size_t(a.rank) * expected_hidden); a.gate_b = read(512u * a.rank);
    a.up_a = read(size_t(a.rank) * expected_hidden); a.up_b = read(512u * a.rank);
    a.down_a = read(size_t(a.rank) * 512u); a.down_b = read(expected_hidden * a.rank);
    char extra; if (in.read(&extra, 1)) throw std::runtime_error("terminal adapter trailing bytes");
    auto zero = [](const std::vector<float> &v) { for (float x : v) if (x != 0.0f) return false; return true; };
    a.gate_b_zero = zero(a.gate_b); a.up_b_zero = zero(a.up_b); a.down_b_zero = zero(a.down_b);
    return a;
  }

  void add_gate_up(const std::vector<float> &x, std::vector<float> &g, std::vector<float> &u) const {
    const float scale = alpha / static_cast<float>(rank);
    auto add = [&](const std::vector<float> &A, const std::vector<float> &B, bool skip, std::vector<float> &out) {
      if (skip) return;
      for (uint32_t r = 0; r < rank; ++r) { float z = 0.0f; for (size_t j = 0; j < hidden; ++j) z += A[size_t(r) * hidden + j] * x[j]; for (size_t i = 0; i < 512; ++i) out[i] += scale * B[size_t(i) * rank + r] * z; }
    };
    add(gate_a, gate_b, gate_b_zero, g); add(up_a, up_b, up_b_zero, u);
  }
  void add_down(const std::vector<float> &act, std::vector<float> &y) const {
    if (down_b_zero) return;
    const float scale = alpha / static_cast<float>(rank);
    for (uint32_t r = 0; r < rank; ++r) { float z = 0.0f; for (size_t j = 0; j < 512; ++j) z += down_a[size_t(r) * 512 + j] * act[j]; for (size_t i = 0; i < hidden; ++i) y[i] += scale * down_b[i * rank + r] * z; }
  }
};

} // namespace strata::ring
