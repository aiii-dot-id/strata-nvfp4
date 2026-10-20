#include "strata/ling/kda.hpp"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char *message) {
  std::printf("  %-64s %s\n", message, ok ? "ok" : "FAIL");
  if (!ok)
    ++failures;
}
bool close(float a, float b, float tolerance = 5e-5f) {
  return std::abs(a - b) <= tolerance * (1.0f + std::max(std::abs(a), std::abs(b)));
}
float sigmoid64(double x) {
  if (x >= 0.0)
    return float(1.0 / (1.0 + std::exp(-x)));
  const double e = std::exp(x);
  return float(e / (1.0 + e));
}
double softplus64(double x) {
  return x > 0.0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}

struct RefResult {
  std::vector<float> output;
  std::vector<float> recurrent;
  std::vector<float> q_history;
  std::vector<float> k_history;
  std::vector<float> v_history;
};

RefResult reference(const strata::ling::KdaConfig &c, const float *q,
                    const float *k, const float *v, const float *gate,
                    const float *beta, int64_t tokens,
                    const strata::ling::KdaWeights &weights) {
  const size_t H = size_t(c.num_heads), HV = size_t(c.value_heads);
  const size_t K = size_t(c.key_dim), V = size_t(c.value_dim), W = size_t(c.conv_kernel);
  const size_t QC = H * K, VC = HV * V;
  RefResult r;
  r.output.assign(size_t(tokens) * VC, 0.0f);
  r.recurrent.assign(HV * K * V, 0.0f);
  r.q_history.assign(QC * W, 0.0f);
  r.k_history.assign(QC * W, 0.0f);
  r.v_history.assign(VC * W, 0.0f);
  auto conv = [&](const float *input, const float *weight, const float *bias,
                  int64_t channels, std::vector<float> &history) {
    std::vector<float> output(size_t(channels), 0.0f);
    std::vector<float> next(history.size());
    for (int64_t channel = 0; channel < channels; ++channel) {
      const size_t base = size_t(channel) * W;
      for (size_t w = 0; w + 1 < W; ++w)
        next[base + w] = history[base + w + 1];
      next[base + W - 1] = input[channel];
      double sum = bias ? bias[channel] : 0.0;
      for (size_t w = 0; w < W; ++w)
        sum += double(next[base + w]) * double(weight[base + w]);
      const double s = 1.0 / (1.0 + std::exp(-sum));
      output[size_t(channel)] = float(sum * s);
    }
    history.swap(next);
    return output;
  };
  for (int64_t t = 0; t < tokens; ++t) {
    auto q_conv = conv(q + size_t(t) * QC, weights.q_conv, weights.q_bias,
                       int64_t(QC), r.q_history);
    auto k_conv = conv(k + size_t(t) * QC, weights.k_conv, weights.k_bias,
                       int64_t(QC), r.k_history);
    auto v_conv = conv(v + size_t(t) * VC, weights.v_conv, weights.v_bias,
                       int64_t(VC), r.v_history);
    const size_t group = HV / H;
    for (size_t vh = 0; vh < HV; ++vh) {
      const size_t qh = vh / group;
      std::vector<double> qn(K), kn(K);
      double qsum = 0.0, ksum = 0.0;
      for (size_t d = 0; d < K; ++d) {
        qsum += double(q_conv[qh * K + d]) * double(q_conv[qh * K + d]);
        ksum += double(k_conv[qh * K + d]) * double(k_conv[qh * K + d]);
      }
      const double qinv = 1.0 / std::sqrt(qsum + c.qk_epsilon);
      const double kinv = 1.0 / std::sqrt(ksum + c.qk_epsilon);
      for (size_t d = 0; d < K; ++d) {
        qn[d] = double(q_conv[qh * K + d]) * qinv;
        kn[d] = double(k_conv[qh * K + d]) * kinv;
      }
      const double a = std::exp(double(weights.a_log[vh]));
      float *state = r.recurrent.data() + vh * K * V;
      for (size_t kd = 0; kd < K; ++kd) {
        const double gate_input =
            double(gate[(size_t(t) * HV + vh) * K + kd]) +
            double(weights.dt_bias[vh * K + kd]);
        const double z = a * gate_input;
        const double factor = c.safe_gate
                                  ? std::exp(double(c.gate_lower_bound) * sigmoid64(z))
                                  : std::exp(-a * softplus64(gate_input));
        for (size_t vd = 0; vd < V; ++vd)
          state[kd * V + vd] = float(double(state[kd * V + vd]) * factor);
      }
      std::vector<double> delta(V, 0.0);
      for (size_t vd = 0; vd < V; ++vd) {
        double dot = 0.0;
        for (size_t kd = 0; kd < K; ++kd)
          dot += double(state[kd * V + vd]) * kn[kd];
        delta[vd] = (double(v_conv[vh * V + vd]) - dot) *
                    double(beta[size_t(t) * HV + vh]);
      }
      for (size_t kd = 0; kd < K; ++kd)
        for (size_t vd = 0; vd < V; ++vd)
          state[kd * V + vd] = float(double(state[kd * V + vd]) + kn[kd] * delta[vd]);
      for (size_t vd = 0; vd < V; ++vd) {
        double sum = 0.0;
        for (size_t kd = 0; kd < K; ++kd)
          sum += double(state[kd * V + vd]) * qn[kd] / std::sqrt(double(c.key_dim));
        r.output[(size_t(t) * HV + vh) * V + vd] = float(sum);
      }
    }
  }
  return r;
}

} // namespace

int main() {
  strata::ling::KdaConfig c;
  c.num_heads = 2;
  c.value_heads = 4;
  c.key_dim = 3;
  c.value_dim = 2;
  c.conv_kernel = 4;
  const size_t H = size_t(c.num_heads), HV = size_t(c.value_heads),
               K = size_t(c.key_dim), V = size_t(c.value_dim), W = size_t(c.conv_kernel);
  const int64_t T = 5;
  const size_t QC = H * K, VC = HV * V;
  std::vector<float> q(size_t(T) * QC), k(q.size()), v(size_t(T) * VC),
      gate(size_t(T) * HV * K), beta(size_t(T) * HV);
  std::vector<float> qw(QC * W), kw(QC * W), vw(VC * W), a(HV), dt(HV * K);
  for (size_t i = 0; i < q.size(); ++i) {
    q[i] = std::sin(float(i + 1) * .17f);
    k[i] = std::cos(float(i + 2) * .13f);
  }
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = std::sin(float(i + 4) * .11f);
  for (size_t i = 0; i < gate.size(); ++i)
    gate[i] = std::cos(float(i + 3) * .07f) * .2f;
  for (size_t i = 0; i < beta.size(); ++i)
    beta[i] = .2f + .1f * float(i % 5);
  for (size_t i = 0; i < qw.size(); ++i) {
    qw[i] = .2f + .03f * float(i % W);
    kw[i] = .1f + .02f * float((i + 1) % W);
  }
  for (size_t i = 0; i < vw.size(); ++i)
    vw[i] = .15f + .01f * float(i % W);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = -.2f + .05f * float(i);
  for (size_t i = 0; i < dt.size(); ++i)
    dt[i] = -.1f + .01f * float(i % K);
  strata::ling::KdaWeights weights{qw.data(), kw.data(), vw.data(), nullptr,
                                   nullptr, nullptr, a.data(), dt.data()};
  strata::ling::KdaState state(c);
  // recurrent (HV*K*V) plus q/k/v convolution histories.
  check(state.logical_payload_bytes() == (4 * 3 * 2 + 2 * 3 * 4 +
                                          2 * 3 * 4 + 4 * 2 * 4) * sizeof(float),
        "KDA logical payload reports every persistent state vector");
  std::vector<float> output(size_t(T) * VC);
  std::string error;
  const RefResult expected = reference(c, q.data(), k.data(), v.data(), gate.data(),
                                      beta.data(), T, weights);
  check(strata::ling::kda_forward(c, q.data(), k.data(), v.data(), gate.data(),
                                  beta.data(), T, weights, state, output.data(), error),
        "KDA recurrence succeeds");
  bool same = output.size() == expected.output.size();
  for (size_t i = 0; i < output.size() && same; ++i)
    same = close(output[i], expected.output[i]);
  check(same, "FP32 recurrence matches independent float64 reference");
  same = true;
  for (size_t i = 0; i < expected.recurrent.size(); ++i)
    same &= close(state.recurrent_data()[i], expected.recurrent[i]);
  check(same, "recurrent K,V state matches independent reference");
  check(state.q_conv_data()[QC * W - 1] != 0.0f &&
            state.v_conv_data()[VC * W - 1] != 0.0f,
        "short-convolution SiLU state advances");

  strata::ling::KdaState full(c), split(c);
  std::vector<float> full_output(output.size()), split_output(output.size());
  check(strata::ling::kda_forward(c, q.data(), k.data(), v.data(), gate.data(),
                                  beta.data(), T, weights, full, full_output.data(), error),
        "full chunk replay succeeds");
  check(strata::ling::kda_forward(c, q.data(), k.data(), v.data(), gate.data(),
                                  beta.data(), 2, weights, split, split_output.data(), error),
        "first partition succeeds");
  check(strata::ling::kda_forward(c, q.data() + 2 * QC, k.data() + 2 * QC,
                                  v.data() + 2 * VC, gate.data() + 2 * HV * K,
                                  beta.data() + 2 * HV, 3, weights, split,
                                  split_output.data() + 2 * VC, error),
        "second partition succeeds");
  same = true;
  for (size_t i = 0; i < full_output.size(); ++i)
    same &= close(full_output[i], split_output[i]);
  check(same, "chunk partition reproduces outputs and cache state");

  const std::vector<float> state_before(state.recurrent_data(),
                                        state.recurrent_data() + HV * K * V);
  const float original = q[0];
  q[0] = std::numeric_limits<float>::quiet_NaN();
  std::fill(output.begin(), output.end(), 17.0f);
  check(!strata::ling::kda_forward(c, q.data(), k.data(), v.data(), gate.data(),
                                   beta.data(), T, weights, state, output.data(), error),
        "non-finite input rejected");
  check(q[0] != q[0] && std::equal(state_before.begin(), state_before.end(),
                                  state.recurrent_data()) &&
            output[0] == 17.0f,
        "invalid call preserves state and output");
  q[0] = original;
  std::vector<float> bad_beta = beta;
  bad_beta[0] = 1.1f;
  check(!strata::ling::kda_forward(c, q.data(), k.data(), v.data(), gate.data(),
                                   bad_beta.data(), T, weights, state, output.data(), error),
        "out-of-range beta rejected");

  strata::ling::KdaConfig non_safe = c;
  non_safe.safe_gate = false;
  strata::ling::KdaState non_safe_state(non_safe);
  std::vector<float> non_safe_output(output.size());
  const RefResult non_safe_expected =
      reference(non_safe, q.data(), k.data(), v.data(), gate.data(), beta.data(),
                T, weights);
  check(strata::ling::kda_forward(non_safe, q.data(), k.data(), v.data(),
                                  gate.data(), beta.data(), T, weights,
                                  non_safe_state, non_safe_output.data(), error),
        "non-safe softplus gate succeeds");
  same = true;
  for (size_t i = 0; i < non_safe_output.size(); ++i)
    same &= close(non_safe_output[i], non_safe_expected.output[i]);
  check(same, "non-safe gate matches independent softplus reference");
  std::vector<float> huge_q(q.size(), FLT_MAX);
  strata::ling::KdaState overflow_state(c);
  std::fill(output.begin(), output.end(), 23.0f);
  check(!strata::ling::kda_forward(c, huge_q.data(), k.data(), v.data(),
                                   gate.data(), beta.data(), T, weights,
                                   overflow_state, output.data(), error),
        "L2 overflow rejected");
  check(overflow_state.recurrent_data()[0] == 0.0f && output[0] == 23.0f,
        "L2 overflow preserves state and output");

  try {
    strata::ling::KdaConfig invalid = c; invalid.conv_kernel = 0;
    strata::ling::KdaState bad(invalid); (void)bad; check(false, "invalid KDA state geometry rejected");
  } catch (const std::invalid_argument &) {
    check(true, "invalid KDA state geometry rejected");
  }
  try {
    strata::ling::KdaConfig overflow = c;
    overflow.value_heads = std::numeric_limits<int64_t>::max();
    overflow.num_heads = 1;
    strata::ling::KdaState bad(overflow); (void)bad; check(false, "overflowing KDA state geometry rejected");
  } catch (const std::invalid_argument &) {
    check(true, "overflowing KDA state geometry rejected");
  }

  strata::ling::KdaConfig real;
  strata::ling::KdaState real_state(real);
  check(real_state.logical_payload_bytes() == 1146880,
        "one Ling KDA logical payload is the fixed 1.09375 MiB state");
  check(18 * real_state.logical_payload_bytes() == 20643840,
        "18 Ling KDA logical payloads total the fixed 19.6875 MiB state");
  std::vector<float> rq(16 * 128), rv(16 * 128), rg(16 * 128), rb(16),
      rwq(16 * 128 * 4, 0.0f), rwk(16 * 128 * 4, 0.0f), rwv(16 * 128 * 4, 0.0f),
      ra(16), rdt(16 * 128);
  std::fill(rb.begin(), rb.end(), .5f);
  std::fill(ra.begin(), ra.end(), 0.0f);
  strata::ling::KdaWeights real_weights{rwq.data(), rwk.data(), rwv.data(),
                                        nullptr, nullptr, nullptr, ra.data(),
                                        rdt.data()};
  std::vector<float> rout(rv.size());
  check(strata::ling::kda_forward(real, rq.data(), rq.data(), rv.data(), rg.data(),
                                  rb.data(), 1, real_weights, real_state,
                                  rout.data(), error),
        "Ling 16x128 geometry succeeds");
  bool finite = true;
  for (float x : rout)
    finite &= std::isfinite(x);
  check(finite, "Ling geometry output is finite");

  // Snapshot restore must be a deep, all-vector transaction. This is the
  // unit-level counterpart of an engine continuation rollback.
  strata::ling::KdaState snapshot_source(c);
  check(strata::ling::kda_forward(c, q.data(), k.data(), v.data(), gate.data(),
                                  beta.data(), 2, weights, snapshot_source,
                                  output.data(), error),
        "KDA snapshot source advances");
  std::vector<float> saved_r(snapshot_source.recurrent_data(),
                             snapshot_source.recurrent_data()+snapshot_source.recurrent_size());
  std::vector<float> saved_q(snapshot_source.q_conv_data(),
                             snapshot_source.q_conv_data()+snapshot_source.q_conv_size());
  std::vector<float> saved_k(snapshot_source.k_conv_data(),
                             snapshot_source.k_conv_data()+snapshot_source.k_conv_size());
  std::vector<float> saved_v(snapshot_source.v_conv_data(),
                             snapshot_source.v_conv_data()+snapshot_source.v_conv_size());
  strata::ling::KdaState restored(c);
  check(restored.restore(saved_r, saved_q, saved_k, saved_v, error),
        "KDA complete snapshot restore succeeds");
  const auto before_bad_r = std::vector<float>(restored.recurrent_data(),
      restored.recurrent_data()+restored.recurrent_size());
  auto bad_q = saved_q; bad_q[0] = std::numeric_limits<float>::quiet_NaN();
  check(!restored.restore(saved_r, bad_q, saved_k, saved_v, error),
        "KDA nonfinite snapshot rejected");
  check(std::equal(before_bad_r.begin(), before_bad_r.end(), restored.recurrent_data()),
        "KDA failed restore preserves prior state");
  auto short_v = saved_v; short_v.pop_back();
  check(!restored.restore(saved_r, saved_q, saved_k, short_v, error),
        "KDA wrong snapshot geometry rejected");
  check(std::equal(before_bad_r.begin(), before_bad_r.end(), restored.recurrent_data()),
        "KDA geometry rejection preserves prior state");
  saved_r[0] += 1.0f;
  check(restored.recurrent_data()[0] != saved_r[0],
        "KDA restored state is independent from snapshot vectors");

  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
