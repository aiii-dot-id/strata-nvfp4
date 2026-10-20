// AI Identity Incorported <james@aiii.id>
// Opt-in Ring BF16 reference executor. Original weights, FP32 working state.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <strata/kernels/bf16_gemv.hpp>
#include <strata/ring/attention.hpp>
#include <strata/ring/continuation.hpp>
#include <strata/ring/expert_cache.hpp>
#include <strata/ring/generation.hpp>
#include <strata/ring/numerics.hpp>
#include <strata/ring/routing.hpp>
#include <strata/ring/tensor_store.hpp>
#include <vector>
#ifdef STRATA_TORCH_ROUTER
#include "attention.hpp"
#include "pointwise.hpp"
#include "router.hpp"
#include "rms.hpp"
#endif

namespace sr = strata::ring;
static volatile std::sig_atomic_t cancelled = 0;
static void cancel_handler(int) { cancelled = 1; }
static void check_cancelled() {
  if (cancelled)
    throw std::runtime_error("execution cancelled; no successful receipt");
}
static void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
static void check_cublas(cublasStatus_t e) {
  if (e != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS operation failed");
}
static float bf16(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
static std::vector<float> floats(const sr::TensorView &t) {
  const size_t n = t.bytes / (t.dtype == "BF16" ? 2 : 4);
  std::vector<float> v(n);
  if (t.dtype == "BF16") {
    for (size_t i = 0; i < n; ++i) {
      uint16_t b;
      std::memcpy(&b, t.data + 2 * i, 2);
      v[i] = bf16(b);
    }
  } else if (t.dtype == "F32")
    std::memcpy(v.data(), t.data, t.bytes);
  else
    throw std::runtime_error("unsupported tensor precision");
  return v;
}
static void shape(const sr::TensorView &t, std::initializer_list<size_t> dims) {
  if (t.shape != std::vector<size_t>(dims))
    throw std::runtime_error("tensor geometry mismatch");
}
static void finite(const std::vector<float> &x) {
  for (float f : x)
    if (!std::isfinite(f))
      throw std::runtime_error("nonfinite forward state");
}
static std::vector<float> norm(const std::vector<float> &x,
                               const sr::TensorView &weight) {
  shape(weight, {x.size()});
  return sr::rms_norm(x, floats(weight));
}
static void add(std::vector<float> &x, const std::vector<float> &y) {
  if (x.size() != y.size())
    throw std::runtime_error("residual shape mismatch");
  for (size_t i = 0; i < x.size(); ++i)
    x[i] += y[i];
  finite(x);
}
struct DeviceBuffer {
  void *p{};
  size_t bytes{};
  explicit DeviceBuffer(size_t n) : bytes(n) { check(cudaMalloc(&p, n)); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  ~DeviceBuffer() {
    if (p)
      cudaFree(p);
  }
};
struct ExpertDevice final : sr::ExpertCache::BackendValue {
  DeviceBuffer weights;
  explicit ExpertDevice(size_t bytes) : weights(bytes) {}
};
struct CudaBackend final : sr::ExpertCache::Backend {
  sr::ExpertCache::BackendValuePtr upload(const sr::ExpertKey &,
                                          const sr::ExpertPayload &p,
                                          std::string &e) override {
    try {
      auto v = std::make_shared<ExpertDevice>(p.bytes());
      size_t at = 0;
      for (auto *b : {&p.gate, &p.up, &p.down}) {
        check(cudaMemcpy(static_cast<uint8_t *>(v->weights.p) + at, b->data(),
                         b->size(), cudaMemcpyHostToDevice));
        at += b->size();
      }
      return v;
    } catch (const std::exception &ex) {
      e = ex.what();
      return {};
    }
  }
};
enum class MatmulBackend { Bf16Gemv, Fp32Cublas };
enum class LmHeadResidency { Off, Cuda };
static LmHeadResidency parse_lm_head_residency(const std::string &v) {
  if (v == "off") return LmHeadResidency::Off;
  if (v == "cuda") return LmHeadResidency::Cuda;
  throw std::invalid_argument("lm-head residency must be off or cuda");
}
static const char *lm_head_residency_name(LmHeadResidency v) {
  return v == LmHeadResidency::Cuda ? "cuda" : "off";
}

enum class RouterBackend { Native, TorchCuda };
static RouterBackend parse_router_backend(const std::string &v) { if(v=="native") return RouterBackend::Native; if(v=="torch-cuda") return RouterBackend::TorchCuda; throw std::invalid_argument("router backend must be native or torch-cuda"); }
static const char *router_backend_name(RouterBackend b) { return b==RouterBackend::TorchCuda ? "torch-cuda" : "native"; }
enum class RmsBackend { Native, TorchCuda };
static RmsBackend parse_rms_backend(const std::string &v) { if(v=="native") return RmsBackend::Native; if(v=="torch-cuda") return RmsBackend::TorchCuda; throw std::invalid_argument("rms mode must be native or torch-cuda"); }
static const char *rms_backend_name(RmsBackend b) { return b==RmsBackend::TorchCuda ? "torch-cuda" : "native"; }
enum class AttentionBackend { Native, TorchCuda };
static AttentionBackend parse_attention_backend(const std::string &v) { if(v=="native") return AttentionBackend::Native; if(v=="torch-cuda") return AttentionBackend::TorchCuda; throw std::invalid_argument("attention backend must be native or torch-cuda"); }
static const char *attention_backend_name(AttentionBackend b) { return b==AttentionBackend::TorchCuda ? "torch-cuda" : "native"; }
enum class PointwiseBackend { Native, TorchCuda };
static PointwiseBackend parse_pointwise_backend(const std::string &v) { if(v=="native") return PointwiseBackend::Native; if(v=="torch-cuda") return PointwiseBackend::TorchCuda; throw std::invalid_argument("pointwise backend must be native or torch-cuda"); }
static const char *pointwise_backend_name(PointwiseBackend b) { return b==PointwiseBackend::TorchCuda ? "torch-cuda" : "native"; }

static const char *matmul_backend_name(MatmulBackend backend) {
  return backend == MatmulBackend::Fp32Cublas ? "fp32-cublas" : "bf16-gemv";
}

static MatmulBackend parse_matmul_backend(const std::string &value) {
  if (value == "bf16-gemv")
    return MatmulBackend::Bf16Gemv;
  if (value == "fp32-cublas")
    return MatmulBackend::Fp32Cublas;
  throw std::invalid_argument("matmul backend must be bf16-gemv or fp32-cublas");
}

__global__ static void widen_bf16(const uint16_t *src, float *dst,
                                  size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const uint32_t bits = uint32_t(src[i]) << 16;
    dst[i] = __uint_as_float(bits);
  }
}

class Matvec {
  // Largest matrix is the unmodified BF16 output head.
  static constexpr size_t max_weight_bytes = size_t(157184) * 2048 * 2;
  DeviceBuffer weight{size_t(157184) * 2048 * 2}, x{5120 * 4}, y{157184 * 4};
  MatmulBackend backend_;
  std::unique_ptr<DeviceBuffer> fp32_weight;
  // Opt-in, separate from the reusable staging allocation. It preserves the
  // original BF16 bytes and still routes through apply_device(), including
  // the existing per-call FP32 widening and pedantic cuBLAS expression.
  std::unique_ptr<DeviceBuffer> resident_lm_head;
  size_t resident_lm_head_bytes_ = 0;
  size_t resident_lm_head_uploads_ = 0;
  size_t resident_lm_head_apply_calls_ = 0;
  cublasHandle_t cublas = nullptr;

public:
  static constexpr size_t fp32_weight_bytes = size_t(157184) * 2048 * 4;
  static constexpr size_t base_workspace_bytes =
      max_weight_bytes + 5120 * 4 + 157184 * 4;
  static constexpr size_t workspace_bytes(MatmulBackend backend) {
    return base_workspace_bytes +
           (backend == MatmulBackend::Fp32Cublas ? fp32_weight_bytes : 0);
  }
  static constexpr size_t fp32_scratch_bytes(MatmulBackend backend) {
    return backend == MatmulBackend::Fp32Cublas ? fp32_weight_bytes : 0;
  }

  explicit Matvec(MatmulBackend backend) : backend_(backend) {
    if (backend_ == MatmulBackend::Fp32Cublas) {
      fp32_weight = std::make_unique<DeviceBuffer>(fp32_weight_bytes);
      check_cublas(cublasCreate(&cublas));
      // Pedantic mode excludes TF32/reduced-precision tensor-core math. The
      // one-column GEMM has no atomics in this API and uses the default stream.
      check_cublas(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH));
      check_cublas(cublasSetAtomicsMode(cublas, CUBLAS_ATOMICS_NOT_ALLOWED));
    }
  }
  ~Matvec() {
    if (cublas)
      cublasDestroy(cublas);
  }

  std::vector<float> apply_device(const uint16_t *w, size_t rows, size_t cols,
                                  const std::vector<float> &input) {
    if (input.size() != cols || cols > 5120 || rows > 157184 || cols % 2)
      throw std::runtime_error("matvec geometry");
    check(cudaMemcpy(x.p, input.data(), cols * 4, cudaMemcpyHostToDevice));
    if (backend_ == MatmulBackend::Bf16Gemv) {
      strata::kernels::bf16_gemv_fp32_mmvf(static_cast<float *>(x.p), w,
                                           static_cast<float *>(y.p), cols, rows,
                                           nullptr);
      check(cudaGetLastError());
    } else {
      if (!fp32_weight || !cublas)
        throw std::runtime_error("cuBLAS backend is not initialized");
      const size_t count = rows * cols;
      const int blocks = static_cast<int>((count + 255) / 256);
      widen_bf16<<<blocks, 256>>>(w, static_cast<float *>(fp32_weight->p),
                                   count);
      check(cudaGetLastError());
      const float alpha = 1.0f, beta = 0.0f;
      check_cublas(cublasSgemm(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(rows), 1,
          static_cast<int>(cols), &alpha,
          static_cast<const float *>(fp32_weight->p), static_cast<int>(cols),
          static_cast<const float *>(x.p), static_cast<int>(cols), &beta,
          static_cast<float *>(y.p), static_cast<int>(rows)));
    }
    std::vector<float> output(rows);
    check(cudaMemcpy(output.data(), y.p, rows * 4, cudaMemcpyDeviceToHost));
    finite(output);
    return output;
  }
  std::vector<float> apply(const sr::TensorView &t,
                           const std::vector<float> &input) {
    if (t.shape.size() != 2 || t.shape[1] != input.size() ||
        t.dtype != "BF16" || t.bytes > weight.bytes)
      throw std::runtime_error("expected compatible original BF16 matrix");
    check(cudaMemcpy(weight.p, t.data, t.bytes, cudaMemcpyHostToDevice));
    return apply_device(static_cast<uint16_t *>(weight.p), t.shape[0],
                        t.shape[1], input);
  }
  void pin_lm_head(const sr::TensorView &t) {
    if (resident_lm_head || t.shape != std::vector<size_t>{157184, 2048} ||
        t.dtype != "BF16" || t.bytes != max_weight_bytes)
      throw std::runtime_error("lm-head residency requires original BF16 [157184,2048]");
    resident_lm_head = std::make_unique<DeviceBuffer>(t.bytes);
    check(cudaMemcpy(resident_lm_head->p, t.data, t.bytes, cudaMemcpyHostToDevice));
    resident_lm_head_bytes_ = t.bytes;
    ++resident_lm_head_uploads_;
  }
  std::vector<float> apply_lm_head(const std::vector<float> &input) {
    if (!resident_lm_head)
      throw std::runtime_error("lm-head residency was requested but is not initialized");
    ++resident_lm_head_apply_calls_;
    return apply_device(static_cast<uint16_t *>(resident_lm_head->p), 157184,
                        2048, input);
  }
  size_t resident_lm_head_bytes() const { return resident_lm_head_bytes_; }
  size_t resident_lm_head_uploads() const { return resident_lm_head_uploads_; }
  size_t resident_lm_head_apply_calls() const { return resident_lm_head_apply_calls_; }
};
static std::vector<float> swiglu(std::vector<float> g,
                                 const std::vector<float> &u) {
  if (g.size() != u.size())
    throw std::runtime_error("SwiGLU geometry");
  for (size_t i = 0; i < g.size(); ++i)
    g[i] = (g[i] / (1 + std::exp(-g[i]))) * u[i];
  finite(g);
  return g;
}
static std::vector<float> mlp(Matvec &mv, const sr::TensorStore &store,
                              const std::string &prefix,
                              const std::vector<float> &x) {
  auto g = mv.apply(store.tensor(prefix + "gate_proj.weight"), x);
  auto u = mv.apply(store.tensor(prefix + "up_proj.weight"), x);
  return mv.apply(store.tensor(prefix + "down_proj.weight"),
                  swiglu(std::move(g), u));
}
static void write_row(const std::filesystem::path &p,
                      const std::vector<float> &x) {
  std::ofstream out(p, std::ios::binary | std::ios::app);
  if (!out.write(reinterpret_cast<const char *>(x.data()), x.size() * 4))
    throw std::runtime_error("output write failed");
}
struct RouteProbeSpec {
  size_t position;
  size_t layer;
};

static RouteProbeSpec parse_route_probe(const std::string &text) {
  const size_t comma = text.find(',');
  if (comma == std::string::npos || comma == 0 || comma + 1 >= text.size() ||
      text.find(',', comma + 1) != std::string::npos)
    throw std::invalid_argument("route probe must be POS,LAYER");
  RouteProbeSpec result{sr::unsigned_decimal(text.substr(0, comma)),
                        sr::unsigned_decimal(text.substr(comma + 1))};
  if (result.layer == 0 || result.layer >= 20)
    throw std::invalid_argument("route probe layer must be a sparse layer 1..19");
  return result;
}

static float router_sigmoid(float x) {
  if (x >= 0.0f) {
    const float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(x);
  return z / (1.0f + z);
}

static void write_float_array(std::ostream &out, const std::vector<float> &x) {
  out << '[' << std::setprecision(9);
  for (size_t i = 0; i < x.size(); ++i) {
    if (i)
      out << ',';
    out << x[i];
  }
  out << ']';
}

static void write_route_probe(const std::filesystem::path &path,
                              const RouteProbeSpec &spec,
                              const std::vector<float> &gate_input,
                              const std::vector<float> &router_logits,
                              const std::vector<float> &sigmoid_scores,
                              const std::vector<float> &expert_bias,
                              const std::vector<float> &bias_corrected,
                              const sr::RoutingResult &selected,
                              const std::map<std::string, std::vector<float>> &stages) {
  if (gate_input.size() != 2048 || router_logits.size() != 256 ||
      sigmoid_scores.size() != 256 || expert_bias.size() != 256 ||
      bias_corrected.size() != 256 || selected.expert_ids.size() != 8 ||
      selected.weights.size() != 8)
    throw std::runtime_error("route probe geometry");
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("route probe output open failed");
  out << "{\"position\":" << spec.position << ",\"layer\":"
      << spec.layer << ",\"gate_input\":";
  write_float_array(out, gate_input);
  out << ",\"router_logits\":";
  write_float_array(out, router_logits);
  out << ",\"sigmoid_scores\":";
  write_float_array(out, sigmoid_scores);
  out << ",\"expert_bias\":";
  write_float_array(out, expert_bias);
  out << ",\"bias_corrected_scores\":";
  write_float_array(out, bias_corrected);
  out << ",\"selected_expert_ids\":[";
  for (size_t i = 0; i < selected.expert_ids.size(); ++i) {
    if (i)
      out << ',';
    out << selected.expert_ids[i];
  }
  out << "],\"selected_weights\":";
  write_float_array(out, selected.weights);
  out << ",\"operator_stages\":{";
  bool first = true;
  for (const auto &entry : stages) {
    if (!first) out << ',';
    first = false;
    out << '\"' << entry.first << "\":";
    write_float_array(out, entry.second);
  }
  out << "}}\n";
  if (!out)
    throw std::runtime_error("route probe output write failed");
}
int main(int argc, char **argv) {
  try {
    if (argc < 7 || (argc - 7) % 2)
      throw std::runtime_error(
          "usage: strata_ring_bf16 ROOT MANIFEST "
          "TOKENS_CSV NEW_OUTPUT CACHE_MIB CHECKPOINT_ID "
          "[--generate N --eos CSV] [--logits-mode all|last] "
          "[--route-probe POS,LAYER] [--lm-head-residency off|cuda] [--matmul bf16-gemv|fp32-cublas] [--router native|torch-cuda] [--rms-mode native|torch-cuda] [--rope-frequency native|original] [--attention-backend native|torch-cuda] [--pointwise native|torch-cuda] [--attention-order native|normalized-first] [--attention-scale native|divide]");
    const std::filesystem::path output = argv[4];
    if (std::filesystem::exists(output))
      throw std::runtime_error("refusing existing output directory");
    const size_t cache_mib = sr::unsigned_decimal(argv[5]);
    if (cache_mib < 6 || cache_mib > 2048)
      throw std::runtime_error("cache budget must be 6..2048 MiB");
    const std::string identity = argv[6];
    if (identity.size() != 64 ||
        identity.find_first_not_of("0123456789abcdef") != std::string::npos)
      throw std::runtime_error(
          "checkpoint identity must be a SHA-256 hex digest");
    size_t generate = 0;
    std::vector<int> eos;
    std::string logits_mode = "all";
    MatmulBackend matmul_backend = MatmulBackend::Bf16Gemv;
    LmHeadResidency lm_head_residency = LmHeadResidency::Off;
    RouterBackend router_backend = RouterBackend::Native;
    RmsBackend rms_backend = RmsBackend::Native;
    AttentionBackend attention_backend = AttentionBackend::Native;
    PointwiseBackend pointwise_backend = PointwiseBackend::Native;
    std::string attention_order = "native", attention_scale = "native", rope_frequency = "native";
    bool has_route_probe = false;
    RouteProbeSpec route_probe{};
    std::filesystem::path state_in, state_out;
    size_t stop_after = std::numeric_limits<size_t>::max();
    std::string state_binding;
    std::vector<std::string> options;
    for (int i = 7; i < argc; i += 2) {
      const std::string option = argv[i];
      if (std::find(options.begin(), options.end(), option) != options.end())
        throw std::invalid_argument("duplicate option");
      options.push_back(option);
      if (option == "--generate")
        generate = sr::unsigned_decimal(argv[i + 1]);
      else if (option == "--eos")
        eos = sr::vocabulary_ids(argv[i + 1], 157184);
      else if (option == "--logits-mode")
        logits_mode = argv[i + 1];
      else if (option == "--matmul")
        matmul_backend = parse_matmul_backend(argv[i + 1]);
      else if (option == "--lm-head-residency")
        lm_head_residency = parse_lm_head_residency(argv[i + 1]);
      else if (option == "--router")
        router_backend = parse_router_backend(argv[i + 1]);
      else if (option == "--rms-mode")
        rms_backend = parse_rms_backend(argv[i + 1]);
      else if (option == "--attention-backend")
        attention_backend = parse_attention_backend(argv[i + 1]);
      else if (option == "--pointwise")
        pointwise_backend = parse_pointwise_backend(argv[i + 1]);
      else if (option == "--rope-frequency")
        rope_frequency = argv[i + 1];
      else if (option == "--attention-scale")
        attention_scale = argv[i + 1];
      else if (option == "--attention-order")
        attention_order = argv[i + 1];
      else if (option == "--route-probe") {
        route_probe = parse_route_probe(argv[i + 1]);
        has_route_probe = true;
      } else if (option == "--state-in") state_in = argv[i + 1];
      else if (option == "--state-out") state_out = argv[i + 1];
      else if (option == "--stop-after") stop_after = sr::unsigned_decimal(argv[i + 1]);
      else if (option == "--state-binding") state_binding = argv[i + 1];
      else
        throw std::invalid_argument("unknown option");
    }
#ifndef STRATA_TORCH_ROUTER
    if (rms_backend == RmsBackend::TorchCuda) throw std::runtime_error("torch-cuda RMS backend is unavailable in this build");
    if (attention_backend == AttentionBackend::TorchCuda) throw std::runtime_error("torch-cuda attention backend is unavailable in this build");
    if (pointwise_backend == PointwiseBackend::TorchCuda) throw std::runtime_error("torch-cuda pointwise backend is unavailable in this build");
#endif
    if (rope_frequency != "native" && rope_frequency != "original")
      throw std::invalid_argument("rope frequency must be native or original");
    if (attention_scale != "native" && attention_scale != "divide")
      throw std::invalid_argument("attention scale must be native or divide");
    if (attention_order != "native" && attention_order != "normalized-first")
      throw std::invalid_argument("attention order must be native or normalized-first");
    if (attention_backend == AttentionBackend::TorchCuda &&
        (attention_order != "native" || attention_scale != "native"))
      throw std::invalid_argument(
          "torch-cuda attention owns source QK/softmax/AV order; use native attention-order and attention-scale");
    if (logits_mode != "all" && logits_mode != "last")
      throw std::invalid_argument("logits mode must be all or last");
    if (generate && logits_mode != "all")
      throw std::invalid_argument(
          "generation requires all logits for validation");
    const bool continuation_mode = !state_in.empty() || !state_out.empty() ||
        stop_after != std::numeric_limits<size_t>::max() || !state_binding.empty();
    if (continuation_mode && (state_binding.size() != 64 || state_binding.find_first_not_of("0123456789abcdef") != std::string::npos))
      throw std::invalid_argument("continuation requires exact lowercase hex state binding");
    if (continuation_mode && (has_route_probe || logits_mode != "all"))
      throw std::invalid_argument("continuation requires all logits and no route probe");
    if (stop_after != std::numeric_limits<size_t>::max() && state_out.empty())
      throw std::invalid_argument("stop-after requires state-out");
    sr::GreedyGeneration generation(sr::vocabulary_ids(argv[3], 157184),
                                    generate, eos);
    if (has_route_probe && route_probe.position >= generation.kv_capacity())
      throw std::invalid_argument("route probe position exceeds evaluation window");
    std::signal(SIGINT, cancel_handler);
    std::signal(SIGTERM, cancel_handler);
    sr::TensorStore store(argv[1], argv[2]);
    const auto &embedding = store.tensor("model.word_embeddings.weight");
    shape(embedding, {157184, 2048});
    if (embedding.dtype != "BF16")
      throw std::runtime_error("original BF16 embedding required");
    const auto &lm_head = store.tensor("lm_head.weight");
    shape(lm_head, {157184, 2048});
    if (lm_head.dtype != "BF16")
      throw std::runtime_error("original BF16 lm_head required");
    check(cudaSetDevice(0));
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0));
    if (props.major != 8 || props.minor != 9)
      throw std::runtime_error("this target is qualified only for sm_89");
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total));
    int cuda_runtime_version = 0; check(cudaRuntimeGetVersion(&cuda_runtime_version));
    const size_t budget = cache_mib * 1024 * 1024;
    constexpr size_t memory_margin = size_t(512) * 1024 * 1024;
    const size_t lm_head_residency_bytes =
        lm_head_residency == LmHeadResidency::Cuda ? lm_head.bytes : 0;
    const size_t required = budget + Matvec::workspace_bytes(matmul_backend) +
                            memory_margin + lm_head_residency_bytes;
    if (free < required) {
      if (lm_head_residency == LmHeadResidency::Cuda)
        throw std::runtime_error("insufficient free GPU memory for requested lm-head residency; no process will be stopped");
      throw std::runtime_error("insufficient free GPU memory; no process will be stopped");
    }
    Matvec mv(matmul_backend);
    if (lm_head_residency == LmHeadResidency::Cuda) {
      try { mv.pin_lm_head(lm_head); }
      catch (const std::exception &e) {
        throw std::runtime_error(std::string("requested lm-head residency initialization failed: ") + e.what());
      }
    }
#ifdef STRATA_TORCH_ROUTER
    std::unique_ptr<strata::torch_router::Router> torch_router;
    std::unique_ptr<strata::torch_router::Rms> torch_rms;
    std::unique_ptr<strata::torch_router::Attention> torch_attention;
    std::unique_ptr<strata::torch_router::Pointwise> torch_pointwise;
    if (router_backend == RouterBackend::TorchCuda) torch_router = std::make_unique<strata::torch_router::Router>(256, 8, 4, 8, 2.5f, "cuda");
    if (rms_backend == RmsBackend::TorchCuda) torch_rms = std::make_unique<strata::torch_router::Rms>("cuda");
    if (attention_backend == AttentionBackend::TorchCuda) torch_attention = std::make_unique<strata::torch_router::Attention>("cuda");
    if (pointwise_backend == PointwiseBackend::TorchCuda) torch_pointwise = std::make_unique<strata::torch_router::Pointwise>("cuda");
#else
    if (router_backend == RouterBackend::TorchCuda) throw std::runtime_error("torch-cuda router backend is unavailable in this build");
#endif
    auto backend = std::make_shared<CudaBackend>();
    sr::ExpertCache cache({budget, 20, 256}, backend);
    sr::ExpertCache::Provider provider = [&](const sr::ExpertKey &key,
                                             sr::ExpertPayload &p,
                                             std::string &error) {
      try {
        const std::string prefix = "model.layers." + std::to_string(key.layer) +
                                   ".mlp.experts." +
                                   std::to_string(key.expert) + ".";
        const auto &g = store.tensor(prefix + "gate_proj.weight");
        const auto &u = store.tensor(prefix + "up_proj.weight");
        const auto &d = store.tensor(prefix + "down_proj.weight");
        shape(g, {512, 2048});
        shape(u, {512, 2048});
        shape(d, {2048, 512});
        if (g.dtype != "BF16" || u.dtype != "BF16" || d.dtype != "BF16")
          throw std::runtime_error("expert is not BF16");
        p.gate_shape = p.up_shape = {512, 2048};
        p.down_shape = {2048, 512};
        p.gate.assign(g.data, g.data + g.bytes);
        p.up.assign(u.data, u.data + u.bytes);
        p.down.assign(d.data, d.data + d.bytes);
        return true;
      } catch (const std::exception &ex) {
        error = ex.what();
        return false;
      }
    };
    std::vector<sr::KvCache> kv;
    for (int i = 0; i < 20; ++i)
      kv.emplace_back(generation.kv_capacity(), 4, 128);
    std::vector<float> last_logits;
    sr::AttentionConfig attention;
    if (!state_in.empty()) {
      auto state = sr::load_continuation(state_in);
      sr::require_continuation(state, state_binding, generation.prompt(), generation.kv_capacity());
      if (state.arrays.size() != 40) throw std::runtime_error("continuation Ring KV array set mismatch");
      std::vector<sr::KvCache> staged_kv;
      for (int l=0;l<20;++l) staged_kv.emplace_back(generation.kv_capacity(),4,128);
      for (int l=0;l<20;++l) {
        const auto kname="kv."+std::to_string(l)+".k", vname="kv."+std::to_string(l)+".v";
        auto ki=state.arrays.find(kname), vi=state.arrays.find(vname);
        if (ki==state.arrays.end() || vi==state.arrays.end()) throw std::runtime_error("continuation Ring KV arrays missing");
        std::string restore_error;
        if (!staged_kv[l].restore_live(static_cast<int64_t>(state.position),ki->second,vi->second,restore_error)) throw std::runtime_error(restore_error);
      }
      // Do not alter the live caches until every layer has been validated.
      auto staged_generation = generation;
      staged_generation.resume_prefix(state.prefix, state.last_logits);
      auto staged_logits = state.last_logits;
      kv.swap(staged_kv);
      generation = std::move(staged_generation);
      last_logits.swap(staged_logits);
    }
    const auto &tokens = generation.evaluated();
    const size_t start_position = state_in.empty() ? 0 : kv[0].length();
    if (stop_after < start_position) throw std::invalid_argument("stop-after precedes restored position");
    attention.rope_dim = 64;
    attention.original_rope_frequency = rope_frequency == "original";
    const sr::RouterConfig routing{256, 8, 4, 8, 2.5f};
    std::filesystem::create_directories(output);
    std::ofstream routes(output / "routes.tsv");
    routes << std::setprecision(9);
    const auto start = std::chrono::steady_clock::now();
    bool route_probe_found = false;
    auto apply_norm = [&](const std::vector<float> &x, const sr::TensorView &weight) {
      if (rms_backend == RmsBackend::Native) return norm(x, weight);
#ifdef STRATA_TORCH_ROUTER
      shape(weight, {x.size()});
      std::vector<float> w = floats(weight), out; std::string e;
      if (!torch_rms->normalize(x.data(), w.data(), 1, x.size(), attention.rms_eps, out, e)) throw std::runtime_error(e);
      return out;
#else
      throw std::runtime_error("torch-cuda RMS backend is unavailable in this build");
#endif
    };
    auto apply_swiglu = [&](std::vector<float> gate, const std::vector<float> &up) {
      if (pointwise_backend == PointwiseBackend::Native)
        return swiglu(std::move(gate), up);
#ifdef STRATA_TORCH_ROUTER
      std::vector<float> out;
      std::string e;
      if (!torch_pointwise->swiglu(gate.data(), up.data(), gate.size(), out, e))
        throw std::runtime_error(e);
      return out;
#else
      throw std::runtime_error("torch-cuda pointwise backend is unavailable in this build");
#endif
    };
    std::vector<float> probe_gate_input, probe_router_logits, probe_sigmoid,
        probe_bias, probe_corrected;
    sr::RoutingResult probe_selected;
    std::map<std::string, std::vector<float>> probe_stages;
    size_t processed_position = start_position;
    for (size_t pos = start_position; pos < tokens.size() && pos < stop_after; ++pos) {
      processed_position = pos + 1;
      check_cancelled();
      std::vector<float> h(2048);
      for (size_t i = 0; i < h.size(); ++i) {
        uint16_t b;
        std::memcpy(&b, embedding.data + (size_t(tokens[pos]) * 2048 + i) * 2,
                    2);
        h[i] = bf16(b);
      }
      for (int layer = 0; layer < 20; ++layer) {
        check_cancelled();
        const std::string pre = "model.layers." + std::to_string(layer) + ".";
        const bool capture_operator = has_route_probe && route_probe.position == pos && route_probe.layer == static_cast<size_t>(layer);
        if (capture_operator) probe_stages["input_hidden"] = h;
        auto a = apply_norm(h, store.tensor(pre + "input_layernorm.weight"));
        if (capture_operator) probe_stages["input_normalized"] = a;
        const auto &qkv_weight =
            store.tensor(pre + "attention.query_key_value.weight");
        shape(qkv_weight, {3072, 2048});
        auto qkv = mv.apply(qkv_weight, a);
        if (capture_operator) probe_stages["qkv_raw"] = qkv;
        const auto &qwt = store.tensor(pre + "attention.query_layernorm.weight");
        const auto &kwt = store.tensor(pre + "attention.key_layernorm.weight");
        shape(qwt, {128}); shape(kwt, {128});
        auto qw = floats(qwt);
        auto kw = floats(kwt);
        std::vector<float> pre_q, pre_k;
        const float *qptr = qkv.data(), *kptr = qkv.data() + 2048;
#ifdef STRATA_TORCH_ROUTER
        if (rms_backend == RmsBackend::TorchCuda) {
          std::string e;
          if (!torch_rms->normalize(qptr, qw.data(), 16, 128, attention.rms_eps, pre_q, e) ||
              !torch_rms->normalize(kptr, kw.data(), 4, 128, attention.rms_eps, pre_k, e)) throw std::runtime_error(e);
          qptr = pre_q.data(); kptr = pre_k.data();
        }
#else
        if (rms_backend == RmsBackend::TorchCuda) throw std::runtime_error("torch-cuda RMS backend is unavailable in this build");
#endif
        if (qw.size() != 128 || kw.size() != 128)
          throw std::runtime_error("QK norm geometry");
        std::vector<float> context(2048);
        std::string error;
        sr::AttentionTrace attention_trace;
        const auto value_order = attention_order == "normalized-first" ? sr::AttentionValueOrder::NormalizeWeightsFirst : sr::AttentionValueOrder::AccumulateUnnormalized;
        if (!sr::attention_forward_order(attention, qptr, kptr,
                                   qkv.data() + 2560, 1, pos, qw.data(),
                                   kw.data(), kv[layer], context.data(), error,
                                   value_order, capture_operator ? &attention_trace : nullptr,
                                   attention_scale == "divide" ? sr::AttentionScoreScale::DivideBySqrt : sr::AttentionScoreScale::ReciprocalSqrt,
                                   rms_backend == RmsBackend::TorchCuda,
#ifdef STRATA_TORCH_ROUTER
                                   attention_backend == AttentionBackend::TorchCuda ? torch_attention.get() : nullptr))
#else
                                   nullptr))
#endif
          throw std::runtime_error(error);
        auto projected = mv.apply(store.tensor(pre + "attention.dense.weight"), context);
        if (capture_operator) {
          probe_stages["query_normalized"] = attention_trace.normalized_q;
          probe_stages["key_normalized"] = attention_trace.normalized_k;
          probe_stages["query_rotated"] = attention_trace.rotated_q;
          probe_stages["key_rotated"] = attention_trace.rotated_k;
          probe_stages["cache_k"] = attention_trace.cache_k;
          probe_stages["cache_v"] = attention_trace.cache_v;
          probe_stages["attention_scores"] = attention_trace.scores;
          probe_stages["attention_probabilities"] = attention_trace.probabilities;
          probe_stages["attention_context"] = context;
          probe_stages["attention_projected"] = projected;
        }
        add(h, projected);
        if (capture_operator) probe_stages["post_attention_hidden"] = h;
        auto x = apply_norm(h, store.tensor(pre + "post_attention_layernorm.weight"));
        if (capture_operator) probe_stages["gate_input"] = x;
        if (layer == 0) {
          if (pointwise_backend == PointwiseBackend::Native)
            add(h, mlp(mv, store, pre + "mlp.", x));
          else {
            auto g = mv.apply(store.tensor(pre + "mlp.gate_proj.weight"), x);
            auto u = mv.apply(store.tensor(pre + "mlp.up_proj.weight"), x);
            add(h, mv.apply(store.tensor(pre + "mlp.down_proj.weight"),
                            apply_swiglu(std::move(g), u)));
          }
        } else {
          auto logits = mv.apply(store.tensor(pre + "mlp.gate.weight"), x);
          auto bias = floats(store.tensor(pre + "mlp.gate.expert_bias"));
          if (logits.size() != 256 || bias.size() != 256)
            throw std::runtime_error("router geometry");
          sr::RoutingResult selected;
          std::vector<float> backend_scores, backend_corrected;
#ifdef STRATA_TORCH_ROUTER
          if (router_backend == RouterBackend::TorchCuda) {
            std::string router_error;
            if (!torch_router->route(logits.data(), bias.data(), 256, selected.expert_ids, selected.weights, backend_scores, backend_corrected, router_error))
              throw std::runtime_error(router_error);
          } else
#endif
            selected = sr::route(logits.data(), bias.data(), 256, routing);
          if (has_route_probe && route_probe.position == pos &&
              route_probe.layer == static_cast<size_t>(layer)) {
            route_probe_found = true;
            probe_gate_input = x;
            probe_router_logits = logits;
            probe_bias = bias;
            probe_sigmoid.resize(logits.size());
            probe_corrected.resize(logits.size());
            for (size_t i = 0; i < logits.size(); ++i) {
#ifdef STRATA_TORCH_ROUTER
              if (router_backend == RouterBackend::TorchCuda) { probe_sigmoid[i] = backend_scores[i]; probe_corrected[i] = backend_corrected[i]; }
              else
#endif
              { probe_sigmoid[i] = router_sigmoid(logits[i]); probe_corrected[i] = probe_sigmoid[i] + bias[i]; }
            }
            probe_selected = selected;
          }
          std::vector<float> mixed(2048, 0), expert_outputs;
          if (pointwise_backend == PointwiseBackend::TorchCuda)
            expert_outputs.reserve(8 * 2048);
          routes << pos << '\t' << layer;
          for (size_t j = 0; j < 8; ++j) {
            const auto expert = selected.expert_ids[j];
            routes << '\t' << expert << ':' << selected.weights[j];
            auto lease = cache.acquire(
                {identity, uint32_t(layer), uint32_t(expert)}, provider, error);
            if (!lease)
              throw std::runtime_error(error);
            auto device = std::dynamic_pointer_cast<const ExpertDevice>(
                lease->backend_value());
            if (!device)
              throw std::runtime_error("wrong backend payload");
            const auto *w = static_cast<const uint16_t *>(device->weights.p);
            constexpr size_t count = 512 * 2048;
            auto g = mv.apply_device(w, 512, 2048, x);
            auto u = mv.apply_device(w + count, 512, 2048, x);
            auto y = mv.apply_device(w + 2 * count, 2048, 512,
                                     apply_swiglu(std::move(g), u));
            if (pointwise_backend == PointwiseBackend::TorchCuda)
              expert_outputs.insert(expert_outputs.end(), y.begin(), y.end());
            else
              for (size_t d = 0; d < mixed.size(); ++d)
                mixed[d] += y[d] * selected.weights[j];
          }
          if (pointwise_backend == PointwiseBackend::TorchCuda) {
#ifdef STRATA_TORCH_ROUTER
            if (!torch_pointwise->weighted_sum(expert_outputs.data(), selected.weights.data(), 8, 2048, mixed, error))
              throw std::runtime_error(error);
#else
            throw std::runtime_error("torch-cuda pointwise backend is unavailable in this build");
#endif
          }
          routes << '\n';
          if (pointwise_backend == PointwiseBackend::Native)
            add(mixed, mlp(mv, store, pre + "mlp.shared_experts.", x));
          else {
            auto shared_gate = mv.apply(store.tensor(pre + "mlp.shared_experts.gate_proj.weight"), x);
            auto shared_up = mv.apply(store.tensor(pre + "mlp.shared_experts.up_proj.weight"), x);
            add(mixed, mv.apply(store.tensor(pre + "mlp.shared_experts.down_proj.weight"),
                                apply_swiglu(std::move(shared_gate), shared_up)));
          }
          add(h, mixed);
        }
        if (capture_operator) probe_stages["layer_output"] = h;
        write_row(output / ("layer-" + std::to_string(layer) + ".f32"), h);
      }
      h = apply_norm(h, store.tensor("model.norm.weight"));
      write_row(output / "final-hidden.f32", h);
      if (logits_mode == "all" || pos + 1 == tokens.size()) {
        auto logits = lm_head_residency == LmHeadResidency::Cuda
                          ? mv.apply_lm_head(h) : mv.apply(lm_head, h);
        write_row(output / "logits.f32", logits);
        const auto top = std::distance(
            logits.begin(), std::max_element(logits.begin(), logits.end()));
        if (continuation_mode) last_logits = logits;
        if (generation.sampling_position(pos) && pos + 1 < stop_after)
          generation.sample(pos, logits);
        std::cerr << "token " << pos << " complete; top1 " << top << '\n';
      } else if (pos % 64 == 0)
        std::cerr << "prefill " << pos << '\n';
    }
    check_cancelled();
    if (has_route_probe) {
      if (!route_probe_found)
        throw std::runtime_error("route probe position was not evaluated");
      write_route_probe(output / "route-probe.json", route_probe,
                        probe_gate_input, probe_router_logits, probe_sigmoid,
                        probe_bias, probe_corrected, probe_selected, probe_stages);
    }
    if (continuation_mode && processed_position == start_position) {
      for (int layer=0;layer<20;++layer) {
        std::ofstream empty(output/("layer-"+std::to_string(layer)+".f32"),std::ios::binary);
        if(!empty) throw std::runtime_error("empty state trace write failed");
      }
      for(const char* name:{"final-hidden.f32","logits.f32"}) {
        std::ofstream empty(output/name,std::ios::binary);
        if(!empty) throw std::runtime_error("empty state trace write failed");
      }
    }
    const bool paused = processed_position < tokens.size() ||
        (generate && generation.finish_reason().empty());
    if (generate && !paused) {
      std::ofstream generated(output / "generation.json");
      generated << generation.receipt();
      generated.flush();
      if (!generated)
        throw std::runtime_error("generation receipt write failed");
    }
    if (!state_out.empty()) {
      if (last_logits.empty() && processed_position) throw std::runtime_error("continuation has no last logits");
      sr::ContinuationState state{state_binding, processed_position,
          std::vector<int>(tokens.begin(), tokens.begin()+processed_position), last_logits, {}};
      for (int l=0;l<20;++l) { state.arrays.emplace("kv."+std::to_string(l)+".k",kv[l].live_keys()); state.arrays.emplace("kv."+std::to_string(l)+".v",kv[l].live_values()); }
      sr::save_continuation(state_out,state);
    }
    if (continuation_mode) {
      std::ofstream side(output / "continuation.json");
      side << "{\"start_position\":" << start_position
           << ",\"processed_position\":" << processed_position
           << ",\"processed_tokens\":"
           << (processed_position - start_position)
           << ",\"paused\":" << (paused ? "true" : "false")
           << ",\"state_in\":" << (!state_in.empty() ? "true" : "false")
           << ",\"state_out\":" << (!state_out.empty() ? "true" : "false")
           << "}\n";
      side.flush();
      if (!side)
        throw std::runtime_error("continuation sidecar write failed");
    }
    auto stats = cache.stats();
    routes.flush();
    if (!routes)
      throw std::runtime_error("route trace write failed");
    std::ostringstream token_json;
    token_json << '[';
    for (size_t i = 0; i < (continuation_mode ? processed_position : tokens.size()); ++i) {
      if (i)
        token_json << ',';
      token_json << tokens[i];
    }
    token_json << ']';
    std::ostringstream receipt;
    receipt << std::setprecision(12)
            << "{\"status\":\"executed\",\"model\":\"Ring-mini-2.0\","
               "\"activation_dtype\":\"FP32\",\"quantized\":false,\"tokens\":"
            << processed_position << ",\"token_ids\":" << token_json.str()
            << ",\"logits_mode\":\"" << logits_mode << "\",\"logits_tokens\":"
            << (continuation_mode ? processed_position-start_position : (logits_mode == "all" ? tokens.size() : 1))
            << ",\"checkpoint_id\":\"" << identity
            << "\",\"layers\":20,\"cache_budget_bytes\":" << budget
            << ",\"cache_resident_bytes\":" << stats.bytes
            << ",\"cache_device_payload_bytes\":" << stats.bytes
            << ",\"cache_host_payload_bytes\":" << stats.bytes
            << ",\"cache_payload_scope\":\"resident payload only; excludes "
               "allocator overhead, one staged expert, mmap pages and other "
               "working buffers\""
            << ",\"host_kv_capacity_bytes\":"
            << size_t(20) * generation.kv_capacity() * 4 * 128 * 2 *
                   sizeof(float)
            << ",\"cache_hits\":" << stats.hits
            << ",\"cache_misses\":" << stats.misses
            << ",\"cache_evictions\":" << stats.evictions
            << ",\"rope_frequency\":\"" << rope_frequency << "\""
            << ",\"attention_value_order\":\"" << attention_order << "\""
            << ",\"attention_score_scale\":\"" << attention_scale << "\""
            << ",\"attention_backend\":\"" << attention_backend_name(attention_backend) << "\""
            << ",\"pointwise_backend\":\"" << pointwise_backend_name(pointwise_backend) << "\""
            << ",\"matmul_backend\":\""
            << matmul_backend_name(matmul_backend)
            << "\",\"lm_head_residency\":\"" << lm_head_residency_name(lm_head_residency)
            << "\",\"lm_head_resident_bytes\":" << mv.resident_lm_head_bytes()
            << ",\"lm_head_resident_uploads\":" << mv.resident_lm_head_uploads()
            << ",\"lm_head_resident_apply_calls\":" << mv.resident_lm_head_apply_calls()
            << ",\"router_backend\":\"" << router_backend_name(router_backend) << "\",\"rms_backend\":\"" << rms_backend_name(rms_backend) << "\",\"cuda_runtime_version\":" << cuda_runtime_version << ",\"matmul_fp32_weight_scratch_bytes\":"
            << Matvec::fp32_scratch_bytes(matmul_backend)
            << ",\"workspace_bytes\":"
            << Matvec::workspace_bytes(matmul_backend)
            << ",\"wall_seconds\":"
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             start)
                   .count()
            << "}\n";
    std::ofstream receipt_file(output / "execution.json");
    receipt_file << receipt.str();
    receipt_file.flush();
    if (!receipt_file)
      throw std::runtime_error("execution receipt write failed");
    std::cout << receipt.str();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
