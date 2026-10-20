// AI Identity Incorported <james@aiii.id>
// Opt-in Ling BF16 reference executor. Original weights, FP32 working state.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <strata/kernels/bf16_gemv.hpp>
#include <strata/ling/kda.hpp>
#include <strata/ling/mla.hpp>
#include <strata/ling/rms.hpp>
#include <strata/ring/continuation.hpp>
#include <strata/ring/expert_cache.hpp>
#include <strata/ring/generation.hpp>
#include <strata/ring/numerics.hpp>
#include <strata/ring/routing.hpp>
#include <strata/ring/tensor_store.hpp>
#include <vector>
#include <map>
#ifdef STRATA_LING_TORCH_OPS
#include "ling_torch_ops.hpp"
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
  if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS operation failed");
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
                               const sr::TensorView &weight,
                               strata::ling::RmsMode mode
#ifdef STRATA_LING_TORCH_OPS
                               , strata::ling_torch::Ops *torch_ops
#endif
                               ) {
  shape(weight, {x.size()});
#ifdef STRATA_LING_TORCH_OPS
  if (torch_ops) { std::vector<float> output; std::string error; if (!torch_ops->rms_norm(x, floats(weight), output, error)) throw std::runtime_error(error); return output; }
#endif
  return strata::ling::rms_norm(x, floats(weight), mode);
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
static const char *matmul_backend_name(MatmulBackend b) { return b == MatmulBackend::Fp32Cublas ? "fp32-cublas" : "bf16-gemv"; }
static MatmulBackend parse_matmul_backend(const std::string &v) {
  if (v == "bf16-gemv") return MatmulBackend::Bf16Gemv;
  if (v == "fp32-cublas") return MatmulBackend::Fp32Cublas;
  throw std::invalid_argument("matmul backend must be bf16-gemv or fp32-cublas");
}
__global__ static void widen_bf16(const uint16_t *src, float *dst, size_t count) {
  const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i<count) dst[i]=__uint_as_float(uint32_t(src[i])<<16);
}
class Matvec {
  static constexpr size_t max_weight_bytes=size_t(157184)*1536*2;
  DeviceBuffer weight{max_weight_bytes}, x{4608*4}, y{157184*4};
  MatmulBackend backend_; std::unique_ptr<DeviceBuffer> fp32_weight;
  // Separate opt-in original-BF16 LM-head storage; matrix arithmetic remains
  // apply_device(), so FP32 widening and pedantic cuBLAS order are unchanged.
  std::unique_ptr<DeviceBuffer> resident_lm_head;
  size_t resident_lm_head_bytes_=0, resident_lm_head_uploads_=0, resident_lm_head_apply_calls_=0;
  cublasHandle_t cublas=nullptr;
public:
  static constexpr size_t fp32_weight_bytes=size_t(157184)*1536*4;
  static constexpr size_t base_workspace_bytes=max_weight_bytes+4608*4+157184*4;
  static constexpr size_t workspace_bytes(MatmulBackend b) { return base_workspace_bytes+(b==MatmulBackend::Fp32Cublas?fp32_weight_bytes:0); }
  static constexpr size_t fp32_scratch_bytes(MatmulBackend b) { return b==MatmulBackend::Fp32Cublas?fp32_weight_bytes:0; }
  explicit Matvec(MatmulBackend b):backend_(b) {
    if (backend_==MatmulBackend::Fp32Cublas) {
      fp32_weight=std::make_unique<DeviceBuffer>(fp32_weight_bytes);
      check_cublas(cublasCreate(&cublas)); check_cublas(cublasSetMathMode(cublas,CUBLAS_PEDANTIC_MATH));
      check_cublas(cublasSetAtomicsMode(cublas,CUBLAS_ATOMICS_NOT_ALLOWED));
    }
  }
  ~Matvec() { if(cublas) cublasDestroy(cublas); }
  std::vector<float> apply_device(const uint16_t *w,size_t rows,size_t cols,const std::vector<float> &input) {
    if(input.size()!=cols || cols==0 || rows==0 || cols>4608 || rows>157184 || cols%2 || rows > max_weight_bytes / 2 / cols) throw std::runtime_error("matvec geometry");
    check(cudaMemcpy(x.p,input.data(),cols*4,cudaMemcpyHostToDevice));
    if(backend_==MatmulBackend::Bf16Gemv) {
      strata::kernels::bf16_gemv_fp32_mmvf(static_cast<float*>(x.p),w,static_cast<float*>(y.p),cols,rows,nullptr);
      check(cudaGetLastError());
    } else {
      if(!fp32_weight || !cublas) throw std::runtime_error("cuBLAS backend is not initialized");
      const size_t count=rows*cols;
      widen_bf16<<<static_cast<int>((count+255)/256),256>>>(w,static_cast<float*>(fp32_weight->p),count);
      check(cudaGetLastError()); const float alpha=1.0f,beta=0.0f;
      check_cublas(cublasSgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,static_cast<int>(rows),1,static_cast<int>(cols),&alpha,
        static_cast<const float*>(fp32_weight->p),static_cast<int>(cols),static_cast<const float*>(x.p),static_cast<int>(cols),&beta,static_cast<float*>(y.p),static_cast<int>(rows)));
    }
    std::vector<float> output(rows); check(cudaMemcpy(output.data(),y.p,rows*4,cudaMemcpyDeviceToHost)); finite(output); return output;
  }
  std::vector<float> apply(const sr::TensorView &t,const std::vector<float> &input) {
    if(t.shape.size()!=2 || t.shape[0]==0 || t.shape[1]==0 || t.shape[1]!=input.size() || t.dtype!="BF16" || t.bytes>weight.bytes || t.shape[0] > max_weight_bytes / 2 / t.shape[1]) throw std::runtime_error("expected compatible original BF16 matrix");
    check(cudaMemcpy(weight.p,t.data,t.bytes,cudaMemcpyHostToDevice)); return apply_device(static_cast<uint16_t*>(weight.p),t.shape[0],t.shape[1],input);
  }
  void pin_lm_head(const sr::TensorView &t) {
    if(resident_lm_head || t.shape != std::vector<size_t>{157184,1536} || t.dtype!="BF16" || t.bytes!=max_weight_bytes)
      throw std::runtime_error("lm-head residency requires original BF16 [157184,1536]");
    resident_lm_head=std::make_unique<DeviceBuffer>(t.bytes);
    check(cudaMemcpy(resident_lm_head->p,t.data,t.bytes,cudaMemcpyHostToDevice));
    resident_lm_head_bytes_=t.bytes; ++resident_lm_head_uploads_;
  }
  std::vector<float> apply_lm_head(const std::vector<float> &input) {
    if(!resident_lm_head) throw std::runtime_error("lm-head residency was requested but is not initialized");
    ++resident_lm_head_apply_calls_;
    return apply_device(static_cast<uint16_t*>(resident_lm_head->p),157184,1536,input);
  }
  size_t resident_lm_head_bytes() const { return resident_lm_head_bytes_; }
  size_t resident_lm_head_uploads() const { return resident_lm_head_uploads_; }
  size_t resident_lm_head_apply_calls() const { return resident_lm_head_apply_calls_; }
};
static std::vector<float> swiglu(std::vector<float> g,
                                 const std::vector<float> &u
#ifdef STRATA_LING_TORCH_OPS
                                 , strata::ling_torch::Ops *torch_ops
#endif
                                 ) {
  if (g.size() != u.size())
    throw std::runtime_error("SwiGLU geometry");
#ifdef STRATA_LING_TORCH_OPS
  if (torch_ops) {
    std::vector<float> output; std::string error;
    if (!torch_ops->swiglu(g, u, output, error)) throw std::runtime_error(error);
    return output;
  }
#endif
  for (size_t i = 0; i < g.size(); ++i)
    g[i] = (g[i] / (1 + std::exp(-g[i]))) * u[i];
  finite(g);
  return g;
}
static std::vector<float> mlp(Matvec &mv, const sr::TensorStore &store,
                              const std::string &prefix,
                              const std::vector<float> &x
#ifdef STRATA_LING_TORCH_OPS
                              , strata::ling_torch::Ops *torch_ops
#endif
                              ) {
  auto g = mv.apply(store.tensor(prefix + "gate_proj.weight"), x);
  auto u = mv.apply(store.tensor(prefix + "up_proj.weight"), x);
  return mv.apply(store.tensor(prefix + "down_proj.weight"),
                  swiglu(std::move(g), u
#ifdef STRATA_LING_TORCH_OPS
                         , torch_ops
#endif
                         ));
}
static void write_row(const std::filesystem::path &p,
                      const std::vector<float> &x) {
  std::ofstream out(p, std::ios::binary | std::ios::app);
  if (!out.write(reinterpret_cast<const char *>(x.data()), x.size() * 4))
    throw std::runtime_error("output write failed");
}

static void write_float_array(std::ostream &out, const std::vector<float> &values) {
  out << '[' << std::setprecision(9);
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}

struct RouteProbe {
  size_t position;
  size_t layer;
};

static RouteProbe parse_route_probe(const std::string &text) {
  if (text.empty() || text.find(',') == std::string::npos ||
      text.find(',', text.find(',') + 1) != std::string::npos)
    throw std::invalid_argument("route probe must be POS,LAYER");
  const size_t comma = text.find(',');
  const size_t position = sr::unsigned_decimal(text.substr(0, comma));
  const size_t layer = sr::unsigned_decimal(text.substr(comma + 1));
  if (position >= 1024 || layer == 0 || layer >= 24)
    throw std::invalid_argument("route probe position/layer out of range");
  return {position, layer};
}

static void write_route_probe(const std::filesystem::path &path,
                              const RouteProbe &probe,
                              const std::vector<float> &input_hidden,
                              const std::vector<float> &post_attention_hidden,
                              const std::vector<float> &gate_input,
                              const std::vector<float> &router_logits,
                              const std::vector<float> &sigmoid_scores,
                              const std::vector<float> &expert_bias,
                              const std::vector<float> &corrected,
                              const std::vector<size_t> &ids,
                              const std::vector<float> &weights) {
  if (input_hidden.size() != 1536 || post_attention_hidden.size() != 1536 ||
      gate_input.size() != 1536 || router_logits.size() != 128 ||
      sigmoid_scores.size() != 128 || expert_bias.size() != 128 ||
      corrected.size() != 128 || ids.size() != 8 || weights.size() != 8)
    throw std::runtime_error("route probe geometry");
  std::ofstream out(path);
  if (!out) throw std::runtime_error("route probe output open failed");
  out << std::setprecision(9)
      << "{\"position\":" << probe.position << ",\"layer\":" << probe.layer
      << ",\"score_reconstruction\":\"native routing sigmoid and bias correction recomputed from the same gate output\","
      << "\"input_hidden\":";
  write_float_array(out, input_hidden);
  out << ",\"post_attention_hidden\":";
  write_float_array(out, post_attention_hidden);
  out << ",\"gate_input\":";
  write_float_array(out, gate_input);
  out << ",\"router_logits\":";
  write_float_array(out, router_logits);
  out << ",\"sigmoid_scores\":";
  write_float_array(out, sigmoid_scores);
  out << ",\"expert_bias\":";
  write_float_array(out, expert_bias);
  out << ",\"bias_corrected_scores\":";
  write_float_array(out, corrected);
  out << ",\"selected_expert_ids\":[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) out << ',';
    out << ids[i];
  }
  out << "],\"selected_weights\":";
  write_float_array(out, weights);
  out << "}" << '\n';
  if (!out) throw std::runtime_error("route probe output failed");
}

namespace sl = strata::ling;
static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
struct KdaLayer {
  sl::KdaConfig config;
  sl::KdaState state{config};
  std::vector<float> qc, kc, vc, alog, dt, output_norm;
  KdaLayer(const sr::TensorStore &store, const std::string &pre) {
    qc = floats(store.tensor(pre + "q_conv1d.weight"));
    kc = floats(store.tensor(pre + "k_conv1d.weight"));
    vc = floats(store.tensor(pre + "v_conv1d.weight"));
    alog = floats(store.tensor(pre + "A_log"));
    dt = floats(store.tensor(pre + "dt_bias"));
    output_norm = floats(store.tensor(pre + "o_norm.weight"));
    if (qc.size()!=8192 || kc.size()!=8192 || vc.size()!=8192 ||
        alog.size()!=16 || dt.size()!=2048 || output_norm.size()!=128)
      throw std::runtime_error("KDA metadata geometry");
    for (auto *v : {&qc,&kc,&vc,&alog,&dt,&output_norm}) finite(*v);
  }
  std::vector<float> step(Matvec &mv, const sr::TensorStore &store,
                          const std::string &pre, const std::vector<float> &x
#ifdef STRATA_LING_TORCH_OPS
                          , strata::ling_torch::Ops *torch_ops
#endif
                          ) {
    auto q=mv.apply(store.tensor(pre+"q_proj.weight"),x);
    auto k=mv.apply(store.tensor(pre+"k_proj.weight"),x);
    auto v=mv.apply(store.tensor(pre+"v_proj.weight"),x);
    auto g=mv.apply(store.tensor(pre+"f_proj.weight"),x);
    auto beta=mv.apply(store.tensor(pre+"b_proj.weight"),x);
    if(q.size()!=2048 || k.size()!=2048 || v.size()!=2048 || g.size()!=2048 || beta.size()!=16)
      throw std::runtime_error("KDA projection geometry");
#ifdef STRATA_LING_TORCH_OPS
    if (torch_ops) { std::vector<float> result; std::string error; if (!torch_ops->sigmoid(beta, result, error)) throw std::runtime_error(error); beta=std::move(result); }
    else
#endif
      for(auto &f:beta) f=sigmoid(f);
    sl::KdaWeights weights{qc.data(),kc.data(),vc.data(),nullptr,nullptr,nullptr,alog.data(),dt.data()};
    std::vector<float> o(2048);
    std::string error;
    if(!sl::kda_forward(config,q.data(),k.data(),v.data(),g.data(),beta.data(),1,weights,state,o.data(),error))
      throw std::runtime_error(error);
    auto gate=mv.apply(store.tensor(pre+"g_proj.weight"),x);
    if(gate.size()!=2048) throw std::runtime_error("KDA output gate geometry");
 #ifdef STRATA_LING_TORCH_OPS
    if (torch_ops) { std::vector<float> result; std::string error; if (!torch_ops->kda_output_norm_gate(o, output_norm, gate, 16, 128, result, error)) throw std::runtime_error(error); o=std::move(result); }
    else
 #endif
    for(size_t head=0;head<16;++head) {
      std::vector<float> part(o.begin()+head*128,o.begin()+(head+1)*128);
      part=sr::rms_norm(part,output_norm);
      for(size_t i=0;i<128;++i) o[head*128+i]=part[i]*sigmoid(gate[head*128+i]);
    }
    return mv.apply(store.tensor(pre+"o_proj.weight"),o);
  }
};
static std::vector<float> mla_step(Matvec &mv, const sr::TensorStore &store,
                                  const std::string &pre, const std::vector<float> &x,
                                  sl::MlaCache &cache, size_t position,
                                  sl::RmsMode rms_mode
#ifdef STRATA_LING_TORCH_OPS
                                  , strata::ling_torch::Ops *torch_ops
#endif
                                  ) {
  auto qa=mv.apply(store.tensor(pre+"q_a_proj.weight"),x);
  qa=norm(qa,store.tensor(pre+"q_a_layernorm.weight"),rms_mode
#ifdef STRATA_LING_TORCH_OPS
          , torch_ops
#endif
          );
  auto q=mv.apply(store.tensor(pre+"q_b_proj.weight"),qa);
  auto compressed=mv.apply(store.tensor(pre+"kv_a_proj_with_mqa.weight"),x);
  if(compressed.size()!=576) throw std::runtime_error("MLA compressed KV geometry");
  std::vector<float> low(compressed.begin(),compressed.begin()+512);
  std::vector<float> rope(compressed.begin()+512,compressed.end());
  low=norm(low,store.tensor(pre+"kv_a_layernorm.weight"),rms_mode
#ifdef STRATA_LING_TORCH_OPS
           , torch_ops
#endif
           );
  auto expanded=mv.apply(store.tensor(pre+"kv_b_proj.weight"),low);
  if(expanded.size()!=4096) throw std::runtime_error("MLA expanded KV geometry");
  std::vector<float> key(2048),value(2048);
  for(size_t h=0;h<16;++h) for(size_t i=0;i<128;++i) {
    key[h*128+i]=expanded[h*256+i]; value[h*128+i]=expanded[h*256+128+i];
  }
  auto gates=mv.apply(store.tensor(pre+"g_proj.weight"),x);
  auto context=cache.step(q,key,rope,value,gates,position);
  return mv.apply(store.tensor(pre+"dense.weight"),context);
}
static std::string kda_state_name(size_t layer, const char *part) {
  return "kda." + std::to_string(layer) + "." + part;
}
static std::string mla_state_name(size_t layer, const char *part) {
  return "mla." + std::to_string(layer) + "." + part;
}
static void require_ling_state_arrays(const sr::ContinuationState &state) {
  size_t expected = 0;
  for (size_t layer = 0; layer < 24; ++layer)
    expected += layer % 4 == 3 ? 2 : 4;
  if (state.arrays.size() != expected)
    throw std::runtime_error("continuation array set mismatch");
  for (size_t layer = 0; layer < 24; ++layer) {
    if (layer % 4 == 3) {
      for (const char *part : {"keys", "values"})
        if (state.arrays.find(mla_state_name(layer, part)) == state.arrays.end())
          throw std::runtime_error("missing MLA continuation state");
    } else {
      for (const char *part : {"recurrent", "q_conv", "k_conv", "v_conv"})
        if (state.arrays.find(kda_state_name(layer, part)) == state.arrays.end())
          throw std::runtime_error("missing KDA continuation state");
    }
  }
}
static void restore_ling_state(const sr::ContinuationState &state,
    std::vector<std::unique_ptr<KdaLayer>> &kda,
    std::vector<std::unique_ptr<sl::MlaCache>> &mla) {
  require_ling_state_arrays(state);
  std::vector<std::unique_ptr<KdaLayer>> next_kda(24);
  std::vector<std::unique_ptr<sl::MlaCache>> next_mla(24);
  for (size_t layer = 0; layer < 24; ++layer) {
    std::string error;
    if (kda[layer]) {
      next_kda[layer] = std::make_unique<KdaLayer>(*kda[layer]);
      const auto &a = state.arrays;
      if (!next_kda[layer]->state.restore(
              a.at(kda_state_name(layer, "recurrent")),
              a.at(kda_state_name(layer, "q_conv")),
              a.at(kda_state_name(layer, "k_conv")),
              a.at(kda_state_name(layer, "v_conv")), error))
        throw std::runtime_error(error);
    } else {
      next_mla[layer] = std::make_unique<sl::MlaCache>(*mla[layer]);
      const auto &a = state.arrays;
      if (!next_mla[layer]->restore_live(size_t(state.position),
              a.at(mla_state_name(layer, "keys")),
              a.at(mla_state_name(layer, "values")), error))
        throw std::runtime_error(error);
    }
  }
  kda.swap(next_kda);
  mla.swap(next_mla);
}
static sr::ContinuationState snapshot_ling_state(
    const std::string &binding, size_t position, const std::vector<int> &prefix,
    const std::vector<float> &last_logits,
    const std::vector<std::unique_ptr<KdaLayer>> &kda,
    const std::vector<std::unique_ptr<sl::MlaCache>> &mla) {
  sr::ContinuationState state;
  state.binding = binding;
  state.position = position;
  state.prefix = prefix;
  state.last_logits = last_logits;
  for (size_t layer = 0; layer < 24; ++layer) {
    if (kda[layer]) {
      const auto &x = kda[layer]->state;
      state.arrays.emplace(kda_state_name(layer, "recurrent"),
          std::vector<float>(x.recurrent_data(), x.recurrent_data() + x.recurrent_size()));
      state.arrays.emplace(kda_state_name(layer, "q_conv"),
          std::vector<float>(x.q_conv_data(), x.q_conv_data() + x.q_conv_size()));
      state.arrays.emplace(kda_state_name(layer, "k_conv"),
          std::vector<float>(x.k_conv_data(), x.k_conv_data() + x.k_conv_size()));
      state.arrays.emplace(kda_state_name(layer, "v_conv"),
          std::vector<float>(x.v_conv_data(), x.v_conv_data() + x.v_conv_size()));
    } else {
      state.arrays.emplace(mla_state_name(layer, "keys"), mla[layer]->live_keys());
      state.arrays.emplace(mla_state_name(layer, "values"), mla[layer]->live_values());
    }
  }
  return state;
}
int main(int argc,char **argv) {
  try {
    if(argc < 7 || (argc - 7) % 2)
      throw std::runtime_error("usage: strata_ling_bf16 ROOT MANIFEST TOKENS_CSV NEW_OUTPUT CACHE_MIB CHECKPOINT_ID [--generate N --eos CSV --route-probe POS,LAYER --state-in FILE --state-out FILE --stop-after N --state-binding HEX64 --lm-head-residency off|cuda --matmul bf16-gemv|fp32-cublas --rope-frequency native|original --rms-mode native|torch-cuda-order --pointwise native|torch-cuda]");

    const std::filesystem::path output=argv[4];
    if(std::filesystem::exists(output)) throw std::runtime_error("refusing existing output directory");
    size_t generate=0; std::vector<int> eos; std::optional<RouteProbe> route_probe;
    std::optional<std::filesystem::path> state_in, state_out;
    std::optional<size_t> stop_after; std::string state_binding;
    sl::MlaRopeFrequency rope_frequency=sl::MlaRopeFrequency::Native;
    sl::RmsMode rms_mode=sl::RmsMode::Native;
    std::string rope_frequency_name="native", rms_mode_name="native", matmul_name="bf16-gemv", pointwise_name="native"; MatmulBackend matmul_backend=MatmulBackend::Bf16Gemv; LmHeadResidency lm_head_residency=LmHeadResidency::Off; std::vector<std::string> options;
    for(int i=7;i<argc;i+=2) {
      const std::string option=argv[i];
      if(std::find(options.begin(),options.end(),option)!=options.end()) throw std::invalid_argument("duplicate option");
      options.push_back(option);
      if(option=="--generate") generate=sr::unsigned_decimal(argv[i+1]);
      else if(option=="--eos") eos=sr::vocabulary_ids(argv[i+1],157184);
      else if(option=="--route-probe") route_probe=parse_route_probe(argv[i+1]);
      else if(option=="--state-in") state_in=argv[i+1];
      else if(option=="--state-out") state_out=argv[i+1];
      else if(option=="--stop-after") stop_after=sr::unsigned_decimal(argv[i+1]);
      else if(option=="--state-binding") state_binding=argv[i+1];
      else if(option=="--matmul") { matmul_name=argv[i+1]; matmul_backend=parse_matmul_backend(matmul_name); }
      else if(option=="--lm-head-residency") lm_head_residency=parse_lm_head_residency(argv[i+1]);
      else if(option=="--rope-frequency") {
        rope_frequency_name=argv[i+1];
        if(rope_frequency_name=="original") rope_frequency=sl::MlaRopeFrequency::Original;
        else if(rope_frequency_name!="native") throw std::invalid_argument("rope frequency must be native or original");
      }
      else if(option=="--rms-mode") {
        rms_mode_name=argv[i+1];
        if(rms_mode_name=="torch-cuda-order") rms_mode=sl::RmsMode::TorchCudaOrder;
        else if(rms_mode_name!="native") throw std::invalid_argument("RMS mode must be native or torch-cuda-order");
      }
      else if(option=="--pointwise") {
        pointwise_name=argv[i+1];
        if(pointwise_name!="native" && pointwise_name!="torch-cuda") throw std::invalid_argument("pointwise mode must be native or torch-cuda");
#ifndef STRATA_LING_TORCH_OPS
        if(pointwise_name=="torch-cuda") throw std::invalid_argument("torch-cuda pointwise mode requires LING_ENABLE_TORCH_OPS");
#endif
      }
      else throw std::invalid_argument("unknown option");
    }
    if(generate>1024) throw std::invalid_argument("generation exceeds 1024-token bound");
    if(generate && eos.empty()) throw std::invalid_argument("generation needs EOS IDs");
    sr::GreedyGeneration generation(sr::vocabulary_ids(argv[3],157184),generate,eos,157184,1024);
    const bool continuation_mode = state_in.has_value() || state_out.has_value() || stop_after.has_value() || !state_binding.empty();
    if (continuation_mode && !sr::continuation_detail::hex64(state_binding))
      throw std::invalid_argument("continuation mode requires --state-binding HEX64");
    if (stop_after && !state_out) throw std::invalid_argument("stop-after requires state-out");
    if (route_probe && continuation_mode)
      throw std::invalid_argument("route probe is incompatible with continuation mode");
    const size_t cache_mib=sr::unsigned_decimal(argv[5]);
    if(cache_mib<5 || cache_mib>2048) throw std::runtime_error("cache budget must be 5..2048 MiB");
    const std::string identity=argv[6];
    if(identity.size()!=64 || identity.find_first_not_of("0123456789abcdef")!=std::string::npos)
      throw std::runtime_error("checkpoint identity must be a SHA-256 hex digest");
    std::signal(SIGINT,cancel_handler); std::signal(SIGTERM,cancel_handler);
    sr::TensorStore store(argv[1],argv[2]);
    const auto &embedding=store.tensor("model.word_embeddings.weight");
    shape(embedding,{157184,1536});
    if(embedding.dtype!="BF16") throw std::runtime_error("original BF16 embedding required");
    const auto &lm_head=store.tensor("lm_head.weight");
    shape(lm_head,{157184,1536});
    if(lm_head.dtype!="BF16") throw std::runtime_error("original BF16 lm_head required");
    check(cudaSetDevice(0));
    cudaDeviceProp props{}; check(cudaGetDeviceProperties(&props,0));
    if(props.major!=8 || props.minor!=9) throw std::runtime_error("this target is qualified only for sm_89");
    size_t free=0,total=0; check(cudaMemGetInfo(&free,&total));
    const size_t budget=cache_mib*1024*1024;
    constexpr size_t memory_margin=size_t(512)*1024*1024;
    const size_t lm_head_residency_bytes=lm_head_residency==LmHeadResidency::Cuda ? lm_head.bytes : 0;
    const size_t required=budget+Matvec::workspace_bytes(matmul_backend)+memory_margin+lm_head_residency_bytes;
    if(free<required) {
      if(lm_head_residency==LmHeadResidency::Cuda) throw std::runtime_error("insufficient free GPU memory for requested lm-head residency; no process will be stopped");
      throw std::runtime_error("insufficient free GPU memory; no process will be stopped");
    }
    Matvec mv(matmul_backend);
    if(lm_head_residency==LmHeadResidency::Cuda) {
      try { mv.pin_lm_head(lm_head); }
      catch(const std::exception &e) { throw std::runtime_error(std::string("requested lm-head residency initialization failed: ")+e.what()); }
    }
#ifdef STRATA_LING_TORCH_OPS
    std::unique_ptr<strata::ling_torch::Ops> torch_ops;
    if (pointwise_name == "torch-cuda") torch_ops=std::make_unique<strata::ling_torch::Ops>("cuda");
#endif
    auto backend=std::make_shared<CudaBackend>();
    sr::ExpertCache cache({budget,24,128},backend);
    sr::ExpertCache::Provider provider=[&](const sr::ExpertKey &key,sr::ExpertPayload &p,std::string &error) {
      try {
        const std::string prefix="model.layers."+std::to_string(key.layer)+".mlp.experts."+std::to_string(key.expert)+".";
        const auto &g=store.tensor(prefix+"gate_proj.weight");
        const auto &u=store.tensor(prefix+"up_proj.weight");
        const auto &d=store.tensor(prefix+"down_proj.weight");
        shape(g,{512,1536}); shape(u,{512,1536}); shape(d,{1536,512});
        if(g.dtype!="BF16" || u.dtype!="BF16" || d.dtype!="BF16") throw std::runtime_error("expert is not BF16");
        p.gate_shape=p.up_shape={512,1536}; p.down_shape={1536,512};
        p.gate.assign(g.data,g.data+g.bytes);p.up.assign(u.data,u.data+u.bytes);p.down.assign(d.data,d.data+d.bytes);
        return true;
      } catch(const std::exception &e) {error=e.what();return false;}
    };
    std::vector<std::unique_ptr<KdaLayer>> kda(24);
    std::vector<std::unique_ptr<sl::MlaCache>> mla(24);
    for(size_t l=0;l<24;++l) {
      if(l%4==3) { sl::MlaConfig config{}; config.rope_frequency=rope_frequency;
        mla[l]=std::make_unique<sl::MlaCache>(config,generation.kv_capacity()); }
      else kda[l]=std::make_unique<KdaLayer>(store,"model.layers."+std::to_string(l)+".attention.");
    }
    size_t start_position = 0;
    bool state_loaded = false;
    if (state_in) {
      const auto restored = sr::load_continuation(*state_in);
      sr::require_continuation(restored, state_binding, generation.prompt(),
                               generation.kv_capacity());
      // Validate generation reconstruction before replacing any live layer
      // state. The layer replacement below is itself staged and atomic.
      auto staged_generation = generation;
      staged_generation.resume_prefix(restored.prefix, restored.last_logits);
      restore_ling_state(restored, kda, mla);
      generation = std::move(staged_generation);
      start_position = size_t(restored.position);
      state_loaded = true;
    }
    if (stop_after && *stop_after < start_position)
      throw std::invalid_argument("stop-after precedes restored position");
    const sr::RouterConfig routing{128,8,4,8,2.5f};
    std::vector<float> probe_input_hidden, probe_post_attention_hidden, probe_gate_input, probe_router_logits, probe_sigmoid,
        probe_bias, probe_corrected, probe_weights;
    std::vector<size_t> probe_ids;
    bool probe_captured=false;
    std::filesystem::create_directories(output);
    std::ofstream routes(output/"routes.tsv"); routes<<std::setprecision(9);
    const auto start=std::chrono::steady_clock::now();
    size_t processed_position = start_position;
    std::vector<float> last_committed_logits;
    for(size_t pos=start_position; pos<generation.evaluated().size() &&
        (!stop_after || pos < *stop_after); ++pos) {
      const auto &tokens=generation.evaluated();
      check_cancelled(); std::vector<float> h(1536);
      for(size_t i=0;i<h.size();++i) {
        uint16_t b; std::memcpy(&b,embedding.data+(size_t(tokens[pos])*1536+i)*2,2);h[i]=bf16(b);
      }
      for(size_t layer=0;layer<24;++layer) {
        check_cancelled(); const std::string pre="model.layers."+std::to_string(layer)+".";
        const std::vector<float> input_hidden = h;
        auto a=norm(h,store.tensor(pre+"input_layernorm.weight"),rms_mode
#ifdef STRATA_LING_TORCH_OPS
                    , torch_ops.get()
#endif
                    );
        if(kda[layer]) add(h,kda[layer]->step(mv,store,pre+"attention.",a
#ifdef STRATA_LING_TORCH_OPS
                                                  , torch_ops.get()
#endif
                                                  ));
        else add(h,mla_step(mv,store,pre+"attention.",a,*mla[layer],pos,rms_mode
#ifdef STRATA_LING_TORCH_OPS
                             , torch_ops.get()
#endif
                             ));
        const std::vector<float> post_attention_hidden = h;
        auto x=norm(h,store.tensor(pre+"post_attention_layernorm.weight"),rms_mode
#ifdef STRATA_LING_TORCH_OPS
                    , torch_ops.get()
#endif
                    );
        if(layer==0) add(h,mlp(mv,store,pre+"mlp.",x
#ifdef STRATA_LING_TORCH_OPS
                               , torch_ops.get()
#endif
                               ));
        else {
          auto logits=mv.apply(store.tensor(pre+"mlp.gate.weight"),x);
          auto bias=floats(store.tensor(pre+"mlp.gate.expert_bias"));
          if(logits.size()!=128 || bias.size()!=128) throw std::runtime_error("router geometry");
          auto selected=sr::route(logits.data(),bias.data(),128,routing);
#ifdef STRATA_LING_TORCH_OPS
          if (torch_ops) { std::vector<size_t> ids; std::vector<float> weights, scores, corrected; std::string error; if (!torch_ops->route(logits,bias,ids,weights,scores,corrected,error)) throw std::runtime_error(error); selected.expert_ids=std::move(ids); selected.weights=std::move(weights); }
#endif
          if(route_probe && route_probe->position==pos && route_probe->layer==layer) {
            probe_gate_input=x; probe_router_logits=logits; probe_bias=bias;
            probe_input_hidden=input_hidden; probe_post_attention_hidden=post_attention_hidden;
            probe_sigmoid.resize(logits.size()); probe_corrected.resize(logits.size());
            for(size_t i=0;i<logits.size();++i) {
              const float score = logits[i] >= 0.0f
                ? 1.0f/(1.0f+std::exp(-logits[i]))
                : std::exp(logits[i])/(1.0f+std::exp(logits[i]));
              probe_sigmoid[i]=score; probe_corrected[i]=score+bias[i];
            }
            probe_ids=selected.expert_ids; probe_weights=selected.weights;
            probe_captured=true;
          }
          std::vector<float> mixed(1536,0); std::vector<std::vector<float>> expert_outputs; routes<<pos<<'\t'<<layer;
          expert_outputs.reserve(8);
          for(size_t j=0;j<8;++j) {
            auto expert=selected.expert_ids[j];routes<<'\t'<<expert<<':'<<selected.weights[j];
            std::string error; auto lease=cache.acquire({identity,uint32_t(layer),uint32_t(expert)},provider,error);
            if(!lease) throw std::runtime_error(error);
            auto device=std::dynamic_pointer_cast<const ExpertDevice>(lease->backend_value());
            if(!device) throw std::runtime_error("wrong backend payload");
            const auto *w=static_cast<const uint16_t *>(device->weights.p);
            constexpr size_t count=512*1536;
            auto g=mv.apply_device(w,512,1536,x);auto u=mv.apply_device(w+count,512,1536,x);
            auto y=mv.apply_device(w+2*count,1536,512,swiglu(std::move(g),u
#ifdef STRATA_LING_TORCH_OPS
                                                               , torch_ops.get()
#endif
                                                               ));
#ifdef STRATA_LING_TORCH_OPS
            if (torch_ops) expert_outputs.push_back(std::move(y));
            else
#endif
              for(size_t d=0;d<mixed.size();++d) mixed[d]+=y[d]*selected.weights[j];
          }
#ifdef STRATA_LING_TORCH_OPS
          if (torch_ops) { std::string error; if (!torch_ops->weighted_sum(expert_outputs, selected.weights, mixed, error)) throw std::runtime_error(error); }
#endif
          routes<<'\n';add(mixed,mlp(mv,store,pre+"mlp.shared_experts.",x
#ifdef STRATA_LING_TORCH_OPS
                                      , torch_ops.get()
#endif
                                      ));add(h,mixed);
        }
        write_row(output/("layer-"+std::to_string(layer)+".f32"),h);
      }
      h=norm(h,store.tensor("model.norm.weight"),rms_mode
#ifdef STRATA_LING_TORCH_OPS
             , torch_ops.get()
#endif
             );write_row(output/"final-hidden.f32",h);
      auto logits=lm_head_residency==LmHeadResidency::Cuda ? mv.apply_lm_head(h) : mv.apply(lm_head,h);write_row(output/"logits.f32",logits);
      if (continuation_mode) last_committed_logits = logits;
      processed_position = pos + 1;
      if(generation.sampling_position(pos)) generation.sample(pos,logits);
      std::cerr<<"token "<<pos<<" complete; top1 "<<std::distance(logits.begin(),std::max_element(logits.begin(),logits.end()))<<'\n';
    }
    check_cancelled();
    if (continuation_mode && processed_position == start_position) {
      for (int layer=0;layer<24;++layer) {
        std::ofstream empty(output/("layer-"+std::to_string(layer)+".f32"),std::ios::binary);
        if(!empty) throw std::runtime_error("empty state trace write failed");
      }
      for(const char* name:{"final-hidden.f32","logits.f32"}) {
        std::ofstream empty(output/name,std::ios::binary);
        if(!empty) throw std::runtime_error("empty state trace write failed");
      }
    }
    const bool paused = stop_after && processed_position < generation.evaluated().size();
    if (paused && !state_out)
      throw std::runtime_error("paused continuation requires --state-out");
    std::vector<int> committed_prefix(generation.evaluated().begin(),
                                      generation.evaluated().begin() + processed_position);
    if (processed_position == start_position && state_loaded) {
      const auto restored = sr::load_continuation(*state_in);
      last_committed_logits = restored.last_logits;
    }
    if (state_out) {
      if (state_out->has_parent_path()) std::filesystem::create_directories(state_out->parent_path());
      auto snapshot = snapshot_ling_state(state_binding, processed_position,
          committed_prefix, last_committed_logits, kda, mla);
      sr::save_continuation(*state_out, snapshot);
    }
    if (continuation_mode) {
      std::ofstream sidecar(output/"continuation.json");
      sidecar << "{\"start_position\":" << start_position
              << ",\"processed_position\":" << processed_position
              << ",\"processed_tokens\":" << (processed_position-start_position)
              << ",\"paused\":" << (paused ? "true" : "false")
              << ",\"loaded\":" << (state_loaded ? "true" : "false")
              << ",\"saved\":" << (state_out ? "true" : "false") << "}\n";
      sidecar.flush();
      if (!sidecar) throw std::runtime_error("continuation sidecar write failed");
    }
    auto stats=cache.stats(); routes.flush();
    if(route_probe) {
      if(!probe_captured) throw std::runtime_error("route probe position was not evaluated");
      write_route_probe(output/"route-probe.json",*route_probe,probe_input_hidden,
                        probe_post_attention_hidden,probe_gate_input,
                        probe_router_logits,probe_sigmoid,probe_bias,probe_corrected,
                        probe_ids,probe_weights);
    }
    if(generate && !paused) {
      std::ofstream generated(output/"generation.json");
      generated << generation.receipt("strata-ling-generation-v1"); generated.flush();
      if(!generated) throw std::runtime_error("generation receipt write failed");
    }
    if(!routes) throw std::runtime_error("route trace write failed");
    std::ostringstream token_json;token_json<<'[';
    for(size_t i=0;i<committed_prefix.size();++i){if(i) token_json<<',';token_json<<committed_prefix[i];}token_json<<']';
    size_t host_attention_state_live_logical_bytes = 0;
    size_t host_attention_state_capacity_logical_bytes = 0;
    for (size_t layer = 0; layer < 24; ++layer) {
      if (mla[layer]) {
        host_attention_state_live_logical_bytes += mla[layer]->live_payload_bytes();
        host_attention_state_capacity_logical_bytes += mla[layer]->capacity_payload_bytes();
      } else {
        const size_t bytes = kda[layer]->state.logical_payload_bytes();
        host_attention_state_live_logical_bytes += bytes;
        host_attention_state_capacity_logical_bytes += bytes;
      }
    }
    std::ostringstream receipt;receipt<<std::setprecision(12)
      <<"{\"status\":\"executed\",\"model\":\"Ling-3.0-tiny\",\"activation_dtype\":\"FP32\",\"quantized\":false,\"tokens\":"
      <<committed_prefix.size()<<",\"token_ids\":"<<token_json.str()<<",\"logits_mode\":\"all\",\"logits_tokens\":"<<(processed_position-start_position)
      <<",\"checkpoint_id\":\""<<identity<<"\",\"layers\":24,\"rope_frequency\":\""<<rope_frequency_name<<"\",\"rms_mode\":\""<<rms_mode_name<<"\",\"pointwise_mode\":\""<<pointwise_name<<"\",\"cache_budget_bytes\":"<<budget
      <<",\"cache_resident_bytes\":"<<stats.bytes<<",\"cache_device_payload_bytes\":"<<stats.bytes<<",\"cache_host_payload_bytes\":"<<stats.bytes
      <<",\"cache_payload_scope\":\"resident payload only; excludes allocator overhead, one staged expert, mmap pages and other working buffers\""
      // Historical live-use field: retain its name and value for existing receipts.
      <<",\"host_attention_state_bytes\":"<<host_attention_state_live_logical_bytes
      <<",\"host_attention_state_live_logical_bytes\":"<<host_attention_state_live_logical_bytes
      <<",\"host_attention_state_capacity_logical_bytes\":"<<host_attention_state_capacity_logical_bytes
      <<",\"host_attention_state_payload_scope\":\"logical float payload: MLA key/value slots plus KDA recurrent and convolution state; excludes vector allocator capacity/overhead and temporary buffers\""
      <<",\"matmul_backend\":\""<<matmul_backend_name(matmul_backend)<<"\",\"lm_head_residency\":\""<<lm_head_residency_name(lm_head_residency)<<"\",\"lm_head_resident_bytes\":"<<mv.resident_lm_head_bytes()<<",\"lm_head_resident_uploads\":"<<mv.resident_lm_head_uploads()<<",\"lm_head_resident_apply_calls\":"<<mv.resident_lm_head_apply_calls()<<",\"matmul_fp32_weight_scratch_bytes\":"<<Matvec::fp32_scratch_bytes(matmul_backend)<<",\"cache_hits\":"<<stats.hits<<",\"cache_misses\":"<<stats.misses<<",\"cache_evictions\":"<<stats.evictions
      <<",\"workspace_bytes\":"<<Matvec::workspace_bytes(matmul_backend)<<",\"wall_seconds\":"
      <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
    std::ofstream out(output/"execution.json");out<<receipt.str();out.flush();
    if(!out) throw std::runtime_error("execution receipt write failed");
    std::cout<<receipt.str();return 0;
  } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
