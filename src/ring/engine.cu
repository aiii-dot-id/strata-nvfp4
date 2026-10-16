// AI Identity Incorported <james@aiii.id>
// Opt-in Ring BF16 reference executor. Original weights, FP32 working state.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <strata/kernels/bf16_gemv.hpp>
#include <strata/ring/attention.hpp>
#include <strata/ring/expert_cache.hpp>
#include <strata/ring/generation.hpp>
#include <strata/ring/numerics.hpp>
#include <strata/ring/routing.hpp>
#include <strata/ring/tensor_store.hpp>
#include <vector>

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
class Matvec {
  // Largest matrix is the unmodified BF16 output head.
  DeviceBuffer weight{size_t(157184) * 2048 * 2}, x{5120 * 4}, y{157184 * 4};

public:
  static constexpr size_t workspace_bytes =
      size_t(157184) * 2048 * 2 + 5120 * 4 + 157184 * 4;
  std::vector<float> apply_device(const uint16_t *w, size_t rows, size_t cols,
                                  const std::vector<float> &input) {
    if (input.size() != cols || cols > 5120 || rows > 157184 || cols % 2)
      throw std::runtime_error("matvec geometry");
    check(cudaMemcpy(x.p, input.data(), cols * 4, cudaMemcpyHostToDevice));
    strata::kernels::bf16_gemv_fp32_mmvf(static_cast<float *>(x.p), w,
                                         static_cast<float *>(y.p), cols, rows,
                                         nullptr);
    check(cudaGetLastError());
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
int main(int argc, char **argv) {
  try {
    if (argc < 7 || (argc - 7) % 2)
      throw std::runtime_error(
          "usage: strata_ring_bf16 ROOT MANIFEST "
          "TOKENS_CSV NEW_OUTPUT CACHE_MIB CHECKPOINT_ID "
          "[--generate N --eos CSV] [--logits-mode all|last]");
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
      else
        throw std::invalid_argument("unknown option");
    }
    if (logits_mode != "all" && logits_mode != "last")
      throw std::invalid_argument("logits mode must be all or last");
    if (generate && logits_mode != "all")
      throw std::invalid_argument(
          "generation requires all logits for validation");
    sr::GreedyGeneration generation(sr::vocabulary_ids(argv[3], 157184),
                                    generate, eos);
    const auto &tokens = generation.evaluated();
    std::signal(SIGINT, cancel_handler);
    std::signal(SIGTERM, cancel_handler);
    sr::TensorStore store(argv[1], argv[2]);
    const auto &embedding = store.tensor("model.word_embeddings.weight");
    shape(embedding, {157184, 2048});
    if (embedding.dtype != "BF16")
      throw std::runtime_error("original BF16 embedding required");
    check(cudaSetDevice(0));
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0));
    if (props.major != 8 || props.minor != 9)
      throw std::runtime_error("this target is qualified only for sm_89");
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total));
    const size_t budget = cache_mib * 1024 * 1024;
    if (free < budget + Matvec::workspace_bytes + size_t(512) * 1024 * 1024)
      throw std::runtime_error(
          "insufficient free GPU memory; no process will be stopped");
    Matvec mv;
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
    sr::AttentionConfig attention;
    attention.rope_dim = 64;
    const sr::RouterConfig routing{256, 8, 4, 8, 2.5f};
    std::filesystem::create_directories(output);
    std::ofstream routes(output / "routes.tsv");
    routes << std::setprecision(9);
    const auto start = std::chrono::steady_clock::now();
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
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
        auto a = norm(h, store.tensor(pre + "input_layernorm.weight"));
        const auto &qkv_weight =
            store.tensor(pre + "attention.query_key_value.weight");
        shape(qkv_weight, {3072, 2048});
        auto qkv = mv.apply(qkv_weight, a);
        auto qw =
            floats(store.tensor(pre + "attention.query_layernorm.weight"));
        auto kw = floats(store.tensor(pre + "attention.key_layernorm.weight"));
        if (qw.size() != 128 || kw.size() != 128)
          throw std::runtime_error("QK norm geometry");
        std::vector<float> context(2048);
        std::string error;
        if (!sr::attention_forward(attention, qkv.data(), qkv.data() + 2048,
                                   qkv.data() + 2560, 1, pos, qw.data(),
                                   kw.data(), kv[layer], context.data(), error))
          throw std::runtime_error(error);
        add(h, mv.apply(store.tensor(pre + "attention.dense.weight"), context));
        auto x = norm(h, store.tensor(pre + "post_attention_layernorm.weight"));
        if (layer == 0)
          add(h, mlp(mv, store, pre + "mlp.", x));
        else {
          auto logits = mv.apply(store.tensor(pre + "mlp.gate.weight"), x);
          auto bias = floats(store.tensor(pre + "mlp.gate.expert_bias"));
          if (logits.size() != 256 || bias.size() != 256)
            throw std::runtime_error("router geometry");
          auto selected = sr::route(logits.data(), bias.data(), 256, routing);
          std::vector<float> mixed(2048, 0);
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
                                     swiglu(std::move(g), u));
            for (size_t d = 0; d < mixed.size(); ++d)
              mixed[d] += y[d] * selected.weights[j];
          }
          routes << '\n';
          add(mixed, mlp(mv, store, pre + "mlp.shared_experts.", x));
          add(h, mixed);
        }
        write_row(output / ("layer-" + std::to_string(layer) + ".f32"), h);
      }
      h = norm(h, store.tensor("model.norm.weight"));
      write_row(output / "final-hidden.f32", h);
      if (logits_mode == "all" || pos + 1 == tokens.size()) {
        auto logits = mv.apply(store.tensor("lm_head.weight"), h);
        write_row(output / "logits.f32", logits);
        const auto top = std::distance(
            logits.begin(), std::max_element(logits.begin(), logits.end()));
        if (generation.sampling_position(pos))
          generation.sample(pos, logits);
        std::cerr << "token " << pos << " complete; top1 " << top << '\n';
      } else if (pos % 64 == 0)
        std::cerr << "prefill " << pos << '\n';
    }
    check_cancelled();
    if (generate) {
      std::ofstream generated(output / "generation.json");
      generated << generation.receipt();
      generated.flush();
      if (!generated)
        throw std::runtime_error("generation receipt write failed");
    }
    auto stats = cache.stats();
    routes.flush();
    if (!routes)
      throw std::runtime_error("route trace write failed");
    std::ostringstream token_json;
    token_json << '[';
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (i)
        token_json << ',';
      token_json << tokens[i];
    }
    token_json << ']';
    std::ostringstream receipt;
    receipt << std::setprecision(12)
            << "{\"status\":\"executed\",\"model\":\"Ring-mini-2.0\","
               "\"activation_dtype\":\"FP32\",\"quantized\":false,\"tokens\":"
            << tokens.size() << ",\"token_ids\":" << token_json.str()
            << ",\"logits_mode\":\"" << logits_mode << "\",\"logits_tokens\":"
            << (logits_mode == "all" ? tokens.size() : 1)
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
            << ",\"workspace_bytes\":" << Matvec::workspace_bytes
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
