// Opt-in Strata BF16 expert composition probe. Synthetic or original BF16 weights.
// Serial buffered staging only: this does not qualify overlap, eviction or a full model.
#include <cuda_runtime.h>
#include <strata/kernels/bf16_gemv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
static float unpack(uint16_t b) {
    uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f;
}
__global__ void swiglu(float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = gate[i]; gate[i] = (g / (1.f + expf(-g))) * up[i]; }
}
struct Slot {
    int n, ff;
    uint16_t *gate{}, *up{}, *down{};
    float *x{}, *g{}, *u{}, *out{};
    cudaStream_t stream{};
    cudaEvent_t start{}, stop{};
    Slot(int n_, int ff_) : n(n_), ff(ff_) {
        try {
        ck(cudaStreamCreate(&stream), "stream");
        ck(cudaEventCreate(&start), "event"); ck(cudaEventCreate(&stop), "event");
        size_t bytes = size_t(n) * ff * 2;
        ck(cudaMalloc(&gate, bytes), "gate"); ck(cudaMalloc(&up, bytes), "up"); ck(cudaMalloc(&down, bytes), "down");
        ck(cudaMalloc(&x, n * 4), "x"); ck(cudaMalloc(&g, ff * 4), "g");
        ck(cudaMalloc(&u, ff * 4), "u"); ck(cudaMalloc(&out, n * 4), "out");
        } catch (...) { release(); throw; }
    }
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    void release() noexcept {
        if (stream) cudaStreamSynchronize(stream);
        cudaFree(gate); cudaFree(up); cudaFree(down); cudaFree(x); cudaFree(g); cudaFree(u); cudaFree(out);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (stream) cudaStreamDestroy(stream);
    }
    ~Slot() { release(); }
};
struct PinnedBuffer {
    uint16_t* data{};
    explicit PinnedBuffer(size_t bytes) {
        ck(cudaHostAlloc(&data, bytes, cudaHostAllocDefault), "pinned buffer");
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    ~PinnedBuffer() { if (data) cudaFreeHost(data); }
};
struct Result {
    size_t weight_bytes{}; double h2d_ms{}, compute_ms{}, roundtrip_wall_ms{};
    double max_abs{}, max_rel{}; bool bytes_equal{true}, finite{true}, close{true}, reuse_equal{true};
    bool passed() const { return bytes_equal && finite && close && reuse_equal; }
};
static Result run(Slot& s, uint16_t* stage, uint16_t* readback, bool corrupt_copy, bool corrupt_output, const std::string& fixtures) {
    const size_t count = size_t(s.n) * s.ff, bytes = count * 2;
    std::mt19937 rng(20261016 + s.n);
    std::vector<uint16_t> gate(count), up(count), down(count);
    // Finite, mixed-sign, exactly representable BF16 source values; no conversion/quantization of model data.
    for (size_t i=0; i<count; ++i) {
        gate[i] = uint16_t(0x3900 + rng()%768) | uint16_t((rng()&1)<<15);
        up[i] = uint16_t(0x3900 + rng()%768) | uint16_t((rng()&1)<<15);
        down[i] = uint16_t(0x3900 + rng()%768) | uint16_t((rng()&1)<<15);
    }
    if (!fixtures.empty()) {
        const std::string family = s.n == 1536 ? "ling" : "ring";
        const char* names[] = {"gate", "up", "down"};
        std::vector<uint16_t>* roles[] = {&gate, &up, &down};
        for (int i=0; i<3; ++i) {
            std::ifstream f(fixtures + "/" + family + "-" + names[i] + ".bf16", std::ios::binary | std::ios::ate);
            if (!f || f.tellg()!=std::streamoff(bytes)) throw std::runtime_error("BF16 fixture size mismatch");
            f.seekg(0);
            if (!f.read(reinterpret_cast<char*>(roles[i]->data()), bytes)) throw std::runtime_error("BF16 fixture read failed");
        }
    }
    std::vector<float> x(s.n), gr(s.ff), ur(s.ff), ref(s.n), output(s.n), previous;
    for (float& f : x) f = float(int(rng()%513)-256)/1024.f;
    // Independent ordinary CPU reductions, not the GPU kernel's reduction tree.
    for (int o=0; o<s.ff; ++o) {
        for (int i=0; i<s.n; ++i) {
            gr[o] += x[i] * unpack(gate[size_t(o)*s.n+i]);
            ur[o] += x[i] * unpack(up[size_t(o)*s.n+i]);
        }
        gr[o] = (gr[o]/(1.f + std::exp(-gr[o]))) * ur[o];
    }
    for (int i=0; i<s.n; ++i)
        for (int o=0; o<s.ff; ++o) ref[i] += gr[o] * unpack(down[size_t(i)*s.ff+o]);
    ck(cudaMemcpyAsync(s.x, x.data(), s.n*4, cudaMemcpyHostToDevice, s.stream), "activation copy");
    Result r; r.weight_bytes = bytes*3;
    const auto begin = std::chrono::steady_clock::now();
    const uint16_t* src[] = {gate.data(), up.data(), down.data()};
    uint16_t* dst[] = {s.gate, s.up, s.down};
    for (int m=0; m<3; ++m) {
        std::memcpy(stage, src[m], bytes);
        ck(cudaEventRecord(s.start, s.stream), "copy start");
        ck(cudaMemcpyAsync(dst[m], stage, bytes, cudaMemcpyHostToDevice, s.stream), "weight copy");
        ck(cudaEventRecord(s.stop, s.stream), "copy stop");
        ck(cudaMemcpyAsync(readback, dst[m], bytes, cudaMemcpyDeviceToHost, s.stream), "weight readback");
        ck(cudaStreamSynchronize(s.stream), "copy ownership boundary");
        float ms; ck(cudaEventElapsedTime(&ms, s.start, s.stop), "copy timing"); r.h2d_ms += ms;
        if (corrupt_copy && m==0) readback[0] ^= 1;
        r.bytes_equal = r.bytes_equal && std::memcmp(src[m], readback, bytes)==0;
    }
    r.roundtrip_wall_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    for (int repeat=0; repeat<3; ++repeat) {
        ck(cudaEventRecord(s.start, s.stream), "compute start");
        strata::kernels::bf16_gemv_fp32_mmvf(s.x, s.gate, s.g, s.n, s.ff, s.stream);
        strata::kernels::bf16_gemv_fp32_mmvf(s.x, s.up, s.u, s.n, s.ff, s.stream);
        swiglu<<<(s.ff+127)/128,128,0,s.stream>>>(s.g, s.u, s.ff);
        ck(cudaGetLastError(), "swiglu");
        strata::kernels::bf16_gemv_fp32_mmvf(s.g, s.down, s.out, s.ff, s.n, s.stream);
        ck(cudaEventRecord(s.stop, s.stream), "compute stop");
        ck(cudaMemcpyAsync(output.data(), s.out, s.n*4, cudaMemcpyDeviceToHost, s.stream), "output readback");
        ck(cudaStreamSynchronize(s.stream), "compute ownership boundary");
        float ms; ck(cudaEventElapsedTime(&ms, s.start, s.stop), "compute timing"); r.compute_ms += ms;
        if (corrupt_output) output[0] += 1.f;
        if (repeat) r.reuse_equal = r.reuse_equal && std::memcmp(previous.data(),output.data(),s.n*4)==0;
        previous=output;
        for (int i=0; i<s.n; ++i) {
            if (!std::isfinite(output[i]) || !std::isfinite(ref[i])) { r.finite=false; r.close=false; continue; }
            double a=std::abs(double(output[i])-ref[i]);
            r.max_abs=std::max(r.max_abs,a); r.max_rel=std::max(r.max_rel,a/std::max(1e-8,std::abs(double(ref[i]))));
            r.close = r.close && a <= 1e-5 + 1e-4*std::abs(double(ref[i]));
        }
    }
    return r;
}
static void emit(const Result& r) {
    std::cout << "{\"weight_bytes\":"<<r.weight_bytes<<",\"h2d_ms\":"<<r.h2d_ms
      <<",\"roundtrip_wall_ms\":"<<r.roundtrip_wall_ms<<",\"three_compute_calls_ms\":"<<r.compute_ms
      <<",\"max_abs_error\":"<<r.max_abs<<",\"max_relative_error\":"<<r.max_rel
      <<",\"byte_identity\":"<<r.bytes_equal<<",\"finite\":"<<r.finite
      <<",\"within_tolerance\":"<<r.close<<",\"reuse_bitwise_equal\":"<<r.reuse_equal<<"}";
}
int main(int argc,char** argv) {
    bool corrupt_copy=false, corrupt_output=false;
    std::string fixtures;
    for(int i=1; i<argc; ++i) {
        std::string a=argv[i];
        if(a=="--mutate-copy") corrupt_copy=true;
        else if(a=="--mutate-output") corrupt_output=true;
        else if(a=="--fixtures" && i+1<argc) fixtures=argv[++i];
        else return 2;
    }
    try {
        ck(cudaSetDevice(0), "select device"); cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop,0), "device properties");
        if(prop.major!=8 || prop.minor!=9) throw std::runtime_error("This qualification is pinned to sm_89");
        const size_t stage_bytes=2048*512*2;
        PinnedBuffer stage(stage_bytes), readback(stage_bytes);
        const auto start=std::chrono::steady_clock::now();
        Slot ling(1536,512), ring(2048,512);
        Result a=run(ling,stage.data,readback.data,corrupt_copy,corrupt_output,fixtures);
        Result b=run(ring,stage.data,readback.data,corrupt_copy,corrupt_output,fixtures);
        bool pass=a.passed() && b.passed();
        std::cout<<std::boolalpha<<std::setprecision(10)<<"{\"status\":\""<<(pass?"pass":"fail")
          <<"\",\"scope\":\"Strata BF16 serial expert composition; synthetic activations; not a model or overlap qualification\""
          <<",\"weight_source\":\""<<(fixtures.empty()?"synthetic":"checkpoint expert fixture")<<"\""
          <<",\"device\":\""<<prop.name<<"\",\"compute_capability\":\"8.9\",\"weight_dtype\":\"BF16\",\"activation_accumulation_dtype\":\"FP32\""
          <<",\"gpu_buffer_bytes\":11046912,\"pinned_host_bytes\":4194304,\"atol\":0.00001,\"rtol\":0.0001,\"mutation\":\""
          <<(corrupt_copy?"copy":corrupt_output?"output":"none")<<"\",\"ling\":";
        emit(a); std::cout<<",\"ring\":";emit(b);
        std::cout<<",\"wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
        return pass?0:1;
    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n";return 1; }
}
