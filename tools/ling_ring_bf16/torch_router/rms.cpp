// Copyright (c) AI Identity Incorported <james@aiii.id>.
#include "rms.hpp"
#include <torch/torch.h>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace strata::torch_router {
struct Rms::Impl { c10::Device device; explicit Impl(const char *d): device(c10::kCPU) { if(!d) throw std::invalid_argument("RMS device is null"); std::string dev(d); if(dev=="cuda") device=c10::kCUDA; else if(dev!="cpu") throw std::invalid_argument("RMS device must be cpu or cuda"); } };
Rms::Rms(const char *d): impl_(std::make_unique<Impl>(d)) {}
Rms::~Rms() = default;
bool Rms::normalize(const float *xp, const float *wp, std::size_t rows, std::size_t dim,
                    float eps, std::vector<float> &out, std::string &error) {
  out.clear();
  try {
    if (!xp || !wp || rows == 0 || dim == 0 || rows > (std::numeric_limits<std::size_t>::max() / dim) ||
        !std::isfinite(eps) || eps <= 0.0f)
      throw std::invalid_argument("invalid RMS geometry or epsilon");
    const std::size_t n = rows * dim;
    for (std::size_t i=0;i<n;i++) if (!std::isfinite(xp[i])) throw std::invalid_argument("nonfinite RMS input");
    for (std::size_t i=0;i<dim;i++) if (!std::isfinite(wp[i])) throw std::invalid_argument("nonfinite RMS weight");
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto x = torch::from_blob(const_cast<float*>(xp), {(long)rows,(long)dim}, opts).clone().to(impl_->device);
    auto w = torch::from_blob(const_cast<float*>(wp), {(long)dim}, opts).clone().to(impl_->device);
    auto mean = x.pow(2).mean(-1, true);
    if (!torch::isfinite(mean).all().item<bool>()) throw std::runtime_error("nonfinite RMS mean");
    auto inv = torch::rsqrt(mean + eps);
    if (!torch::isfinite(inv).all().item<bool>()) throw std::runtime_error("nonfinite RMS inverse");
    auto y = x * inv * w;
    auto cpu = y.to(torch::kCPU).contiguous();
    auto a = cpu.accessor<float,2>();
    std::vector<float> tmp(n);
    for (std::size_t r=0;r<rows;r++) for (std::size_t c=0;c<dim;c++) {
      float v=a[(long)r][(long)c];
      if (!std::isfinite(v)) throw std::runtime_error("nonfinite RMS output");
      tmp[r*dim+c]=v;
    }
    out.swap(tmp); return true;
  } catch (const std::exception &e) { error=e.what(); return false; }
}
}
