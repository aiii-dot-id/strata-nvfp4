// Copyright (c) AI Identity Incorported <james@aiii.id>.
#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
namespace strata::torch_router {
class Router {
 public:
  Router(std::size_t experts, std::size_t groups, std::size_t top_groups,
         std::size_t top_k, float scale, const char *device);
  ~Router();
  Router(const Router&) = delete;
  bool route(const float *logits, const float *bias, std::size_t count,
             std::vector<std::size_t> &ids, std::vector<float> &weights,
             std::vector<float> &scores, std::vector<float> &corrected,
             std::string &error);
 private:
  struct Impl; std::unique_ptr<Impl> impl_;
};
}
