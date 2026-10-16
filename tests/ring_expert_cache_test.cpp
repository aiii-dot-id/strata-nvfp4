#include "strata/ring/expert_cache.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
namespace {
using namespace strata::ring;
struct V : ExpertCache::BackendValue {
  std::size_t n;
  explicit V(std::size_t x) : n(x) {}
};
struct B : ExpertCache::Backend {
  int uploads = 0;
  bool fail = false;
  bool thrower = false;
  ExpertCache::BackendValuePtr upload(const ExpertKey &, const ExpertPayload &p,
                                      std::string &e) override {
    if (thrower)
      throw std::runtime_error("forced backend exception");
    if (fail) {
      e = "forced upload failure";
      return {};
    }
    ++uploads;
    return std::make_shared<V>(p.bytes());
  }
};
ExpertPayload make(std::uint8_t x) {
  ExpertPayload p;
  p.gate_shape = {1, 2};
  p.up_shape = {1, 2};
  p.down_shape = {2, 1};
  p.gate.assign(4, x);
  p.up.assign(4, x + 1);
  p.down.assign(4, x + 2);
  return p;
}
} // namespace
int main() {
  auto b = std::make_shared<B>();
  ExpertCache c({12, 4, 8}, b);
  ExpertKey a{"cp", 0, 0}, d{"cp", 0, 1}, e{"cp", 0, 2};
  int calls = 0;
  std::string er;
  auto prov = [&](const ExpertKey &k, ExpertPayload &p, std::string &) {
    ++calls;
    p = make(std::uint8_t(k.expert + 10));
    return true;
  };
  auto x = c.acquire(a, prov, er);
  assert(x && x->bytes() == 12 && calls == 1 && b->uploads == 1);
  auto h = c.acquire(a, prov, er);
  assert(h && h->payload().gate[0] == 10 && calls == 1);
  h.reset();
  auto pin = std::move(*x);
  auto blocked = c.acquire(d, prov, er);
  assert(!blocked && er.find("pinned") != std::string::npos);
  pin = {};
  auto y = c.acquire(d, prov, er);
  assert(y && c.stats().evictions == 1 && y->payload().up[0] == 12);
  y.reset();
  auto z = c.acquire(e, prov, er);
  assert(z && z->payload().down[0] == 14);
  auto s = c.stats();
  assert(s.hits == 1 && s.misses == 4 && s.bytes == 12);
  auto shape_bad = make(1);
  shape_bad.up_shape = {2, 2};
  shape_bad.up.assign(8, 1);
  auto shape_inv = c.acquire(
      {"cp", 0, 7},
      [&](const ExpertKey &, ExpertPayload &p, std::string &) {
        p = shape_bad;
        return true;
      },
      er);
  assert(!shape_inv && er.find("gate/up") != std::string::npos &&
         c.stats().bytes == 12);
  auto bad = make(1);
  bad.gate.resize(1);
  auto inv = c.acquire(
      {"cp", 0, 3},
      [&](const ExpertKey &, ExpertPayload &p, std::string &) {
        p = bad;
        return true;
      },
      er);
  assert(!inv && er.find("gate") != std::string::npos && c.stats().bytes == 12);
  z.reset();
  b->fail = true;
  auto f = c.acquire({"cp", 1, 0}, prov, er);
  assert(!f && c.stats().bytes <= 12);
  b->fail = false;
  auto r = c.acquire({"cp", 1, 0}, prov, er);
  assert(r && c.stats().bytes == 12);
  r.reset();
  auto provider_throw = c.acquire(
      {"cp", 2, 0},
      [&](const ExpertKey &, ExpertPayload &, std::string &) -> bool {
        throw std::runtime_error("forced provider exception");
      },
      er);
  assert(!provider_throw && er.find("provider threw") != std::string::npos &&
         c.stats().bytes <= 12);
  b->thrower = true;
  auto backend_throw = c.acquire({"cp", 2, 1}, prov, er);
  assert(!backend_throw && er.find("backend upload threw") != std::string::npos &&
         c.stats().bytes <= 12);
  b->thrower = false;
  auto recovered = c.acquire({"cp", 2, 1}, prov, er);
  assert(recovered && c.stats().bytes == 12);
  auto bk = c.acquire({"", 0, 0}, prov, er);
  assert(!bk && !er.empty());
  std::cout << "ring_expert_cache_test: ok\n";
}
