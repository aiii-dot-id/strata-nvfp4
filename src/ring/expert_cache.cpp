#include "strata/ring/expert_cache.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::ring {
namespace {
std::size_t shape_bytes(const ExpertShape &s) noexcept {
  const auto n = std::uint64_t(s.rows) * std::uint64_t(s.cols);
  return n > std::numeric_limits<std::size_t>::max() / 2
             ? 0
             : std::size_t(n * 2);
}

void set_callback_error(std::string &e, const char *kind,
                        const std::exception *ex = nullptr) {
  e = std::string("expert cache ") + kind + " threw";
  if (ex && ex->what() && *ex->what()) {
    e += ": ";
    e += ex->what();
  }
}
} // namespace

std::size_t ExpertKeyHash::operator()(const ExpertKey &k) const noexcept {
  std::size_t h = std::hash<std::string>{}(k.checkpoint);
  h ^= std::size_t(k.layer) + 0x9e3779b9u + (h << 6) + (h >> 2);
  h ^= std::size_t(k.expert) + 0x9e3779b9u + (h << 6) + (h >> 2);
  return h;
}

const ExpertKey &ExpertCache::Lease::key() const {
  if (!entry_)
    throw std::logic_error("invalid expert cache lease");
  return entry_->key;
}
const ExpertPayload &ExpertCache::Lease::payload() const {
  if (!entry_)
    throw std::logic_error("invalid expert cache lease");
  return entry_->payload;
}
ExpertCache::BackendValuePtr ExpertCache::Lease::backend_value() const {
  return entry_ ? entry_->backend : BackendValuePtr{};
}
std::size_t ExpertCache::Lease::bytes() const noexcept {
  return entry_ ? entry_->payload.bytes() : 0;
}

ExpertCache::ExpertCache(Config c, std::shared_ptr<Backend> b)
    : config_(c), backend_(std::move(b)) {
  if (!config_.byte_budget)
    throw std::invalid_argument("expert cache budget must be positive");
  if (!backend_)
    throw std::invalid_argument("expert cache backend must be non-null");
}

ExpertCache::~ExpertCache() { clear(); }

bool ExpertCache::validate_key(const ExpertKey &k, std::string &e) const {
  if (k.checkpoint.empty()) {
    e = "expert cache checkpoint identity must be non-empty";
    return false;
  }
  if (config_.max_layers && k.layer >= config_.max_layers) {
    e = "expert cache layer is outside configured bounds";
    return false;
  }
  if (config_.max_experts && k.expert >= config_.max_experts) {
    e = "expert cache expert is outside configured bounds";
    return false;
  }
  return true;
}

bool ExpertCache::validate_payload(const ExpertPayload &p, std::string &e) {
  const ExpertShape *s[] = {&p.gate_shape, &p.up_shape, &p.down_shape};
  const std::vector<std::uint8_t> *b[] = {&p.gate, &p.up, &p.down};
  const char *n[] = {"gate", "up", "down"};
  for (int i = 0; i < 3; ++i) {
    const auto w = shape_bytes(*s[i]);
    if (!s[i]->rows || !s[i]->cols || !w) {
      e = std::string("expert cache ") + n[i] + " shape is invalid";
      return false;
    }
    if (b[i]->size() != w || b[i]->size() % 2) {
      e = std::string("expert cache ") + n[i] +
          " bytes do not match BF16 shape";
      return false;
    }
  }
  if (p.gate_shape.rows != p.up_shape.rows ||
      p.gate_shape.cols != p.up_shape.cols) {
    e = "expert cache gate/up shapes must match";
    return false;
  }
  if (p.down_shape.rows != p.gate_shape.cols ||
      p.down_shape.cols != p.gate_shape.rows) {
    e = "expert cache down shape must transpose gate/up";
    return false;
  }
  if (!p.bytes()) {
    e = "expert cache payload is empty";
    return false;
  }
  return true;
}

bool ExpertCache::make_room(std::size_t n, std::string &e) {
  if (n > config_.byte_budget) {
    e = "expert payload exceeds cache byte budget";
    return false;
  }
  while (stats_.bytes > config_.byte_budget - n) {
    bool done = false;
    for (auto i = lru_.begin(); i != lru_.end();) {
      auto f = entries_.find(*i);
      if (f == entries_.end()) {
        i = lru_.erase(i);
        continue;
      }
      if (f->second.use_count() != 1) {
        ++i;
        continue;
      }
      stats_.bytes -= f->second->payload.bytes();
      entries_.erase(f);
      i = lru_.erase(i);
      ++stats_.evictions;
      done = true;
      break;
    }
    if (!done) {
      e = "expert cache budget is pinned by active leases";
      return false;
    }
  }
  return true;
}

std::optional<ExpertCache::Lease>
ExpertCache::acquire(const ExpertKey &k, const Provider &p, std::string &e) {
  std::lock_guard lock(mutex_);
  if (!validate_key(k, e))
    return {};
  if (!p) {
    e = "expert cache provider must be non-null";
    return {};
  }
  auto f = entries_.find(k);
  if (f != entries_.end()) {
    ++stats_.hits;
    lru_.erase(f->second->lru);
    lru_.push_back(k);
    f->second->lru = std::prev(lru_.end());
    return Lease(f->second);
  }
  ++stats_.misses;

  ExpertPayload x;
  try {
    if (!p(k, x, e)) {
      if (e.empty())
        e = "expert cache provider failed";
      return {};
    }
  } catch (const std::exception &ex) {
    set_callback_error(e, "provider", &ex);
    return {};
  } catch (...) {
    set_callback_error(e, "provider");
    return {};
  }
  if (!validate_payload(x, e))
    return {};
  const auto n = x.bytes();
  if (!make_room(n, e))
    return {};

  BackendValuePtr u;
  try {
    u = backend_->upload(k, x, e);
  } catch (const std::exception &ex) {
    set_callback_error(e, "backend upload", &ex);
    return {};
  } catch (...) {
    set_callback_error(e, "backend upload");
    return {};
  }
  if (!u) {
    if (e.empty())
      e = "expert cache backend upload failed";
    return {};
  }

  try {
    auto q = std::make_shared<Entry>();
    q->key = k;
    q->payload = std::move(x);
    q->backend = std::move(u);
    lru_.push_back(k);
    q->lru = std::prev(lru_.end());
    try {
      entries_.emplace(q->key, q);
    } catch (...) {
      lru_.erase(q->lru);
      throw;
    }
    stats_.bytes += n;
    stats_.entries = entries_.size();
    return Lease(std::move(q));
  } catch (const std::exception &ex) {
    set_callback_error(e, "admission", &ex);
    return {};
  } catch (...) {
    set_callback_error(e, "admission");
    return {};
  }
}

ExpertCache::Stats ExpertCache::stats() const {
  std::lock_guard lock(mutex_);
  auto s = stats_;
  s.entries = entries_.size();
  return s;
}

void ExpertCache::clear() {
  std::lock_guard lock(mutex_);
  for (auto i = lru_.begin(); i != lru_.end();) {
    auto f = entries_.find(*i);
    if (f == entries_.end()) {
      i = lru_.erase(i);
      continue;
    }
    if (f->second.use_count() != 1) {
      ++i;
      continue;
    }
    stats_.bytes -= f->second->payload.bytes();
    entries_.erase(f);
    i = lru_.erase(i);
  }
  stats_.entries = entries_.size();
}
} // namespace strata::ring
