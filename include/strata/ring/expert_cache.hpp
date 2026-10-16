#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
namespace strata::ring {
struct ExpertKey {
  std::string checkpoint;
  std::uint32_t layer = 0, expert = 0;
  bool operator==(const ExpertKey &o) const noexcept {
    return layer == o.layer && expert == o.expert && checkpoint == o.checkpoint;
  }
};
struct ExpertKeyHash {
  std::size_t operator()(const ExpertKey &) const noexcept;
};
struct ExpertShape {
  std::uint32_t rows = 0, cols = 0;
};
struct ExpertPayload {
  ExpertShape gate_shape, up_shape, down_shape;
  std::vector<std::uint8_t> gate, up, down;
  std::size_t bytes() const noexcept {
    return gate.size() + up.size() + down.size();
  }
};
class ExpertCache {
  struct Entry;

public:
  class BackendValue {
  public:
    virtual ~BackendValue() = default;
  };
  using BackendValuePtr = std::shared_ptr<const BackendValue>;
  class Backend {
  public:
    virtual ~Backend() =
        default; /// Upload is called only after cache room is made; a null
                 /// result must release any partial backend allocation.
    virtual BackendValuePtr upload(const ExpertKey &, const ExpertPayload &,
                                   std::string &) = 0;
  };
  using Provider =
      std::function<bool(const ExpertKey &, ExpertPayload &, std::string &)>;
  struct Config {
    std::size_t byte_budget = 0;
    std::uint32_t max_layers = 0, max_experts = 0;
  };
  struct Stats {
    std::uint64_t hits = 0, misses = 0, evictions = 0;
    std::size_t bytes = 0, entries = 0;
  };
  class Lease {
  public:
    Lease() = default;
    bool valid() const noexcept { return !!entry_; }
    const ExpertKey &key() const;
    const ExpertPayload &payload() const;
    BackendValuePtr backend_value() const;
    std::size_t bytes() const noexcept;

  private:
    using EntryPtr = std::shared_ptr<ExpertCache::Entry>;
    explicit Lease(EntryPtr e) : entry_(std::move(e)) {}
    EntryPtr entry_;
    friend class ExpertCache;
  };
  /// clear removes unleased entries; entries held by Lease remain resident and
  /// counted until their last Lease dies.
  ExpertCache(Config, std::shared_ptr<Backend>);
  ExpertCache(const ExpertCache &) = delete;
  ExpertCache &operator=(const ExpertCache &) = delete;
  ~ExpertCache();
  std::optional<Lease> acquire(const ExpertKey &, const Provider &,
                               std::string &);
  Stats stats() const;
  void clear();

private:
  struct Entry {
    ExpertKey key;
    ExpertPayload payload;
    BackendValuePtr backend;
    std::list<ExpertKey>::iterator lru;
  };
  using Map =
      std::unordered_map<ExpertKey, std::shared_ptr<Entry>, ExpertKeyHash>;
  bool validate_key(const ExpertKey &, std::string &) const;
  static bool validate_payload(const ExpertPayload &, std::string &);
  bool make_room(std::size_t, std::string &);
  Config config_;
  std::shared_ptr<Backend> backend_;
  mutable std::mutex mutex_;
  Map entries_;
  std::list<ExpertKey> lru_;
  Stats stats_;
};
} // namespace strata::ring
