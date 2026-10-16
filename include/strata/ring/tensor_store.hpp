#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
namespace strata::ring {
struct TensorView {
  std::string dtype;
  std::vector<std::size_t> shape;
  const std::uint8_t *data = nullptr;
  std::size_t bytes = 0;
};
class TensorStore {
public:
  TensorStore(std::string root, std::string manifest);
  TensorStore(const TensorStore &) = delete;
  TensorStore &operator=(const TensorStore &) = delete;
  TensorStore(TensorStore &&) noexcept;
  TensorStore &operator=(TensorStore &&) noexcept;
  ~TensorStore();
  const TensorView &tensor(std::string_view name) const;
  std::size_t size() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace strata::ring
