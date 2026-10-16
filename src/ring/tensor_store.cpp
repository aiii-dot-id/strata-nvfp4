#include "strata/ring/tensor_store.hpp"
#include <climits>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
namespace strata::ring {
namespace {
struct Map {
  int fd = -1;
  size_t n = 0;
  const uint8_t *p = nullptr;
  ~Map() {
    if (p)
      ::munmap(const_cast<uint8_t *>(p), n);
    if (fd >= 0)
      ::close(fd);
  }
};
std::string err(const std::string &s) { return "tensor store: " + s; }
std::vector<std::string> split(const std::string &s) {
  std::vector<std::string> v;
  size_t a = 0;
  for (;;) {
    auto p = s.find('\t', a);
    if (p == std::string::npos) {
      v.push_back(s.substr(a));
      return v;
    }
    v.push_back(s.substr(a, p - a));
    a = p + 1;
  }
}
size_t num(const std::string &s) {
  if (s.empty() || s[0] == '-')
    throw std::invalid_argument(err("invalid number"));
  size_t p = 0;
  unsigned long long x = std::stoull(s, &p);
  if (p != s.size() || x > std::numeric_limits<size_t>::max())
    throw std::invalid_argument(err("invalid number"));
  return (size_t)x;
}
std::vector<size_t> shape(const std::string &s) {
  if (s.empty())
    throw std::invalid_argument(err("empty shape"));
  std::vector<size_t> v;
  size_t a = 0;
  for (;;) {
    auto p = s.find(',', a);
    auto q = s.substr(a, p == std::string::npos ? s.size() - a : p - a);
    v.push_back(num(q));
    if (p == std::string::npos)
      return v;
    a = p + 1;
  }
}
size_t prod(const std::vector<size_t> &v) {
  size_t n = 1;
  for (auto x : v) {
    if (x && n > SIZE_MAX / x)
      throw std::invalid_argument(err("shape overflow"));
    n *= x;
  }
  return n;
}
void safe(const std::string &s) {
  if (s.empty() || s[0] == '/' || s[0] == '\\')
    throw std::invalid_argument(err("unsafe shard path"));
  size_t a = 0;
  for (;;) {
    auto p = s.find('/', a);
    auto q = s.substr(a, p == std::string::npos ? s.size() - a : p - a);
    if (q.empty() || q == "." || q == "..")
      throw std::invalid_argument(err("unsafe shard path"));
    if (p == std::string::npos)
      return;
    a = p + 1;
  }
}
std::string root(const std::string &s) {
  char b[PATH_MAX];
  if (!realpath(s.c_str(), b))
    throw std::runtime_error(err("bad root"));
  struct stat st{};
  if (stat(b, &st) || !S_ISDIR(st.st_mode))
    throw std::runtime_error(err("root is not directory"));
  return b;
}
std::string path(const std::string &r, const std::string &x) {
  safe(x);
  std::string p = r + "/" + x;
  size_t a = r.size() + 1;
  for (;;) {
    auto q = p.find('/', a);
    auto z = q == std::string::npos ? p.size() : q;
    struct stat st{};
    if (lstat(p.substr(0, z).c_str(), &st) || S_ISLNK(st.st_mode))
      throw std::invalid_argument(err("unsafe shard component"));
    if (q == std::string::npos)
      return p;
    a = q + 1;
  }
}
} // namespace
struct TensorStore::Impl {
  std::unordered_map<std::string, std::unique_ptr<Map>> maps;
  std::unordered_map<std::string, TensorView> views;
};
TensorStore::TensorStore(std::string rd, std::string mf)
    : impl_(std::make_unique<Impl>()) {
  auto r = root(rd);
  std::ifstream in(mf);
  if (!in)
    throw std::runtime_error(err("cannot open manifest"));
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    auto f = split(line);
    if (f.size() != 7 || f[0].empty() || impl_->views.count(f[0]))
      throw std::invalid_argument(err("bad manifest row"));
    size_t w = f[1] == "BF16" ? 2 : f[1] == "F32" ? 4 : 0;
    if (!w)
      throw std::invalid_argument(err("unsupported dtype"));
    auto sh = shape(f[2]);
    auto n = prod(sh), off = num(f[4]), bytes = num(f[5]), expected = num(f[6]);
    if (n > SIZE_MAX / w || bytes != n * w || off < 8)
      throw std::invalid_argument(err("bad tensor size"));
    auto fn = path(r, f[3]);
    auto i = impl_->maps.find(fn);
    if (i == impl_->maps.end()) {
      auto m = std::make_unique<Map>();
      m->fd = open(fn.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
      if (m->fd < 0)
        throw std::runtime_error(err("cannot open shard"));
      struct stat st{};
      if (fstat(m->fd, &st) || size_t(st.st_size) != expected)
        throw std::invalid_argument(err("shard size mismatch"));
      m->n = expected;
      m->p = (const uint8_t *)mmap(nullptr, m->n, PROT_READ, MAP_PRIVATE, m->fd,
                                   0);
      if (m->p == MAP_FAILED) {
        m->p = nullptr;
        throw std::runtime_error(err("cannot map shard"));
      }
      i = impl_->maps.emplace(fn, std::move(m)).first;
    } else if (i->second->n != expected)
      throw std::invalid_argument(err("inconsistent shard size"));
    if (off > i->second->n || bytes > i->second->n - off)
      throw std::invalid_argument(err("tensor range outside shard"));
    TensorView v{f[1], std::move(sh), i->second->p + off, bytes};
    impl_->views.emplace(f[0], std::move(v));
  }
}
TensorStore::TensorStore(TensorStore &&) noexcept = default;
TensorStore &TensorStore::operator=(TensorStore &&) noexcept = default;
TensorStore::~TensorStore() = default;
const TensorView &TensorStore::tensor(std::string_view n) const {
  auto i = impl_->views.find(std::string(n));
  if (i == impl_->views.end())
    throw std::out_of_range(err("tensor not found"));
  return i->second;
}
size_t TensorStore::size() const noexcept { return impl_->views.size(); }
} // namespace strata::ring
