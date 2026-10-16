#include "strata/ring/tensor_store.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace strata::ring;
void wr(const std::filesystem::path &p, const std::string &s) {
  std::ofstream f(p, std::ios::binary);
  f << s;
}
template <class F> void bad(F f) {
  bool x = false;
  try {
    f();
  } catch (...) {
    x = true;
  }
  assert(x);
}
int main() {
  char q[] = "/tmp/ts-XXXXXX";
  int fd = mkstemp(q);
  close(fd);
  unlink(q);
  std::filesystem::create_directory(q);
  std::filesystem::path d = q;
  wr(d / "a.bin", "HHHHHHHHabcd");
  wr(d / "m.tsv", "x\tBF16\t2,1\ta.bin\t8\t4\t12\n");
  TensorStore s(d.string(), (d / "m.tsv").string());
  assert(s.size() == 1 && s.tensor("x").data[0] == 'a' &&
         s.tensor("x").bytes == 4);
  bad([&] { s.tensor("y"); });
  wr(d / "m.tsv", "x\tF16\t2\ta.bin\t8\t4\t12\n");
  bad([&] { TensorStore z(d.string(), (d / "m.tsv").string()); });
  wr(d / "m.tsv", "x\tBF16\t2\t../a.bin\t8\t4\t12\n");
  bad([&] { TensorStore z(d.string(), (d / "m.tsv").string()); });
  std::filesystem::remove_all(d);
  std::cout << "ring_tensor_store_test: ok\n";
}
