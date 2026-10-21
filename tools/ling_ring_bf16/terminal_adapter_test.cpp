#include <strata/ring/terminal_adapter.hpp>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

static void u32(std::ofstream &f, uint32_t v) { f.write(reinterpret_cast<const char *>(&v), 4); }
static void f32(std::ofstream &f, float v) { f.write(reinterpret_cast<const char *>(&v), 4); }
static void zeros(std::ofstream &f, size_t n) { for (size_t i=0; i<n; ++i) f32(f, 0.0f); }

int main() {
  const auto path = std::filesystem::temp_directory_path() / "strata-terminal-adapter-test.bin";
  { std::ofstream f(path, std::ios::binary|std::ios::trunc); f.write("STRALOR1",8); u32(f,1); u32(f,2); u32(f,1); f32(f,2.0f);
    f32(f,1); f32(f,0); // gate A
    f32(f,3); zeros(f,511); // gate B
    f32(f,0); f32(f,1); // up A: row 1, col 1
    f32(f,4); zeros(f,511); // up B
    f32(f,1); zeros(f,511); // down A
    f32(f,5); f32(f,0); // down B
  }
  auto a = strata::ring::TerminalAdapter::load(path.string(),1,2,128);
  std::vector<float> x{1,2}, g(512,0), u(512,0);
  a.add_gate_up(x,g,u); assert(g[0]==6.0f && u[0]==16.0f);
  std::vector<float> act(512,0); act[0]=2; std::vector<float> y(2,0); a.add_down(act,y); assert(y[0]==20.0f && y[1]==0.0f);
  std::filesystem::remove(path);

  const auto zero_path = std::filesystem::temp_directory_path() / "strata-terminal-adapter-zero.bin";
  { std::ofstream f(zero_path, std::ios::binary|std::ios::trunc); f.write("STRALOR1",8); u32(f,1); u32(f,0); u32(f,1); f32(f,16);
    zeros(f,2); zeros(f,512); zeros(f,2); zeros(f,512); zeros(f,512); zeros(f,2); }
  auto z = strata::ring::TerminalAdapter::load(zero_path.string(),1,2,128);
  g.assign(512,7); u.assign(512,9); act.assign(512,3); y.assign(2,4); z.add_gate_up(x,g,u); z.add_down(act,y);
  assert(g[0]==7 && u[0]==9 && y[0]==4); std::filesystem::remove(zero_path);

  const auto bad = std::filesystem::temp_directory_path() / "strata-terminal-adapter-bad.bin";
  { std::ofstream f(bad, std::ios::binary|std::ios::trunc); f.write("STRALOR1",8); u32(f,1); u32(f,0); u32(f,1); f32(f,16); zeros(f,2+512+2+512+512+2); f32(f,1); }
  bool rejected=false; try { (void)strata::ring::TerminalAdapter::load(bad.string(),1,2,128); } catch (...) { rejected=true; }
  assert(rejected); std::filesystem::remove(bad); return 0;
}
