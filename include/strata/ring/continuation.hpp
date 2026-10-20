// AI Identity Incorported <james@aiii.id>
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::ring {
// Exact live model state. Expert/head residency is performance state only.
struct ContinuationState {
  std::string binding;
  uint64_t position = 0;
  std::vector<int> prefix;
  std::vector<float> last_logits;
  std::map<std::string, std::vector<float>> arrays;
};
namespace continuation_detail {
constexpr uint64_t max_bytes = uint64_t(512) * 1024 * 1024;
inline uint64_t checksum(const std::vector<uint8_t>& data, size_t n) {
  // CRC32/IEEE detects accidental transport corruption. The verified wrapper
  // checks the same CRC and separately binds the payload with SHA-256.
  static const std::array<uint32_t,256> table=[] {
    std::array<uint32_t,256> t{};
    for(uint32_t i=0;i<256;++i){uint32_t c=i;for(int j=0;j<8;++j)c=(c&1)?(UINT32_C(0xedb88320)^(c>>1)):(c>>1);t[i]=c;}return t;
  }();
  uint32_t crc=UINT32_C(0xffffffff);
  for(size_t i=0;i<n;++i)crc=table[(crc^data[i])&255]^(crc>>8);
  return crc^UINT32_C(0xffffffff);
}
inline void u64(std::vector<uint8_t>& b,uint64_t n) {
  for (int i=0;i<8;++i) b.push_back(uint8_t(n>>(i*8)));
}
inline void text(std::vector<uint8_t>& b,const std::string& s) {
  u64(b,s.size()); b.insert(b.end(),s.begin(),s.end());
}
inline void floats(std::vector<uint8_t>& b,const std::vector<float>& x) {
  if (x.size() > max_bytes/4 || b.size()+8+x.size()*4>max_bytes)
    throw std::runtime_error("continuation payload exceeds bound");
  u64(b,x.size());
  for (float v:x) {
    if (!std::isfinite(v)) throw std::runtime_error("nonfinite continuation state");
    uint32_t bits; std::memcpy(&bits,&v,4);
    for (int j=0;j<4;++j) b.push_back(uint8_t(bits>>(j*8)));
  }
}
inline bool hex64(const std::string& s) {
  return s.size()==64 && s.find_first_not_of("0123456789abcdef")==std::string::npos;
}
struct Reader {
  const std::vector<uint8_t>& b; size_t p=0,end;
  explicit Reader(const std::vector<uint8_t>& data):b(data),end(data.size()-8){}
  uint64_t number() {
    if(p>end || end-p<8) throw std::runtime_error("truncated continuation number");
    uint64_t n=0;for(int i=0;i<8;++i)n|=uint64_t(b[p++])<<(i*8);return n;
  }
  std::string string(size_t limit) {
    auto n=number();if(n>limit || n>end-p)throw std::runtime_error("invalid continuation string length");
    std::string s(b.begin()+p,b.begin()+p+n);p+=n;return s;
  }
  std::vector<float> vector() {
    auto n=number();if(n>(end-p)/4)throw std::runtime_error("invalid continuation array length");
    std::vector<float> out;out.reserve(n);
    for(uint64_t i=0;i<n;++i){uint32_t bits=0;for(int j=0;j<4;++j)bits|=uint32_t(b[p++])<<(j*8);float x;std::memcpy(&x,&bits,4);if(!std::isfinite(x))throw std::runtime_error("nonfinite continuation state");out.push_back(x);}return out;
  }
};
}
inline void require_continuation(const ContinuationState& s,
    const std::string& binding,const std::vector<int>& prompt,
    size_t capacity,int vocab=157184) {
  if(!continuation_detail::hex64(binding) || s.binding!=binding)
    throw std::runtime_error("continuation binding mismatch");
  if(s.position!=s.prefix.size() || s.position>capacity || s.position>4096)
    throw std::runtime_error("continuation position mismatch");
  for(int x:s.prefix)if(x<0 || x>=vocab)throw std::runtime_error("invalid continuation token");
  const size_t shared=std::min(s.prefix.size(),prompt.size());
  if(!std::equal(s.prefix.begin(),s.prefix.begin()+shared,prompt.begin()))
    throw std::runtime_error("continuation prefix mismatch");
  if((s.position && s.last_logits.size()!=size_t(vocab)) || (!s.position && !s.last_logits.empty()))
    throw std::runtime_error("continuation logits shape mismatch");
  for(float x:s.last_logits)if(!std::isfinite(x))throw std::runtime_error("nonfinite continuation logits");
}
inline ContinuationState load_continuation(const std::filesystem::path& path) {
  namespace fs=std::filesystem;using namespace continuation_detail;
  if(fs::is_symlink(path)||!fs::is_regular_file(path))throw std::runtime_error("continuation must be a regular file");
  auto n=fs::file_size(path);if(n<32 || n>max_bytes)throw std::runtime_error("invalid continuation file size");
  std::ifstream in(path,std::ios::binary);std::vector<uint8_t>b(n);
  if(!in.read(reinterpret_cast<char*>(b.data()),n))throw std::runtime_error("continuation read failed");
  uint64_t expected=0;for(int j=0;j<8;++j)expected|=uint64_t(b[n-8+j])<<(j*8);
  if(checksum(b,n-8)!=expected)throw std::runtime_error("continuation checksum mismatch");
  Reader r(b);if(r.string(16)!="STRATA-CONT-1")throw std::runtime_error("unsupported continuation schema");
  ContinuationState s;s.binding=r.string(64);if(!hex64(s.binding))throw std::runtime_error("invalid continuation binding");
  s.position=r.number();auto count=r.number();if(count!=s.position||count>4096)throw std::runtime_error("invalid continuation prefix length");
  for(uint64_t i=0;i<count;++i){auto x=r.number();if(x>=157184)throw std::runtime_error("invalid continuation token");s.prefix.push_back(int(x));}
  s.last_logits=r.vector();auto arrays=r.number();if(arrays>128)throw std::runtime_error("too many continuation arrays");
  for(uint64_t i=0;i<arrays;++i){auto name=r.string(128);if(name.empty()||name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")!=std::string::npos)throw std::runtime_error("invalid continuation array name");auto v=r.vector();if(!s.arrays.emplace(name,std::move(v)).second)throw std::runtime_error("duplicate continuation array");}
  if(r.p!=r.end)throw std::runtime_error("trailing continuation payload");
  return s;
}
inline void save_continuation(const std::filesystem::path& path,const ContinuationState& s) {
  namespace fs=std::filesystem;using namespace continuation_detail;
  require_continuation(s,s.binding,s.prefix,4096);
  if(s.arrays.size()>128)throw std::runtime_error("too many continuation arrays");
  std::vector<uint8_t>b;text(b,"STRATA-CONT-1");text(b,s.binding);u64(b,s.position);u64(b,s.prefix.size());
  for(int x:s.prefix)u64(b,x);floats(b,s.last_logits);u64(b,s.arrays.size());
  for(const auto& item:s.arrays){if(item.first.empty()||item.first.size()>128||item.first.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")!=std::string::npos)throw std::runtime_error("invalid continuation array name");text(b,item.first);floats(b,item.second);}
  if(b.size()+8>max_bytes)throw std::runtime_error("continuation payload exceeds bound");u64(b,checksum(b,b.size()));
  if(fs::exists(path)||fs::is_symlink(path))throw std::runtime_error("refusing existing continuation output");
  const fs::path temp=path.string()+".partial";
  if(!fs::create_directory(temp))throw std::runtime_error("continuation temporary path exists");
  try {
    const fs::path file=temp/"payload";std::ofstream out(file,std::ios::binary);out.write(reinterpret_cast<const char*>(b.data()),b.size());out.flush();if(!out)throw std::runtime_error("continuation write failed");out.close();
    fs::create_hard_link(file,path); // Atomic create; never replaces an existing bundle.
    fs::remove_all(temp);
  } catch(...) {fs::remove_all(temp);throw;}
}
}
