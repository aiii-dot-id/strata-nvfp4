// AI Identity Incorported <james@aiii.id>
#include <strata/ring/continuation.hpp>
#include <strata/ring/generation.hpp>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>
using namespace strata::ring;
template<class F> void rejects(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}assert(failed);}
int main(){
 std::vector<uint8_t> crc_probe={'1','2','3','4','5','6','7','8','9'};assert(continuation_detail::checksum(crc_probe,crc_probe.size())==UINT64_C(0xcbf43926));
 auto root=std::filesystem::temp_directory_path()/("strata-continuation-"+std::to_string(getpid()));
 assert(std::filesystem::create_directory(root));
 try{
  ContinuationState a;a.binding=std::string(64,'a');a.arrays["layer.0.k"]={};
  save_continuation(root/"empty",a);auto empty=load_continuation(root/"empty");require_continuation(empty,a.binding,{1,2},32);assert(empty.position==0);
  a.position=2;a.prefix={1,2};a.last_logits.assign(157184,0);a.last_logits[3]=1;a.arrays["layer.0.k"]={1,-2,3.25f};
  save_continuation(root/"full",a);auto b=load_continuation(root/"full");assert(b.prefix==a.prefix&&b.last_logits==a.last_logits&&b.arrays==a.arrays);require_continuation(b,a.binding,{1,2,9},32);
  auto fork=b;fork.arrays["layer.0.k"][0]=99;assert(b.arrays["layer.0.k"][0]==1);
  auto rollback=load_continuation(root/"full");assert(rollback.arrays==a.arrays);
  rejects([&]{require_continuation(b,std::string(64,'b'),{1,2},32);});rejects([&]{require_continuation(b,a.binding,{1,9},32);});rejects([&]{require_continuation(b,a.binding,{1,2},1);});
  rejects([&]{save_continuation(root/"full",a);});
  auto bad=a;bad.arrays["layer.0.k"][0]=std::numeric_limits<float>::quiet_NaN();rejects([&]{save_continuation(root/"nan",bad);});assert(!std::filesystem::exists(root/"nan"));
  std::ifstream in(root/"full",std::ios::binary);std::vector<char> data((std::istreambuf_iterator<char>(in)),{});data[data.size()/2]^=1;std::ofstream broken(root/"broken",std::ios::binary);broken.write(data.data(),data.size());broken.close();rejects([&]{load_continuation(root/"broken");});
  std::ofstream truncated(root/"truncated");truncated<<"x";truncated.close();rejects([&]{load_continuation(root/"truncated");});
  std::filesystem::create_symlink(root/"full",root/"link");rejects([&]{load_continuation(root/"link");});
  GreedyGeneration full({1,2},3,{3},4,32);std::vector<float> l={2,0,0,0};full.sample(1,l);
  GreedyGeneration resumed({1,2},3,{3},4,32);resumed.resume_prefix({1,2},l);assert(resumed.generated()==full.generated()&&resumed.evaluated()==full.evaluated());
  l={0,0,0,3};full.sample(2,l);resumed.sample(2,l);assert(full.receipt()==resumed.receipt());
  GreedyGeneration prefill({1,2},3,{3},4,32);prefill.resume_prefix({},{});prefill.resume_prefix({1},{});assert(prefill.evaluated()==std::vector<int>({1,2}));
  auto old=prefill.evaluated();auto oldgenerated=prefill.generated();l[1]=std::numeric_limits<float>::infinity();rejects([&]{prefill.resume_prefix({1,2},l);});assert(prefill.evaluated()==old&&prefill.generated()==oldgenerated);
  rejects([&]{prefill.resume_prefix({1,2,3},std::vector<float>(4));});
  GreedyGeneration branch({1,2,0,1},3,{3},4,32);branch.resume_prefix({1,2},{});assert(branch.generated().empty()&&branch.evaluated()==std::vector<int>({1,2,0,1}));
  std::filesystem::remove_all(root);std::cout<<"Continuation serialization, fork/rollback, corruption and generation resume controls passed\n";
 }catch(...){std::filesystem::remove_all(root);throw;}
}
