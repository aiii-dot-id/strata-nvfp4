// Copyright (c) AI Identity Incorported <james@aiii.id>.
#include "router.hpp"
#include <torch/torch.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
namespace strata::torch_router {
struct Router::Impl { std::size_t experts,groups,top_groups,top_k; float scale; c10::Device device; Impl(std::size_t e,std::size_t g,std::size_t tg,std::size_t k,float s,const char*d):experts(e),groups(g),top_groups(tg),top_k(k),scale(s),device(c10::kCPU){ if(!d) throw std::invalid_argument("router device is null"); std::string dev(d); if(dev=="cuda") device=c10::kCUDA; else if(dev!="cpu") throw std::invalid_argument("router device must be cpu or cuda"); } };
Router::Router(std::size_t e,std::size_t g,std::size_t tg,std::size_t k,float s,const char*d):impl_(std::make_unique<Impl>(e,g,tg,k,s,d)){}
Router::~Router()=default;
bool Router::route(const float *lp,const float *bp,std::size_t count,std::vector<std::size_t>&ids,std::vector<float>&weights,std::vector<float>&scores,std::vector<float>&corrected,std::string&error){
 try { ids.clear(); weights.clear(); scores.clear(); corrected.clear(); auto &x=*impl_; if(!lp || !bp) throw std::invalid_argument("null Torch router input"); if(count!=x.experts||x.groups==0||x.experts%x.groups||x.experts/x.groups<2||x.top_groups==0||x.top_groups>x.groups||x.top_k==0||x.top_k>x.top_groups*(x.experts/x.groups)||!std::isfinite(x.scale)) throw std::invalid_argument("invalid Torch router geometry");
  for(size_t i=0;i<count;i++) if(!std::isfinite(lp[i])||!std::isfinite(bp[i])) throw std::invalid_argument("nonfinite Torch router input");
  auto o=torch::TensorOptions().dtype(torch::kFloat32); auto l=torch::from_blob(const_cast<float*>(lp),{1,(long)x.experts},o).clone(); auto b=torch::from_blob(const_cast<float*>(bp),{1,(long)x.experts},o).clone(); l=l.to(x.device); b=b.to(x.device);
  auto sc=torch::sigmoid(l), routed=sc+b; auto grouped=routed.view({1,(long)x.groups,(long)(x.experts/x.groups)}); auto gs=std::get<0>(torch::topk(grouped,2,-1,true,true)).sum(-1); auto gi=std::get<1>(torch::topk(gs,(long)x.top_groups,-1,true,false)); auto mask=torch::zeros_like(gs); mask.scatter_(1,gi,1); auto expanded=mask.unsqueeze(-1).expand({1,(long)x.groups,(long)(x.experts/x.groups)}).reshape({1,(long)x.experts}); auto masked=torch::where(expanded.to(torch::kBool),routed,torch::full_like(routed,-std::numeric_limits<float>::infinity())); auto top=torch::topk(masked,(long)x.top_k,1,true,true); auto ti=std::get<1>(top); auto tw=torch::gather(sc,1,ti); if(x.top_k>1) tw=tw/(tw.sum(-1,true)+1e-20); tw*=x.scale;
  auto sc_cpu=sc.to(torch::kCPU).contiguous(); auto rc_cpu=routed.to(torch::kCPU).contiguous(); auto ti_cpu=ti.to(torch::kCPU).contiguous(); auto tw_cpu=tw.to(torch::kCPU).contiguous(); auto sa=sc_cpu.accessor<float,2>(); auto ra=rc_cpu.accessor<float,2>(); auto ia=ti_cpu.accessor<int64_t,2>(); auto wa=tw_cpu.accessor<float,2>(); std::vector<float> ns(count), nc(count), nw(x.top_k); std::vector<std::size_t> ni(x.top_k); for(size_t i=0;i<count;i++){ns[i]=sa[0][i];nc[i]=ra[0][i]; if(!std::isfinite(ns[i])||!std::isfinite(nc[i])) throw std::runtime_error("nonfinite Torch router output");} for(size_t i=0;i<x.top_k;i++){if(ia[0][i]<0 || std::size_t(ia[0][i])>=count || !std::isfinite(wa[0][i])) throw std::runtime_error("invalid Torch router selection"); ni[i]=size_t(ia[0][i]); nw[i]=wa[0][i];} scores=std::move(ns); corrected=std::move(nc); ids=std::move(ni); weights=std::move(nw); return true;
 } catch(const std::exception&e){error=e.what();return false;}
}
}
