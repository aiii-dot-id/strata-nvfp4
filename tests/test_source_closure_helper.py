import importlib.util,os,sys,tempfile,types
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]; TOOLS=REPO/"tools/ling_ring_bf16"
def load(kind, monkeypatch):
 name="run_"+kind+"_verified"
 monkeypatch.syspath_prepend(str(TOOLS))
 spec=importlib.util.spec_from_file_location(name,TOOLS/(name+".py"))
 m=importlib.util.module_from_spec(spec)
 monkeypatch.setitem(sys.modules,name,m)
 spec.loader.exec_module(m)
 return m
def touch(repo,items):
 for rel in items:
  p=repo/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(str(rel))
def test_helper_in_both_closures_and_mutation_detected(tmp_path, monkeypatch):
 for kind in ("ring","ling"):
  m=load(kind, monkeypatch);repo=tmp_path/(kind+"repo");ck=tmp_path/(kind+"ck");repo.mkdir();ck.mkdir();touch(repo,[Path("tools/ling_ring_bf16/verified_hash.py")])
  if kind=="ring":
   touch(ck,[Path("modeling_bailing_moe_v2.py"),Path("configuration_bailing_moe_v2.py")]);items=[Path("include/strata/ring/r.hpp"),Path("include/strata/kernels/bf16_gemv.hpp"),Path("include/strata/kernels/bf16_bits.hpp"),Path("include/strata/kernels/pdl.hpp"),Path("src/ring/r.cpp"),Path("src/kernels/cuda/native_bf16.cu"),Path("src/kernels/cuda/s26_tsum.cuh"),Path("tools/ling_ring_bf16/CMakeLists.txt"),Path("tools/ling_ring_bf16/inspect_checkpoint.py"),Path("tools/ling_ring_bf16/prepare_ring_manifest.py"),Path("tools/ling_ring_bf16/ring_original_reference.py"),Path("tools/ling_ring_bf16/compare_ring_reference.py"),Path("tools/ling_ring_bf16/run_ring_verified.py"),Path("tools/ling_ring_bf16/generate_ring.py")];fn=m._source_files
  else:
   touch(ck,[Path("modeling_bailing_moe_v3.py"),Path("configuration_bailing_moe_v3.py")]);items=[Path("include/strata/ling/l.hpp"),Path("src/ling/l.cpp"),Path("include/strata/ring/expert_cache.hpp"),Path("include/strata/ring/generation.hpp"),Path("include/strata/ring/numerics.hpp"),Path("include/strata/ring/routing.hpp"),Path("include/strata/ring/tensor_store.hpp"),Path("src/ring/routing.cpp"),Path("src/ring/expert_cache.cpp"),Path("src/ring/tensor_store.cpp"),Path("include/strata/kernels/bf16_gemv.hpp"),Path("include/strata/kernels/bf16_bits.hpp"),Path("include/strata/kernels/pdl.hpp"),Path("src/kernels/cuda/native_bf16.cu"),Path("src/kernels/cuda/s26_tsum.cuh"),Path("tools/ling_ring_bf16/prepare_ling_manifest.py"),Path("tools/ling_ring_bf16/ling/CMakeLists.txt"),Path("tools/ling_ring_bf16/ling/ling_torch_ops.hpp"),Path("tools/ling_ring_bf16/ling/ling_torch_ops.cpp"),Path("tools/ling_ring_bf16/run_ling_verified.py"),Path("tools/ling_ring_bf16/generate_ling.py")];fn=m._source_files
  touch(repo,items);key="tools/ling_ring_bf16/verified_hash.py";_,a=fn(ck,repo);assert key in a;before=a[key]["sha256"];(repo/key).write_text("changed");_,b=fn(ck,repo);assert b[key]["sha256"]!=before
