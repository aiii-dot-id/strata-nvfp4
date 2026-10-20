import contextlib,importlib.util,io,os,sys,tempfile,types
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]; TOOLS=REPO/"tools/ling_ring_bf16"
def load(kind, monkeypatch):
 name=f"run_{kind}_verified"; mod=types.ModuleType(name); mod.VerificationError=RuntimeError; mod._file=lambda p,*a:Path(p); mod._read_json=lambda *a:{"eos_token_id":[3]}; mod._sha=lambda *a:"test"; mod.run_verified=lambda *a,**k:None; mod._require_generation=lambda *a,**k:None; mod.VOCAB=157184; mod.MAX_TOKENS=1024
 with monkeypatch.context() as patcher:
  patcher.setitem(sys.modules,name,mod)
  spec=importlib.util.spec_from_file_location(name,TOOLS/f"generate_{kind}.py")
  m=importlib.util.module_from_spec(spec)
  spec.loader.exec_module(m)
 return m,mod
class Tok:
 def apply_chat_template(self,*a,**k):return [42]
 def decode(self,*a,**k):return "ok"
def test_frontend_forwards_workers_and_records(monkeypatch):
 for kind in ("ring","ling"):
  m,r=load(kind, monkeypatch); seen={}
  def f(*a,**k):seen.update(k);return {"generation":{"generated_token_ids":[3],"finish_reason":"eos"},"frontend":{}}
  m.run_verified=f;m._binding_paths=lambda *_:[];m._binding_snapshot=lambda *_:{};m._snapshot=lambda *_:{};m._eos_ids=(lambda *_:[3]) if kind=="ring" else (lambda *_:([3],"test"));m._tokenizer=lambda *_:Tok()
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)/"out";out.mkdir() if kind=="ring" else None;m.generate(Path(td),Path(td),Path(td),"p",out,6 if kind=="ring" else 5,1,hash_workers=4)
  assert seen["hash_workers"]==4
def test_cli_rejects_out_of_range(monkeypatch):
 for kind in ("ring","ling"):
  m,_=load(kind, monkeypatch); old=sys.argv;sys.argv=["gen","--checkpoint","/tmp/c","--manifest","/tmp/m","--executable","/tmp/e","--prompt","p","--output","/tmp/o","--cache-mib","6" if kind=="ring" else "5","--max-new-tokens","1","--hash-workers","5"]
  try:
   with contextlib.redirect_stderr(io.StringIO()):
    try:m.main()
    except SystemExit as e:assert e.code==2
    else:raise AssertionError("accepted worker 5")
  finally:sys.argv=old

def test_actual_cli_forwards_valid_workers(monkeypatch):
 import inspect
 for kind in ("ring","ling"):
  m,_=load(kind,monkeypatch)
  signature=inspect.signature(m.generate)
  seen={}
  def capture(*args,**kwargs):
   seen.update(signature.bind(*args,**kwargs).arguments)
   return {"frontend":{"decoded":"ok"}}
  m.generate=capture
  monkeypatch.setattr(sys,"argv",["gen","--checkpoint","/unused/c","--manifest","/unused/m","--executable","/unused/e","--prompt","p","--output","/unused/o","--cache-mib","6" if kind=="ring" else "5","--max-new-tokens","1","--hash-workers","4"])
  with contextlib.redirect_stdout(io.StringIO()):m.main()
  assert seen["hash_workers"]==4
