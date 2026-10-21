import copy,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent)); import numpy as np, pytest, torch
from bounded_update import Bounds,bounded_update
from capability_update import update
class Toy(torch.nn.Module):
 def __init__(self): super().__init__(); self.proj=torch.nn.Linear(4,5); self.register_buffer('sentinel',torch.zeros(1)); self.max_batch=0; self.mutate_after=False; self.consume_rng=False
 def forward(self,x):
  self.max_batch=max(self.max_batch,int(x.shape[0]))
  if self.mutate_after: self.sentinel.add_(1)
  if self.consume_rng: torch.rand(())
  return self.proj(x)
def records(m,rewards=(0.,0.,1.,0.)):
 torch.manual_seed(22); xs=[torch.randn(2,4) for _ in range(4)]; out=[]
 with torch.no_grad():
  for i,x in enumerate(xs):
   z=m(x).detach(); ids=z.argmax(-1).tolist(); lp=torch.log_softmax(z.double(),-1).gather(1,torch.tensor(ids)[:,None]).squeeze().tolist(); out.append({'group':0 if i<2 else 1,'family':'f','ids':ids,'features':x,'logits':z.numpy(),'sampling_logprobs':lp,'grade':{'reward':rewards[i]}})
 return out
def eq(a,b):
 if torch.is_tensor(a): return torch.equal(a,b)
 if isinstance(a,dict): return set(a)==set(b) and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 return a==b
def state(m,o): return ([p.detach().clone() for p in m.parameters()],[None if p.grad is None else p.grad.detach().clone() for p in m.parameters()],copy.deepcopy(o.state_dict()),torch.get_rng_state().clone(),m.sentinel.clone())
def setup():
 torch.manual_seed(3); m=Toy(); o=torch.optim.Adam(m.parameters(),lr=.01); r=records(m); update(m,o,r,'rlvr',clip_norm=None); [setattr(p,'grad',torch.ones_like(p)) for p in m.parameters()]; return m,o,records(m)
def test_accepts_small_update_and_preserves_objective():
 m,o,r=setup(); out=bounded_update(m,o,r,'rlvr',bounds=Bounds(100,100,100,1),clip_norm=None); assert out['accepted']; assert out['update']['objective']=='rlvr'
def test_acceptance_matches_unwrapped_update_exactly():
 m,o,r=setup(); m2=Toy(); m2.load_state_dict(copy.deepcopy(m.state_dict())); o2=torch.optim.Adam(m2.parameters(),lr=.01); o2.load_state_dict(copy.deepcopy(o.state_dict())); r2=records(m2); baseline=update(m2,o2,r2,'rlvr',clip_norm=None); out=bounded_update(m,o,r,'rlvr',bounds=Bounds(100,100,100,1),clip_norm=None); assert out['accepted']; assert abs(out['update']['loss']-baseline['loss'])<1e-6; assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-6) for a,b in zip(m.parameters(),m2.parameters()))
def test_reject_restores_nonempty_adam_grads_and_rng():
 m,o,r=setup(); torch.manual_seed(99); before=state(m,o); rng=torch.get_rng_state().clone()
 def wrapped(*a,**k): torch.rand(()); return update(*a,**k)
 out=bounded_update(m,o,r,'rlvr',bounds=Bounds(0,0,0,1),clip_norm=None,update_fn=wrapped); assert not out['accepted']; assert eq(before,state(m,o)); assert torch.equal(rng,torch.get_rng_state())
def test_nonfinite_post_update_restores():
 m,o,r=setup(); before=state(m,o)
 def bad(*a,**k):
  z=update(*a,**k); next(m.parameters()).data.fill_(float('nan')); return z
 out=bounded_update(m,o,r,'rlvr',bounds=Bounds(100,100,100,1),clip_norm=None,update_fn=bad); assert not out['accepted']; assert out['reason'].startswith('nonfinite'); assert eq(before,state(m,o))
def test_buffer_mutation_is_fatal_and_not_claimed_restored():
 m,o,r=setup(); before=state(m,o)
 def bad(*a,**k): m.register_buffer('bad',torch.ones(1)); return update(*a,**k)
 with pytest.raises(RuntimeError,match='frozen buffer'): bounded_update(m,o,r,'rlvr',bounds=Bounds(100,100,100,1),clip_norm=None,update_fn=bad)
 assert eq(before[:4],state(m,o)[:4])
def test_post_metric_buffer_mutation_is_fatal():
 m,o,r=setup()
 def bad(*a,**k):
  z=update(*a,**k); m.mutate_after=True; return z
 with pytest.raises(RuntimeError,match='during metric'): bounded_update(m,o,r,'rlvr',bounds=Bounds(100,100,100,1),clip_norm=None,update_fn=bad)
def test_chunked_metrics_invariant_and_bounded_forward_batch():
 m,o,r=setup(); m.max_batch=0; a=bounded_update(m,o,r,'rlvr',bounds=Bounds(100,100,100,1),clip_norm=None); assert m.max_batch<=1; m2,o2,r2=setup(); m2.max_batch=0; b=bounded_update(m2,o2,r2,'rlvr',bounds=Bounds(100,100,100,2),clip_norm=None); assert abs(a['metrics']['mean_kl']-b['metrics']['mean_kl'])<1e-8

def test_cpu_path_never_calls_cuda_rng(monkeypatch):
 m,o,r=setup()
 def fail(*args,**kwargs): raise AssertionError("CUDA RNG touched on CPU path")
 monkeypatch.setattr(torch.cuda,"get_rng_state",fail); monkeypatch.setattr(torch.cuda,"set_rng_state",fail)
 out=bounded_update(m,o,r,"rlvr",bounds=Bounds(100,100,100,1),clip_norm=None)
 assert out["accepted"]

def test_metric_forward_restores_post_update_rng():
 m,o,r=setup(); expected=[None]
 def wrapped(*a,**k):
  z=update(*a,**k); expected[0]=torch.get_rng_state().clone(); m.consume_rng=True; return z
 out=bounded_update(m,o,r,"rlvr",bounds=Bounds(100,100,100,1),clip_norm=None,update_fn=wrapped)
 assert out["accepted"]; assert torch.equal(expected[0],torch.get_rng_state())

def test_nonfinite_receipt_metrics_are_json_safe():
 m,o,r=setup()
 def bad(*a,**k):
  z=update(*a,**k); next(m.parameters()).data.fill_(float("nan")); return z
 out=bounded_update(m,o,r,"rlvr",bounds=Bounds(100,100,100,1),clip_norm=None,update_fn=bad)
 json.dumps(out,allow_nan=False)

def _reference_hash(t):
 import hashlib
 x=t.detach().contiguous().cpu(); x=x.reshape(1) if x.ndim==0 else x; h=hashlib.sha256(); h.update(str(t.dtype).encode()); h.update(repr(tuple(t.shape)).encode()); h.update(x.view(torch.uint8).numpy().tobytes()); return h.hexdigest()

def test_frozen_hash_matches_reference_for_layouts_and_dtypes():
 from bounded_update import _tensor_hash
 vals=[torch.empty(0,dtype=torch.float32),torch.tensor(3.25),torch.arange(24,dtype=torch.float32).reshape(4,6).t(),torch.arange(64,dtype=torch.float32).reshape(8,8)[::2,1::2],torch.arange(12,dtype=torch.bfloat16).reshape(3,4).t()]
 for t in vals: assert _tensor_hash(t)==_reference_hash(t)

def test_frozen_hash_chunk_transfer_bound(monkeypatch):
 import bounded_update as bu
 seen=[]; original=bu._chunk_bytes
 def wrapped(chunk):
  seen.append(chunk.numel()*chunk.element_size()); return original(chunk)
 monkeypatch.setattr(bu,'_chunk_bytes',wrapped)
 bu._tensor_hash(torch.arange((bu._HASH_CHUNK_BYTES//4)+17,dtype=torch.float32))
 assert seen and max(seen)<=bu._HASH_CHUNK_BYTES

def test_frozen_hash_rejects_unsupported_layout():
 from bounded_update import _tensor_hash
 class Fake:
  layout='sparse'
 with pytest.raises(ValueError,match='unsupported buffer layout'): _tensor_hash(Fake())

def test_buffer_mutation_wins_over_nonfinite_receipt():
 m,o,r=setup(); before=state(m,o)
 def bad(*a,**k):
  z=update(*a,**k); m.sentinel.add_(1); next(m.parameters()).data.fill_(float('nan')); return z
 with pytest.raises(RuntimeError,match='frozen buffer'): bounded_update(m,o,r,'rlvr',bounds=Bounds(100,100,100,1),clip_norm=None,update_fn=bad)
 assert eq(before[:4],state(m,o)[:4]); assert not torch.equal(m.sentinel,before[4])
