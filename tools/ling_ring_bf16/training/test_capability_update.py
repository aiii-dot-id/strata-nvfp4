import copy
import math
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from capability_update import update
from on_policy_losses import rlvr, self_opd

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.proj=torch.nn.Linear(4,5,bias=True)
    def forward(self,x): return self.proj(x)

def make_records(model, rewards=(0.,0.,1.,0.), with_teacher=False):
    torch.manual_seed(22); features=[torch.randn(2,4) for _ in range(4)]
    records=[]
    with torch.no_grad():
        for i,x in enumerate(features):
            logits=model(x).detach().numpy().astype('<f4'); ids=[int(logits[0].argmax()),int(logits[1].argmax())]
            lp=torch.log_softmax(torch.from_numpy(logits).double(),-1).gather(1,torch.tensor(ids)[:,None]).squeeze().numpy().tolist()
            r={'group':0 if i<2 else 1,'family':'zero-family' if i<2 else 'signal-family','ids':ids,'features':x,'logits':logits,'sampling_logprobs':lp,'grade':{'reward':rewards[i]}}
            if with_teacher: r['teacher']=(logits+0.1*np.eye(5,dtype='<f4')[ids]).astype('<f4')
            records.append(r)
    return records

def clone_model(m):
    n=Toy(); n.load_state_dict(copy.deepcopy(m.state_dict())); return n

def test_rlvr_chunked_matches_full_objective_gradient_and_adam():
    torch.manual_seed(5); base=Toy(); chunked=clone_model(base)
    rec=make_records(base)
    flat=torch.cat([base(r['features']) for r in rec]); ids=torch.tensor(sum((r['ids'] for r in rec),[]),dtype=torch.long)
    lengths=[2]*4; rewards=[r['grade']['reward'] for r in rec]; groups=[r['group'] for r in rec]
    full_loss,_=rlvr(flat,ids,lengths,rewards,groups,1.0); opt_full=torch.optim.Adam(base.parameters(),lr=.01); opt_full.zero_grad(); full_loss.backward(); opt_full.step()
    opt_chunk=torch.optim.Adam(chunked.parameters(),lr=.01); got=update(chunked,opt_chunk,rec,'rlvr',chunk_rows=1,clip_norm=None)
    for a,b in zip(base.parameters(),chunked.parameters()): assert torch.allclose(a,b,atol=2e-6,rtol=2e-6)
    assert math.isfinite(got['loss']) and got['action_tokens_only']
    assert got['family_gradient_norm']['zero-family']==0.0
    assert got['family_gradient_norm']['signal-family']>0

def test_self_opd_chunked_matches_full_objective_gradient():
    torch.manual_seed(7); base=Toy(); chunked=clone_model(base); rec=make_records(base,with_teacher=True)
    logits=torch.cat([base(r['features']) for r in rec]); teacher=torch.from_numpy(np.concatenate([r['teacher'] for r in rec]))
    full_loss,_=self_opd(logits,teacher,1.0); opt=torch.optim.SGD(base.parameters(),lr=.01); opt.zero_grad(); (full_loss/len(logits)).backward(); opt.step()
    opt2=torch.optim.SGD(chunked.parameters(),lr=.01); got=update(chunked,opt2,rec,'self_opd',chunk_rows=1,clip_norm=None)
    for a,b in zip(base.parameters(),chunked.parameters()): assert torch.allclose(a,b,atol=2e-6,rtol=2e-6)
    assert got['family_gradient_norm']['zero-family']>=0 and got['gradient_norm']>0

def test_optimizer_snapshot_restore_matches_continuation():
    torch.manual_seed(8); first=Toy(); rec=make_records(first); opt=torch.optim.Adam(first.parameters(),lr=.003)
    update(first,opt,rec,'rlvr',chunk_rows=2,clip_norm=None)
    snap_model=copy.deepcopy(first.state_dict()); snap_opt=copy.deepcopy(opt.state_dict())
    resumed=Toy(); resumed.load_state_dict(snap_model); opt2=torch.optim.Adam(resumed.parameters(),lr=.003); opt2.load_state_dict(snap_opt)
    # A fresh native-bound batch at the restored parameters.
    rec2=make_records(first,rewards=(0.,1.,0.,1.))
    update(first,opt,rec2,'rlvr',chunk_rows=2,clip_norm=None); update(resumed,opt2,rec2,'rlvr',chunk_rows=2,clip_norm=None)
    for a,b in zip(first.parameters(),resumed.parameters()): assert torch.equal(a,b)
    def same(a,b):
        if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a,b)
        if isinstance(a,dict): return isinstance(b,dict) and set(a)==set(b) and all(same(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    assert same(opt.state_dict(),opt2.state_dict())

def test_bad_batches_and_uniform_rewards_fail_before_update():
    torch.manual_seed(9); m=Toy(); opt=torch.optim.Adam(m.parameters(),lr=.01); rec=make_records(m,rewards=(0.,0.,0.,0.))
    with pytest.raises(ValueError,match='uniform reward'): update(m,opt,rec,'rlvr')
    bad=copy.deepcopy(rec); bad[0]['logits'][0,0]=np.nan
    with pytest.raises(ValueError): update(m,opt,bad,'rlvr')
    missing=copy.deepcopy(make_records(m,rewards=(0.,1.,0.,1.))); missing[0]['sampling_logprobs']=[]
    with pytest.raises(ValueError,match='record row alignment'): update(m,opt,missing,'rlvr')

def test_nonfinite_teacher_fails_self_opd():
    torch.manual_seed(10); m=Toy(); opt=torch.optim.Adam(m.parameters(),lr=.01); rec=make_records(m,with_teacher=True); rec[0]['teacher'][0,0]=np.inf
    with pytest.raises(ValueError): update(m,opt,rec,'self_opd')


def test_completion_action_positions_match_native_response_rows():
    from completion import CompletionConfig, run_completion
    enc=lambda s:list(s.encode()); dec=lambda ids:bytes(ids).decode()
    rows={"completion":{"ids":[7,8],"sampled_logprobs":[-.1,-.2],"text":"ab","eos":False,"finish_reason":"length"}}
    r=run_completion([10,11,12],"A",lambda p,**kw: rows[kw["stage"]],enc,dec,CompletionConfig(model="ling",max_new_tokens=2,context_capacity=8))
    # Native response logits are indexed at prompt_len-1+i for action i.
    assert tuple(p-1 for p in r.action_positions)==(2,3)
