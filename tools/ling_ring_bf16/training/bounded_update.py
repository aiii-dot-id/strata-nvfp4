"""Optional bounded trial around the existing exact RLVR/Self-OPD update.
AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations
import copy, hashlib, math
from dataclasses import dataclass, asdict
import torch
import capability_update

@dataclass(frozen=True)
class Bounds:
    max_mean_kl: float
    max_max_kl: float
    max_selected_log_ratio: float
    chunk_rows: int
    def __post_init__(self):
        vals=(self.max_mean_kl,self.max_max_kl,self.max_selected_log_ratio)
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(float(x)) for x in vals): raise ValueError('bounds must be finite numbers')
        if self.max_mean_kl<0 or self.max_max_kl<0 or self.max_selected_log_ratio<0: raise ValueError('bounds must be nonnegative')
        if type(self.chunk_rows) is not int or self.chunk_rows<1: raise ValueError('chunk_rows must be positive')

def _tensor_hash(t):
    x=t.detach().cpu().contiguous(); h=hashlib.sha256(); h.update(str(x.dtype).encode()); h.update(repr(tuple(x.shape)).encode()); h.update(x.numpy().tobytes()); return h.hexdigest()
def _buffer_hashes(model): return {n:_tensor_hash(b) for n,b in model.named_buffers()}
def _finite_state(x):
    if torch.is_tensor(x): return (not x.is_floating_point()) or bool(torch.isfinite(x).all())
    if isinstance(x,dict): return all(_finite_state(v) for v in x.values())
    if isinstance(x,(list,tuple)): return all(_finite_state(v) for v in x)
    return True

def _validate_binding(model,opt):
    named=list(model.named_parameters()); params=[p for _,p in named]
    if not params: raise ValueError('module has no parameters')
    devices={p.device for p in params}
    if len(devices)!=1: raise ValueError('parameters must use one device')
    device=next(iter(devices))
    if device.type not in ('cpu','cuda'): raise ValueError('unsupported parameter device')
    if device.type=='cuda' and device.index is None: raise ValueError('CUDA device must be explicit')
    if any(b.device!=device for b in model.buffers()): raise ValueError('buffers must share parameter device')
    if len({id(p) for _,p in named})!=len(named): raise ValueError('duplicate parameter identity')
    if not all(_finite_state(p) and (p.grad is None or _finite_state(p.grad)) for p in params): raise ValueError('nonfinite parameter or gradient')
    ids=[id(p) for p in params]; group_ids=[id(p) for g in opt.param_groups for p in g['params']]
    if sorted(ids)!=sorted(group_ids) or len(set(group_ids))!=len(group_ids): raise ValueError('optimizer parameter binding mismatch')
    if not _finite_state(opt.state_dict()): raise ValueError('nonfinite optimizer state')
    for state in opt.state.values():
        for value in state.values():
            if torch.is_tensor(value) and value.device not in (device,torch.device('cpu')): raise ValueError('optimizer state device mismatch')
    return named,device

def _snapshot(model,opt,named,device):
    return {'params':[p.detach().clone() for _,p in named], 'grads':[None if p.grad is None else p.grad.detach().clone() for _,p in named], 'optimizer':copy.deepcopy(opt.state_dict()), 'param_names':[n for n,_ in named], 'param_ids':[id(p) for _,p in named], 'buffers':_buffer_hashes(model), 'rng_cpu':torch.get_rng_state().clone(), 'device':device, 'rng_cuda':torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None}

def _restore(model,opt,s):
    named=list(model.named_parameters())
    if [n for n,_ in named]!=s['param_names'] or [id(p) for _,p in named]!=s['param_ids']: raise RuntimeError('parameter structure changed')
    with torch.no_grad():
        for (_,p),x,g in zip(named,s['params'],s['grads']): p.copy_(x); p.grad=None if g is None else g.clone()
    opt.load_state_dict(copy.deepcopy(s['optimizer'])); torch.set_rng_state(s['rng_cpu'])
    if s['rng_cuda'] is not None: torch.cuda.set_rng_state(s['rng_cuda'],device=s['device'])

def _stream_metrics(records,model,temperature,chunk_rows):
    device=next(model.parameters()).device; total=torch.zeros((),dtype=torch.float64,device=device); rows=0; maxkl=0.; maxratio=0.
    with torch.no_grad():
        for r in records:
            f=r['features'] if torch.is_tensor(r['features']) else torch.as_tensor(r['features'])
            old=torch.as_tensor(r['logits']); ids=torch.as_tensor(r['ids'],dtype=torch.long)
            if f.shape[0]!=old.shape[0] or ids.numel()!=f.shape[0]: raise ValueError('record fixed-state alignment')
            for st in range(0,f.shape[0],chunk_rows):
                a=old[st:st+chunk_rows].to(device=device,dtype=torch.float64)/temperature
                b=model(f[st:st+chunk_rows].to(device)).double()/temperature
                if a.shape!=b.shape or not torch.isfinite(b).all() or not torch.isfinite(a).all(): return {'finite':False,'mean_kl':None,'max_kl':None,'max_selected_log_ratio':None,'rows':rows}
                lp=torch.log_softmax(a,-1); lq=torch.log_softmax(b,-1); kl=(lp.exp()*(lp-lq)).sum(-1); ratio=(lq-lp).gather(1,ids[st:st+chunk_rows,None].to(device)).squeeze(1).abs()
                total+=kl.sum(); maxkl=max(maxkl,float(kl.max().detach())); maxratio=max(maxratio,float(ratio.max().detach())); rows+=a.shape[0]
    mean=float((total/rows).detach()) if rows else None; return {'finite':bool(rows and mean is not None and math.isfinite(mean) and math.isfinite(maxkl) and math.isfinite(maxratio)),'mean_kl':mean,'max_kl':maxkl,'max_selected_log_ratio':maxratio,'rows':rows}

def bounded_update(trainer,opt,records,arm,*,bounds,temperature=1.,chunk_rows=None,clip_norm=1.,update_fn=None):
    if not isinstance(bounds,Bounds): raise TypeError('explicit Bounds required')
    if isinstance(temperature,bool) or not math.isfinite(float(temperature)) or temperature<=0: raise ValueError('temperature')
    if chunk_rows is None: chunk_rows=bounds.chunk_rows
    if type(chunk_rows) is not int or chunk_rows<1: raise ValueError('chunk_rows')
    named,device=_validate_binding(trainer,opt); snap=_snapshot(trainer,opt,named,device); update_fn=update_fn or capability_update.update
    try:
        result=update_fn(trainer,opt,records,arm,temperature=temperature,chunk_rows=chunk_rows,clip_norm=clip_norm)
        if not all(_finite_state(p) and (p.grad is None or _finite_state(p.grad)) for _,p in trainer.named_parameters()) or not _finite_state(opt.state_dict()):
            _restore(trainer,opt,snap); return {'accepted':False,'reason':'nonfinite post-update state','objective':arm,'update':result,'metrics':{'finite':False,'mean_kl':None,'max_kl':None,'max_selected_log_ratio':None,'rows':0},'bounds':asdict(bounds),'restored':True}
        if _buffer_hashes(trainer)!=snap['buffers']:
            _restore(trainer,opt,snap); raise RuntimeError('frozen buffer mutation; trial invalid and caller must abort')
        metric_rng_cpu=torch.get_rng_state().clone()
        metric_rng_cuda=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
        try:
            metrics=_stream_metrics(records,trainer,temperature,chunk_rows)
        finally:
            torch.set_rng_state(metric_rng_cpu)
            if metric_rng_cuda is not None: torch.cuda.set_rng_state(metric_rng_cuda,device=device)
        if _buffer_hashes(trainer)!=snap['buffers']:
            _restore(trainer,opt,snap); raise RuntimeError('frozen buffer mutation during metric evaluation; caller must abort')
        reason='accepted'
        if not metrics['finite']: reason='nonfinite KL/log-ratio metrics'
        elif metrics['mean_kl']>bounds.max_mean_kl: reason='mean KL bound exceeded'
        elif metrics['max_kl']>bounds.max_max_kl: reason='max KL bound exceeded'
        elif metrics['max_selected_log_ratio']>bounds.max_selected_log_ratio: reason='selected-token log-ratio bound exceeded'
        if reason!='accepted': _restore(trainer,opt,snap)
        return {'accepted':reason=='accepted','reason':reason,'objective':arm,'update':result,'metrics':metrics,'bounds':asdict(bounds),'restored':reason!='accepted'}
    except BaseException:
        _restore(trainer,opt,snap); raise
