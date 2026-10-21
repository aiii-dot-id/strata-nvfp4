"""Bounded exact-objective updates for native terminal adapters.
AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations
import time, math
import numpy as np
import torch
import qualify
from on_policy_losses import self_opd

def advantages(records):
    groups={}
    for i,r in enumerate(records): groups.setdefault(r['group'],[]).append(i)
    out=[0.]*len(records)
    for inds in groups.values():
        if len(inds)<2: raise ValueError('each prompt requires at least two trajectories')
        vals=[float(records[i]['grade']['reward']) for i in inds]
        if not all(math.isfinite(x) for x in vals):raise ValueError('nonfinite rewards')
        total=sum(vals)
        for i,v in zip(inds,vals):out[i]=v-(total-v)/(len(inds)-1)
    return out

def update(trainer,opt,records,arm,*,temperature=1.,chunk_rows=8,clip_norm=1.0):
    if arm not in ('rlvr','self_opd') or not records:raise ValueError('objective/batch')
    if type(chunk_rows)is not int or chunk_rows<1:raise ValueError('chunk_rows')
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('temperature')
    start=time.monotonic();device=next(trainer.parameters()).device
    params=list(trainer.parameters());opt.zero_grad(set_to_none=True)
    adv=advantages(records) if arm=='rlvr' else None
    if arm=='rlvr' and not any(x!=0 for x in adv):raise ValueError('uniform reward batch')
    total_rows=sum(len(r['ids']) for r in records);total_loss=0.;max_error=0.;max_lp_error=0.;families={}
    before=qualify.tensors_digest(trainer.named_parameters())
    for i,r in enumerate(records):
        n=len(r['ids'])
        if n<1 or len(r['features'])!=n or len(r['logits'])!=n or len(r['sampling_logprobs'])!=n:raise ValueError('record row alignment')
        fam=r['family']; fg=families.setdefault(fam,[torch.zeros_like(p) for p in params])
        for startrow in range(0,n,chunk_rows):
            end=min(startrow+chunk_rows,n)
            features=r['features'][startrow:end].to(device)
            logits=trainer(features)
            native=np.asarray(r['logits'][startrow:end])
            parity=qualify.compare_logits(logits.detach().cpu().numpy(),native)
            if not parity['passed']:raise ValueError('trainer/current-native parity failed')
            max_error=max(max_error,parity['max_abs'])
            ids=torch.tensor(r['ids'][startrow:end],dtype=torch.long,device=device)
            logprobs=(logits.detach().double()/temperature).log_softmax(-1).gather(1,ids[:,None]).squeeze(1).cpu().numpy()
            lp_error=float(np.max(np.abs(logprobs-np.asarray(r['sampling_logprobs'][startrow:end]))))
            max_lp_error=max(max_lp_error,lp_error)
            if lp_error>0.002:raise ValueError('sampled behavior logprob mismatch')
            if arm=='rlvr':
                lp=(logits/temperature).log_softmax(-1).gather(1,ids[:,None]).sum()
                loss=-adv[i]*lp/len(records)
            else:
                teacher=torch.as_tensor(np.array(r['teacher'][startrow:end],copy=True),dtype=logits.dtype,device=device)
                loss,_=self_opd(logits,teacher,temperature);loss=loss/total_rows
            if not torch.isfinite(loss):raise ValueError('nonfinite objective')
            grads=torch.autograd.grad(loss,params,allow_unused=False)
            for j,(p,g) in enumerate(zip(params,grads)):
                if not torch.isfinite(g).all():raise ValueError('nonfinite gradient')
                gd=g.detach()
                p.grad=gd.clone() if p.grad is None else p.grad+gd
                fg[j].add_(gd)
            total_loss+=float(loss.detach())
    famnorm={k:float(sum(g.double().square().sum() for g in vs).sqrt()) for k,vs in families.items()}
    norm=float(sum(p.grad.double().square().sum() for p in params).sqrt())
    if not norm>0 or not math.isfinite(norm):raise ValueError('zero or nonfinite gradient')
    grads_before_clip=qualify.tensors_digest((n,p.grad) for n,p in trainer.named_parameters())
    if clip_norm is not None:torch.nn.utils.clip_grad_norm_(params,clip_norm,error_if_nonfinite=True)
    opt.step()
    after=qualify.tensors_digest(trainer.named_parameters())
    if before==after:raise ValueError('no parameter change')
    return {'objective':arm,'loss':total_loss,'response_tokens':total_rows,'trajectories':len(records),
            'gradient_norm':norm,'family_gradient_norm':famnorm,'gradient':grads_before_clip,
            'parameters':after,'sampled_logprob_max_abs':max_lp_error,'native_logits_max_abs':max_error,
            'advantages':adv,'update_seconds':time.monotonic()-start,'chunk_rows':chunk_rows,
            'clip_norm':clip_norm,'action_tokens_only':True}
