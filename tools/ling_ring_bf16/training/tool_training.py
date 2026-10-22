"""Bounded RLVR and Self-OPD updates for actual tool-action trajectories.
AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
import terminal, qualify, native
from capability_update import advantages
from capability_runner import perform_update, release
import boundedpolicy

def nominate(records,hidden,expert_count):
    a=advantages(records)
    counts=torch.zeros(expert_count,dtype=torch.float64)
    if not any(abs(x)>0 for x in a):raise ValueError("no reward contrast within sampled task groups")
    for r,v in zip(records,a):
        routes=r["features"][:,3*hidden:3*hidden+8]
        if not torch.equal(routes,routes.round()) or (routes<0).any() or (routes>=expert_count).any():raise ValueError("invalid captured routes")
        counts+=torch.bincount(routes.long().flatten(),minlength=expert_count).double()*abs(v)
    expert=int(counts.argmax())
    if counts[expert]<=0:raise ValueError("no informative route")
    return expert,{"expert":expert,"route_weight":float(counts[expert]),"rule":"advantage-weighted sampled action exposure"}

def records_digest(records):
    return hashlib.sha256(json.dumps([{"receipt":native.sha(r["path"]/"ROLLOUT.json"),"stages":r["native_stages"]} for r in records],sort_keys=True,separators=(",",":")).encode()).hexdigest()

def replay_qualified_update(trainer,opt,records,arm,*,policy,output,round_no,steps_before,current_adapter,
                            checkpoint_sha256,campaign_plan_sha256,device="cuda"):
    """Each proposed update is exactly restorable; rejected proposals never export."""
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    policy=boundedpolicy.parse_policy(policy)
    if arm not in ("rlvr","self_opd"):raise ValueError("training objective")
    if type(round_no)is not int or round_no<1 or type(steps_before)is not int or steps_before<0:raise ValueError("training counters")
    if not opt.param_groups or any(float(g["lr"])!=policy["learning_rates"][arm] for g in opt.param_groups):raise ValueError("optimizer/policy learning rate mismatch")
    parent_sha=native.sha(current_adapter)
    if any(r.get("policy_adapter_sha256")!=parent_sha for r in records):raise ValueError("rollout/parent adapter binding")
    batch=records_digest(records)
    config={"policy_sha256":boundedpolicy.policy_sha256(policy),"campaign_plan":campaign_plan_sha256,
            "objective":arm,"checkpoint":checkpoint_sha256,"expert":trainer.expert,"learning_rate":opt.param_groups[0]["lr"]}
    trainer.to(device)
    before_parameters=qualify.tensors_digest(trainer.named_parameters())
    frozen=qualify.tensors_digest(trainer.named_buffers())
    snapshot=out/"before"
    terminal.save_training_state(trainer,opt,snapshot,step=steps_before,iteration_cursor=round_no,data_hash=batch,config_binding=config)
    try:
        decision=perform_update(trainer,opt,records,arm,policy=policy,chunk_rows=1)
        metric=decision["update"]
        if not decision["accepted"]:
            if before_parameters!=qualify.tensors_digest(trainer.named_parameters()):raise ValueError("rejected parameter change")
            if frozen!=qualify.tensors_digest(trainer.named_buffers()):raise ValueError("frozen parameter change")
            native.dump(out/"DECISION.json",boundedpolicy.receipt(policy,arm=arm,round_no=round_no,steps_before=steps_before,
                steps_after=steps_before,current_adapter_sha256=native.sha(current_adapter),batch_sha256=batch,decision=decision))
            return {"accepted":False,"adapter":current_adapter,"decision":decision,"steps":steps_before}
        end_parameters={k:v.detach().clone() for k,v in trainer.named_parameters()}
        end_opt=copy.deepcopy(opt.state_dict());end_rng=torch.get_rng_state().clone()
        cuda_rng=torch.cuda.get_rng_state().clone() if str(device).startswith("cuda") else None
        terminal.restore_training_state(trainer,opt,snapshot,data_hash=batch,config_binding=config)
        repeated=perform_update(trainer,opt,records,arm,policy=policy,chunk_rows=1)
        if repeated["accepted"] is not True:raise ValueError("accepted update rejected on replay")
        qualify.tree_exact(dict(trainer.named_parameters()),end_parameters)
        qualify.tree_exact(opt.state_dict(),end_opt)
        qualify.tree_exact(torch.get_rng_state(),end_rng)
        if cuda_rng is not None:qualify.tree_exact(torch.cuda.get_rng_state(),cuda_rng)
        for key in ("loss","gradient","parameters"):qualify.tree_exact(metric[key],repeated["update"][key])
        qualify.tree_exact(decision["metrics"],repeated["metrics"])
        if frozen!=qualify.tensors_digest(trainer.named_buffers()):raise ValueError("frozen buffer mutation")
        candidate=out/"candidate.adapter";cp=native.MODELS[trainer.model]["checkpoint"]
        terminal.save_adapter(trainer,candidate,cp,checkpoint_sha256=checkpoint_sha256)
        expected=qualify.tensors_digest(trainer.named_parameters())
        with torch.no_grad():
            for p in trainer.parameters():p.zero_()
        terminal.load_adapter(trainer,candidate,checkpoint_sha256=checkpoint_sha256)
        if qualify.tensors_digest(trainer.named_parameters())!=expected:raise ValueError("export/reload changed parameters")
        native.dump(out/"DECISION.json",boundedpolicy.receipt(policy,arm=arm,round_no=round_no,steps_before=steps_before,
            steps_after=steps_before+1,current_adapter_sha256=native.sha(current_adapter),batch_sha256=batch,decision=decision))
        native.dump(out/"QUALIFIED.json",{"replayed_update_exact":True,"optimizer_rng_exact":True,"export_reload_exact":True,
            "candidate_sha256":native.sha(candidate),"batch_sha256":batch,"no_quantization":True,"promoted":False})
        return {"accepted":True,"adapter":candidate,"decision":decision,"steps":steps_before+1}
    finally:
        release(trainer)
