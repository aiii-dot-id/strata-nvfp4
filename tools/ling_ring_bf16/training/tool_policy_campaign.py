"""Fresh tool-trajectory RLVR teacher campaign; research only.
AI Identity Incorported <james@aiii.id>.
Self-OPD is not scheduled by this runner. A qualified teacher and fresh,
separately frozen tasks are prerequisites for that later stage.
"""
from __future__ import annotations
import argparse,fcntl,gc,json,os,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
import native,terminal,qualify,boundedpolicy
from capability_runner import release
from capability_update import advantages
from tool_admission import load_inputs,EXPERIMENT
from tool_workflow import sample_tool_task,selection_gate
from tool_training import nominate,replay_qualified_update
from tool_teacher_gate import teacher_gate

CONFIG={"seed":822210,"rounds":4,"samples_per_prompt":4,"calibration_samples":2,
        "max_episode_seconds":600,"max_job_seconds":43200,"rank":8,"alpha":16,
        "answer_tokens":96,"thinking_tokens":128,"max_calls":4}
def write(p,x):native.dump(p,x)
def task_schedule(tasks):
    by={}
    for t in tasks:by.setdefault(t["split"],[]).append(t)
    for group in by.values():group.sort(key=lambda t:t["id"])
    train=by.get("training",[])
    if len(train)<14 or len(by.get("teacher_selection",[]))!=8 or len(by.get("held_out",[]))!=24:raise ValueError("tool suite coverage")
    return {"calibration":train[2:6],"rounds":[train[6+2*i:8+2*i] for i in range(4)],
            "selection":by["teacher_selection"],"held_out":by["held_out"]}
def require_admission(directory,model,repo):
    p=Path(directory);r=json.loads((p/"RESULT.json").read_text());ret=json.loads(p.with_suffix(".RETURN.json").read_text())
    if (r.get("status")!="qualified_native_tool_path" or r.get("model")!=model or
        r.get("zero_adapter_replay_exact") is not True or r.get("terminal_parity",{}).get("passed") is not True or
        ret.get("error") is not None or ret.get("checkpoint_exact") is not True or ret.get("foreign_gpu_preserved") is not True):
        raise ValueError("native tool admission required")
    admitted=json.loads((p/"PLAN.json").read_text())
    for name,pin in admitted["source_closure"].items():
        if native.sha(repo/name)!=pin["sha256"]:raise ValueError("admitted implementation changed")
    return {"result":native.sha(p/"RESULT.json"),"return":native.sha(p.with_suffix(".RETURN.json")),
            "plan":native.sha(p/"PLAN.json")}
def require_cuda_gate(root):
    p=Path(root);r=json.loads((p/"RESULT.json").read_text());s=json.loads((p/"SUPERVISOR-RETURN.json").read_text())
    if (r.get("status")!="passed" or s.get("status")!="finished" or s.get("exit_code")!=0 or
        s.get("error") is not None or s.get("timed_out") is not False or s.get("foreign_preserved") is not True):
        raise ValueError("bounded GPU update safeguards not qualified")
    return {"result":native.sha(p/"RESULT.json"),"supervisor":native.sha(p/"SUPERVISOR-RETURN.json")}
def compact(r,plan_sha):
    out={k:v for k,v in r.items() if k not in ("features","logits","path")}
    out["path"]=str(r["path"]);out["plan_sha256"]=plan_sha
    return out
def run(model,suite_root,admission,cuda_gate,output,bridge,bridge_sha256,policy):
    out=Path(output);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    root=Path(suite_root);public,oracle,manifests,pins=load_inputs(root);schedule=task_schedule(public)
    state=native._git_state(native.ROOT)
    if state["dirty"]:raise ValueError("source must be frozen before native campaign")
    closure=native._source_closure(native.ROOT)
    prerequisites={"native":require_admission(admission,model,native.ROOT),"safeguards":require_cuda_gate(cuda_gate)}
    policy=boundedpolicy.parse_policy(policy)
    c=native.MODELS[model];manifest=native.sha(c["manifest"]);exe=EXPERIMENT/f"build/{model}-torch/strata_{model}_bf16"
    tok=AutoTokenizer.from_pretrained(c["checkpoint"],local_files_only=True)
    plan={"schema":"strata-tool-rlvr-teacher-v1","model":model,"source_commit":state["commit"],"source_closure":closure,
          "prerequisites":prerequisites,"suite_pins":pins,"bridge_sha256":bridge_sha256,"policy":policy,"config":CONFIG,
          "schedule":{k:([x["id"] for x in v] if k!="rounds" else [[x["id"] for x in group] for group in v]) for k,v in schedule.items()},
          "checkpoint_manifest":manifest,"native_executable_sha256":native.sha(exe),"heldout_sealed_until_teacher_freeze":True,
          "self_opd_run":False,"no_quantization":True,"promoted":False,
          "scope":"one final-layer routed expert; on-policy RLOO actual tool actions; no historical-action SFT",
          "limit":"held-out tasks share one source component; result is a limited pilot, not broad generalization"}
    write(out/"PLAN.json",plan);plan_sha=native.sha(out/"PLAN.json")
    started=time.monotonic();trainer=None;opt=None;steps=0;current=None;zero=None
    def release_model():release(trainer)
    def sample(task,folder,adapter,seed):
        remaining=CONFIG["max_job_seconds"]-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError("tool teacher campaign deadline")
        for name,pin in pins.items():
            if native.sha(root/name)!=pin:raise ValueError("suite changed")
        r=sample_tool_task(model,tok,task,root/task["fixture_relative_path"],manifests[task["id"]]["files"],oracle[task["id"]]["expected"],
            binary=bridge,binary_sha256=bridge_sha256,output=out/folder,executable=exe,source_closure=closure,
            adapter=adapter,seed=seed,release=release_model,max_episode_seconds=min(CONFIG["max_episode_seconds"],remaining))
        print(json.dumps({"stage":str(folder),"task":task["id"],"success":r["success"],"complete":r["complete"],
                          "attempts":r["tool_attempts"],"wall_seconds":r["wall_seconds"]}),flush=True)
        return r
    try:
        calibration=[]
        for i,t in enumerate(schedule["calibration"]):
            for j in range(CONFIG["calibration_samples"]):
                calibration.append(sample(t,f"calibration/{t['id']}/{j}",None,CONFIG["seed"]+i*10+j))
        if not any(advantages(calibration)):
            result={"status":"no_within_task_reward_contrast","model":model,"training_updates":0,"teacher_qualified":False,
                    "self_opd_run":False,"promoted":False,"no_quantization":True}
            write(out/"RESULT.json",result);return result
        expert,nomination=nominate(calibration,c["hidden"],128 if model=="ling" else 256)
        write(out/"NOMINATION.json",nomination);calibration.clear();gc.collect()
        torch.manual_seed(CONFIG["seed"]);torch.cuda.manual_seed_all(CONFIG["seed"])
        trainer=terminal.build_terminal(c["checkpoint"],model,expert,rank=CONFIG["rank"],alpha=CONFIG["alpha"],device="cpu")
        opt=torch.optim.Adam(trainer.parameters(),lr=policy["learning_rates"]["rlvr"])
        zero=out/"original-zero.adapter";terminal.save_adapter(trainer,zero,c["checkpoint"],checkpoint_sha256=manifest);current=zero
        decisions=[]
        for rnd,group in enumerate(schedule["rounds"],1):
            records=[]
            for i,t in enumerate(group):
                for j in range(CONFIG["samples_per_prompt"]):
                    records.append(sample(t,f"training/round-{rnd}/{t['id']}/{j}",current,CONFIG["seed"]+10000*rnd+100*i+j))
            if not any(advantages(records)):
                write(out/f"ROUND-{rnd}.json",{"accepted":False,"reason":"uniform_reward","steps":steps});records.clear();gc.collect();continue
            decision=replay_qualified_update(trainer,opt,records,"rlvr",policy=policy,output=out/f"update-{rnd}",
                round_no=rnd,steps_before=steps,current_adapter=current,checkpoint_sha256=manifest,campaign_plan_sha256=plan_sha)
            steps=decision["steps"];current=decision["adapter"]
            decisions.append({"round":rnd,"accepted":decision["accepted"],"steps":steps,"adapter_sha256":native.sha(current)})
            records.clear();gc.collect()
        # Candidate identity freezes before either evaluation split is sampled.
        freeze={"adapter_sha256":native.sha(current),"original_zero_sha256":native.sha(zero),"plan_sha256":plan_sha,
                "steps":steps,"checkpoint_manifest":manifest,"no_more_training":True}
        write(out/"TEACHER-FREEZE.json",freeze)
        evaluated={"teacher_selection":{"baseline":[],"rlvr":[]},"held_out":{"baseline":[],"rlvr":[]}}
        for split,key in (("teacher_selection","selection"),("held_out","held_out")):
            for i,t in enumerate(schedule[key]):
                order=("baseline","rlvr") if i%2==0 else ("rlvr","baseline")
                for arm in order:
                    adapter=None if arm=="baseline" else current
                    r=sample(t,f"evaluation/{split}/{t['id']}/{arm}",adapter,CONFIG["seed"]+100000+(0 if split=="teacher_selection" else 10000)+i)
                    evaluated[split][arm].append(compact(r,plan_sha));del r;gc.collect()
            write(out/f"EVALUATION-{split}.json",evaluated[split])
        d=evaluated["teacher_selection"];h=evaluated["held_out"]
        verdict=teacher_gate(d["baseline"],d["rlvr"],h["baseline"],h["rlvr"],
            declared_selection_ids=[t["id"] for t in schedule["selection"]],
            declared_task_ids=[t["id"] for t in schedule["held_out"]],
            teacher_adapter_sha256=freeze["adapter_sha256"],plan_sha256=plan_sha)
        write(out/"TEACHER-GATE.json",verdict)
        # A result can qualify a pilot teacher without authorizing promotion or
        # reuse of its consumed evaluation tasks for later self-distillation.
        result={"status":verdict["status"],"model":model,"training_updates":steps,"decisions":decisions,
                "teacher_qualified":verdict["status"]=="limited-pilot-qualified","teacher_adapter":str(current),
                "teacher_adapter_sha256":native.sha(current),"no_quantization":True,"promoted":False,
                "self_opd_run":False,"next_gate":"fresh separately frozen Self-OPD training and held-out tasks only after a qualified teacher",
                "wall_seconds":time.monotonic()-started}
        write(out/"RESULT.json",result);return result
    finally:release(trainer)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",choices=native.MODELS,required=True)
    for name in ("suite-root","admission","cuda-gate","output","bridge","policy"):ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--bridge-sha256",required=True);a=ap.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with open("/work/AI/experiments/strata-dev8-gpu.lock","a+") as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8";torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        before=qualify.bind_checkpoint(a.model);foreign=qualify.gpu_processes();error=None;result=None
        write(a.output.with_suffix(".binding-before.json"),before)
        try:result=run(a.model,a.suite_root,a.admission,a.cuda_gate,a.output,a.bridge,a.bridge_sha256,boundedpolicy.parse_policy(a.policy))
        except BaseException as exc:error=repr(exc);raise
        finally:
            release();after=qualify.bind_checkpoint(a.model);write(a.output.with_suffix(".binding-after.json"),after)
            try:qualify.check_foreign(foreign);preserved=True
            except RuntimeError:preserved=False
            write(a.output.with_suffix(".RETURN.json"),{"error":error,"checkpoint_exact":before==after,
                "foreign_gpu_preserved":preserved,"services_touched":False,"mac_touched":False,
                "status":result["status"] if result else "failed","promoted":False})
            if error is None and (before!=after or not preserved):raise RuntimeError("research return failed")
if __name__=="__main__":main()
