"""Native tool-path admission on dev8, before any tool-strategy updates.
AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations
import argparse, fcntl, gc, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
import native,terminal,qualify
from capability_runner import release
from tool_workflow import sample_tool_task
from tool_environment import verify_fixture
from capability_update import advantages
EXPERIMENT=Path('/work/AI/experiments/strata-on-policy-20261021-r1')
AUTHOR="AI Identity Incorported <james@aiii.id>"

def load_inputs(suite_root):
    root=Path(suite_root)
    names=["SUITE.public.json","ORACLES.private.json","FIXTURE-MANIFESTS.public.json","ROOT-INDEPENDENT-REVIEW.json","CANONICAL-TOOL-CPU-CONTROLS.json"]
    pins={n:native.sha(root/n) for n in names}
    public=json.loads((root/names[0]).read_text())["tasks"]
    oracle={x["id"]:x for x in json.loads((root/names[1]).read_text())["tasks"]}
    manifests=json.loads((root/names[2]).read_text())["manifests"]
    for n in names[3:]:
        if json.loads((root/n).read_text()).get("passed") is not True:raise ValueError("fixture CPU controls failed")
    if len({x["id"] for x in public})!=len(public) or set(oracle)!={x["id"] for x in public}:raise ValueError("task IDs")
    for t in public:
        verify_fixture(root/t["fixture_relative_path"],manifests[t["id"]]["files"])
    return public,oracle,manifests,pins

def run(model,suite_root,output,binary,binary_sha):
    root=Path(suite_root);out=Path(output);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    public,oracle,manifests,pins=load_inputs(root)
    selected=sorted([x for x in public if x["split"]=="training"],key=lambda x:x["id"])[:2]
    if len(selected)!=2:raise ValueError("two training tasks required")
    state=native._git_state(native.ROOT)
    if state["dirty"]:raise ValueError("source must be frozen")
    closure=native._source_closure(native.ROOT)
    checkpoint=native.MODELS[model]["checkpoint"];manifest=native.sha(native.MODELS[model]["manifest"])
    executable=EXPERIMENT/f"build/{model}-torch/strata_{model}_bf16"
    if native.sha(binary)!=binary_sha:raise ValueError("tool binary pin")
    tok=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    plan={"schema":"strata-tool-native-admission-v1","author":AUTHOR,"model":model,"tasks":[t["id"] for t in selected],
          "samples_per_task":2,"training_updates":0,"heldout_used":False,"no_quantization":True,"promoted":False,
          "source_commit":state["commit"],"source_closure":closure,"suite_pins":pins,"bridge_sha256":binary_sha,
          "native_executable":str(executable),"native_sha256":native.sha(executable),"checkpoint_manifest":manifest,
          "seed":812210,"episode_seconds":600,"answer_tokens":96,"thinking_tokens":128,"max_calls":4,
          "scope":"actual tool calls, generation/action alignment, terminal readout parity, zero-adapter replay"}
    native.dump(out/"PLAN.json",plan)
    trainer=None
    def release_model():release(trainer)
    records=[]
    try:
        for i,t in enumerate(selected):
            for j in range(2):
                for name,pin in pins.items():
                    if native.sha(root/name)!=pin:raise ValueError("fixture suite changed")
                r=sample_tool_task(model,tok,t,root/t["fixture_relative_path"],manifests[t["id"]]["files"],oracle[t["id"]]["expected"],
                    binary=binary,binary_sha256=binary_sha,output=out/f"baseline/{t['id']}/{j}",executable=executable,
                    source_closure=closure,seed=plan["seed"]+i*100+j,release=release_model,max_episode_seconds=600)
                records.append(r)
                print(json.dumps({"stage":"tool_episode","model":model,"task":t["id"],"sample":j,"success":r["success"],
                    "complete":r["complete"],"tool_calls":r["tool_calls"],"attempts":r["tool_attempts"],"seconds":r["wall_seconds"]}),flush=True)
        if not all(len(r["ids"])==len(r["features"])==len(r["logits"])==len(r["sampling_logprobs"]) for r in records):raise ValueError("training row alignment")
        if sum(r["tool_calls"] for r in records)==0:raise ValueError("model made no actual tool calls")
        if sum(r["success"] for r in records)==0:raise ValueError("no verified tool completion")
        h=native.MODELS[model]["hidden"];expert_count=128 if model=="ling" else 256
        routes=torch.cat([r["features"][:,3*h:3*h+8].long().flatten() for r in records])
        expert=int(torch.bincount(routes,minlength=expert_count).argmax())
        torch.manual_seed(plan["seed"]);torch.cuda.manual_seed_all(plan["seed"])
        trainer=terminal.build_terminal(checkpoint,model,expert,rank=8,alpha=16,device="cpu")
        # CPU storage, qualified CUDA math, only three observed action rows.
        chosen=records[0];indices=sorted(set([0,len(chosen["ids"])//2,len(chosen["ids"])-1]))
        trainer.cuda()
        with torch.no_grad():expected=np.concatenate([trainer(chosen["features"][i:i+1].cuda()).cpu().numpy() for i in indices])
        parity=qualify.compare_logits(expected,np.asarray(chosen["logits"])[indices]);release(trainer)
        native.dump(out/"TERMINAL-PARITY.json",dict(parity,indices=indices,expert=expert))
        if not parity["passed"]:raise ValueError("native tool action terminal parity failed")
        zero=out/"zero.adapter";terminal.save_adapter(trainer,zero,checkpoint,checkpoint_sha256=manifest)
        t=selected[0]
        replay=sample_tool_task(model,tok,t,root/t["fixture_relative_path"],manifests[t["id"]]["files"],oracle[t["id"]]["expected"],
            binary=binary,binary_sha256=binary_sha,output=out/"zero-replay",executable=executable,source_closure=closure,
            adapter=zero,seed=plan["seed"],release=release_model,max_episode_seconds=600)
        if replay["ids"]!=chosen["ids"] or replay["transcript_ids"]!=chosen["transcript_ids"] or not np.array_equal(replay["logits"],chosen["logits"]):
            raise ValueError("zero-adapter tool replay differs")
        reward_variance=any(abs(x)>0 for x in advantages(records))
        result={"status":"qualified_native_tool_path","model":model,"episodes":len(records),
            "verified_completions":sum(r["success"] for r in records),"incomplete":sum(not r["complete"] for r in records),
            "actual_tool_calls":sum(r["tool_calls"] for r in records),"tool_attempts":sum(r["tool_attempts"] for r in records),
            "within_task_reward_contrast":reward_variance,"zero_adapter_replay_exact":True,"terminal_parity":parity,
            "training_updates":0,"rlvr_teacher_qualified":False,"self_opd_run":False,"heldout_used":False,
            "checkpoint_unchanged_by_design":True,"no_quantization":True,"promoted":False}
        native.dump(out/"RESULT.json",result)
        return result
    finally:release(trainer)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",choices=native.MODELS,required=True)
    ap.add_argument("--suite-root",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--bridge",type=Path,required=True);ap.add_argument("--bridge-sha256",required=True);a=ap.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with open("/work/AI/experiments/strata-dev8-gpu.lock","a+") as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8";torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        before=qualify.bind_checkpoint(a.model);foreign=qualify.gpu_processes()
        native.dump(a.output.with_suffix(".binding-before.json"),before);error=None;result=None
        try:result=run(a.model,a.suite_root,a.output,a.bridge,a.bridge_sha256)
        except BaseException as exc:error=repr(exc);raise
        finally:
            release();after=qualify.bind_checkpoint(a.model);native.dump(a.output.with_suffix(".binding-after.json"),after)
            try:qualify.check_foreign(foreign);preserved=True
            except RuntimeError:preserved=False
            native.dump(a.output.with_suffix(".RETURN.json"),{"error":error,"checkpoint_exact":before==after,
                "foreign_gpu_preserved":preserved,"before_gpu":foreign,"after_gpu":qualify.gpu_processes(),
                "services_touched":False,"mac_touched":False,"training_updates":0,"promoted":False,
                "status":result["status"] if result else "failed"})
            if error is None and (before!=after or not preserved):raise RuntimeError("runtime restoration failed")
    return 0
if __name__=="__main__":raise SystemExit(main())
