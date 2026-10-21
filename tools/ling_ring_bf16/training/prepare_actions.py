#!/usr/bin/env python3
"""Freeze corrected-action prompts, token targets, and non-training retention probes."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from transformers import AutoTokenizer

TOKENIZER_NAMES={"ling":"Ling-3.0-tiny","ring":"Ring-mini-2.0"}
MODEL_VOCAB=157184
RULES=(
    "Choose exactly one next action from: inspect path=<ABS>, read path=<ABS>, test path=<ABS>, repair path=<ABS>, setcwd path=<ABS>.",
    "Use only facts in the observed state. Paths must be absolute and under /workspace.",
    "If cwd differs from target, setcwd to target before inspecting or changing files.",
    "If a target is missing and its parent is uninspected, inspect the parent before deciding how to repair it.",
    "After validation fails or observed content is stale, reread the target before testing or repairing it.",
    "Return exactly one action line; do not include reasoning or a prefix/suffix."
)
RETENTION=(
    ("Compute 9 + 8. Return only the exact answer.","17"),
    ("Extract the value after key= from this record: key=42. Return only the exact answer.","42"),
    ("Which color is named in this record? color=blue. Return only the exact answer.","blue"),
    ("Extract the quarter from this record: quarter=Q3. Return only the exact answer.","Q3"),
)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def oracle(state):
    if state.get("target_kind") == "directory" and state.get("cwd") != state.get("target"):
        return f"setcwd path={state['target']}"
    if state.get("parent_inspected") is False:
        return f"inspect path={state['parent']}"
    if state.get("last_test") == "FAIL" or state.get("last_read_hash") != state.get("current_hash"):
        return f"read path={state['target']}"
    raise ValueError("state has no uniquely defined corrected action")

def encode(tokenizer, text):
    ids=tokenizer(text, add_special_tokens=False).input_ids
    if not ids: raise ValueError("empty encoded text")
    if any(type(x) is not int or not 0<=x<MODEL_VOCAB for x in ids): raise ValueError("token outside model vocabulary")
    return ids

def prepare_case(tokenizer, case, model):
    state=case["state"]
    visible={k:state[k] for k in sorted(state)}
    user=("You are operating a private resettable workspace.\n"+"\n".join(RULES)+
            "\nObserved state (use only this evidence):\n"+json.dumps(visible,sort_keys=True,separators=(",",":"))+"\nAction:")
    kwargs={"tokenize":False,"add_generation_prompt":True}
    if model=="ling": kwargs["enable_thinking"]=False
    rendered=tokenizer.apply_chat_template([{"role":"user","content":user}],**kwargs)
    controller_prefix="</think>\n" if model=="ring" else ""
    full=rendered+controller_prefix
    prompt_ids=encode(tokenizer,full)
    action=oracle(state)
    eos_id=tokenizer.eos_token_id
    if type(eos_id) is not int or not 0<=eos_id<MODEL_VOCAB: raise ValueError("EOS token missing or outside model vocabulary")
    target_text=action+tokenizer.eos_token
    target_ids=encode(tokenizer,target_text)
    joined_ids=encode(tokenizer,full+target_text)
    if joined_ids != prompt_ids+target_ids: raise ValueError("encoded prompt/target boundary changed")
    p=len(prompt_ids); rows=list(range(p-1,p-1+len(target_ids)))
    return {"case_id":case["id"],"split":case["split"],"source_lineage":case["lineage"],"prompt":full,"controller_prefix":controller_prefix,"prompt_ids":prompt_ids,"prompt_length":p,"corrected_action":action,"target_ids":target_ids,"target_row_indices":rows,"target_length":len(target_ids),"target_boundary_verified":True,"eos_token":tokenizer.eos_token,"eos_id":eos_id}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cases",required=True); ap.add_argument("--output",required=True); ap.add_argument("--retention-output",required=True); ap.add_argument("--tokenizer-dir",required=True); args=ap.parse_args()
    cases=json.loads(Path(args.cases).read_text())["cases"]
    if {x["split"] for x in cases}!={"train","dev","heldout"}: raise SystemExit("split set mismatch")
    expected={"train":16,"dev":8,"heldout":16}
    if {s:sum(c["split"]==s for c in cases) for s in expected} != expected: raise SystemExit("split counts mismatch")
    out=[]; pins={}
    for model in ("ling","ring"):
        tok_dir=Path(args.tokenizer_dir)/TOKENIZER_NAMES[model]
        tok_path=tok_dir/"tokenizer.json"; config_path=tok_dir/"config.json"; gen_path=tok_dir/"generation_config.json"
        tok=AutoTokenizer.from_pretrained(str(tok_dir),local_files_only=True)
        cfg=json.loads(config_path.read_text())
        if len(tok)!=157153 or int(cfg.get("vocab_size",0))!=MODEL_VOCAB: raise SystemExit("tokenizer/model vocab sanity mismatch")
        pins[model]={"tokenizer": {"path":str(tok_path),"sha256":sha(tok_path),"bytes":tok_path.stat().st_size,"len":len(tok)},"config":{"path":str(config_path),"sha256":sha(config_path)},"generation_config":{"path":str(gen_path),"sha256":sha(gen_path)},"model_vocab_size":MODEL_VOCAB,"eos_token":tok.eos_token,"eos_id":tok.eos_token_id,"enable_thinking":False if model=="ling" else None,"chat_template_sha256":hashlib.sha256(tok.chat_template.encode()).hexdigest()}
        if tok.eos_token_id is None: raise SystemExit("generation EOS missing")
        out += [{"model":model,"record":prepare_case(tok,c,model)} for c in cases]
    Path(args.output).write_text(json.dumps({"schema":"strata-corrected-action-prepared-v2","rules":RULES,"models":["ling","ring"],"records":out,"tokenizer_pins":pins},sort_keys=True,indent=2)+"\n")
    retention=[]
    for model in ("ling","ring"):
        tok=AutoTokenizer.from_pretrained(str(Path(args.tokenizer_dir)/TOKENIZER_NAMES[model]),local_files_only=True)
        for i,(prompt,answer) in enumerate(RETENTION):
            kwargs={"tokenize":False,"add_generation_prompt":True};
            if model=="ling": kwargs["enable_thinking"]=False
            rendered=tok.apply_chat_template([{"role":"user","content":prompt}],**kwargs); full=rendered+("</think>\n" if model=="ring" else "")
            pi=encode(tok,full); target=answer+tok.eos_token; ti=encode(tok,target)
            if encode(tok,full+target)!=pi+ti: raise SystemExit("retention prompt boundary mismatch")
            retention.append({"model":model,"probe_index":i,"prompt":full,"prompt_ids":pi,"target_answer":answer,"target_ids":ti,"target_row_indices":list(range(len(pi)-1,len(pi)-1+len(ti))),"train":False})
    Path(args.retention_output).write_text(json.dumps({"schema":"strata-retention-probes-v2","records":retention,"training":"never"},sort_keys=True,indent=2)+"\n")

if __name__=="__main__": main()
