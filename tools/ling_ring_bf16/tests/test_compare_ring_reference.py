import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from compare_ring_reference import compare

def make(tmp, corrupt=None):
    ref=tmp/"ref"; cand=tmp/"cand"; ref.mkdir(); cand.mkdir(); L,T,H,V=2,3,4,7
    x=np.arange(L*T*H,dtype=np.float32); x.tofile(ref/"hidden_layers.fp32"); (x[:T*H]+1).reshape(T,H).tofile(ref/"final_hidden.fp32"); np.arange(T*V,dtype=np.float32).tofile(ref/"logits.fp32")
    layers=[]
    lines=[]
    for l in range(L):
        ids=[]; ws=[]
        for p in range(T):
            if l == 0: continue
            ids.append(list(range(8))); ws.append([0.1]*8); lines.append(f"{p}\t{l}\t"+"\t".join(f"{i}:0.1" for i in range(8)))
        layers.append({"layer":l,"router_ids":ids,"router_weights":ws})
        (x[l*T*H:(l+1)*T*H]).reshape(T,H).tofile(cand/f"layer-{l}.f32")
    (x[:T*H].reshape(T,H)+1).tofile(cand/"final-hidden.f32"); np.arange(T*V,dtype=np.float32).tofile(cand/"logits.f32"); (cand/"routes.tsv").write_text("\n".join(lines)+"\n"); (ref/"layers.json").write_text(json.dumps(layers)); (ref/"router_meta.json").write_text("[]")
    ref_outputs={}
    for name in ("hidden_layers.fp32","final_hidden.fp32","logits.fp32","layers.json","router_meta.json"):
        ref_outputs[name]={"sha256":hashlib.sha256((ref/name).read_bytes()).hexdigest()}
    pins={"config":{"sha256":"c"*64,"size":1},"index":{"sha256":"d"*64,"size":1},"shards":{"shard.safetensors":{"sha256":"e"*64,"size":1}}}
    (ref/"report.json").write_text(json.dumps({"status":"executed","hidden_layers_shape":[L,T,H],"tokens":[1,2,3],"config":{"vocab_size":V,"first_k_dense_replace":1},"checkpoint":pins,"source":{"sha256":"f"*64,"configuration_sha256":"1"*64},"outputs":ref_outputs}))
    (cand/"execution.json").write_text(json.dumps({"status":"executed","model":"Ring-mini-2.0","activation_dtype":"FP32","quantized":False,"tokens":T,"token_ids":[1,2,3],"layers":L,"checkpoint_id":"a"*64}))
    native={}
    for name in ("routes.tsv","final-hidden.f32","logits.f32","execution.json",*list(f"layer-{i}.f32" for i in range(L))):
        native[name]={"sha256":hashlib.sha256((cand/name).read_bytes()).hexdigest(),"size":(cand/name).stat().st_size}
    source_pin={"sha256":"f"*64,"size":1}; configuration_pin={"sha256":"1"*64,"size":1}
    (cand/"provenance.json").write_text(json.dumps({"schema":"strata-ring-verified-run-v1","status":"verified","checkpoint_id":"a"*64,"manifest":{"sha256":"a"*64,"size":1},"executable":{"sha256":"1"*64,"size":1},"checkpoint":pins,"original_source":{"modeling_bailing_moe_v2.py":source_pin,"configuration_bailing_moe_v2.py":configuration_pin},"relevant_source":{"src/ring/engine.cu":source_pin},"execution":json.loads((cand/"execution.json").read_text()),"native_outputs":native}))
    if corrupt: corrupt(ref,cand)
    return ref,cand
def test_pass(tmp_path):
    r,c=make(tmp_path); assert compare(r,c,tmp_path/"out")["passed"]

def test_reference_three_dimensional_logits_shape(tmp_path):
    r,c=make(tmp_path)
    report=json.loads((r/"report.json").read_text()); report["logits_shape"]=[1,3,7]; (r/"report.json").write_text(json.dumps(report))
    assert compare(r,c,tmp_path/"out")["passed"]

def test_current_nested_modeling_source_pin(tmp_path):
    r,c=make(tmp_path)
    report=json.loads((r/"report.json").read_text()); report["source"]={"modeling":{"sha256":"f"*64,"size":1},"configuration":{"sha256":"1"*64,"size":1}}; (r/"report.json").write_text(json.dumps(report))
    assert compare(r,c,tmp_path/"out")["passed"]

def test_configuration_source_mismatch_fails(tmp_path):
    r,c=make(tmp_path)
    provenance=json.loads((c/"provenance.json").read_text())
    provenance["original_source"]["configuration_bailing_moe_v2.py"]["sha256"]="2"*64
    (c/"provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="configuration source"):
        compare(r,c,tmp_path/"out")

def test_generation_trace_is_exact(tmp_path):
    r,c=make(tmp_path)
    trace={"prompt_token_ids":[1,2],"generated_token_ids":[6,6],"eos_token_ids":[9],"max_new_tokens":2,"finish_reason":"length"}
    (c/"generation.json").write_text(json.dumps(trace))
    report=json.loads((r/"report.json").read_text()); report["tokens"]=[1,2,6]; (r/"report.json").write_text(json.dumps(report))
    execution=json.loads((c/"execution.json").read_text()); execution["token_ids"]=[1,2,6]; execution["logits_mode"]="all"; execution["logits_tokens"]=3; (c/"execution.json").write_text(json.dumps(execution))
    p=json.loads((c/"provenance.json").read_text()); p["execution"]=execution; p["native_outputs"]["execution.json"]={"sha256":hashlib.sha256((c/"execution.json").read_bytes()).hexdigest(),"size":(c/"execution.json").stat().st_size}; p["native_outputs"]["generation.json"]={"sha256":hashlib.sha256((c/"generation.json").read_bytes()).hexdigest(),"size":(c/"generation.json").stat().st_size}; p["frontend"]={name:{"sha256":"1"*64,"size":1} for name in ("tokenizer.json","tokenizer_config.json","generation_config.json")}; (c/"provenance.json").write_text(json.dumps(p))
    assert compare(r,c,tmp_path/"out")["passed"]
    trace["generated_token_ids"]=[5,9]; (c/"generation.json").write_text(json.dumps(trace))
    with pytest.raises(ValueError): compare(r,c,tmp_path/"bad")
def test_corruption_fails(tmp_path):
    def bad(r,c):
        a=np.fromfile(c/"layer-0.f32",dtype=np.float32);a[0]=99;a.tofile(c/"layer-0.f32")
    r,c=make(tmp_path,bad)
    try: compare(r,c,tmp_path/"out")
    except ValueError: pass
    else: assert False
def test_missing_truncated_route_nonfinite_fail(tmp_path):
    def bad(r,c): (c/"layer-1.f32").unlink()
    r,c=make(tmp_path,bad)
    try: compare(r,c,tmp_path/"out")
    except ValueError: pass
    else: assert False

def test_route_and_nonfinite_fail(tmp_path):
    def bad(r,c):
        lines=(c/"routes.tsv").read_text().splitlines()
        lines[0]=lines[0].replace("0:0.1", "1:nan")
        (c/"routes.tsv").write_text("\n".join(lines)+"\n")
    r,c=make(tmp_path,bad)
    try: compare(r,c,tmp_path/"out")
    except ValueError: pass
    else: assert False

@pytest.mark.parametrize("mutation", ["wrong-source", "wrong-checkpoint", "identity", "model", "precision", "status", "missing-receipt", "changed-receipt", "changed-output", "wrong-output-hash"])
def test_provenance_failures(tmp_path, mutation):
    r,c=make(tmp_path)
    provenance=json.loads((c/"provenance.json").read_text())
    execution=json.loads((c/"execution.json").read_text())
    if mutation == "wrong-source": provenance["original_source"]["modeling_bailing_moe_v2.py"]["sha256"]="bad"
    elif mutation == "wrong-checkpoint": provenance["checkpoint"]["shards"]["shard.safetensors"]["sha256"]="bad"
    elif mutation == "identity": execution["checkpoint_id"]="b"*64
    elif mutation == "model": execution["model"]="Other"
    elif mutation == "precision": execution["activation_dtype"]="BF16"
    elif mutation == "status": execution["status"]="failed"
    elif mutation == "missing-receipt": (c/"provenance.json").unlink()
    elif mutation == "changed-receipt": execution["tokens"]=99
    elif mutation == "changed-output": (c/"logits.f32").write_bytes((c/"logits.f32").read_bytes()+b"x")
    elif mutation == "wrong-output-hash": provenance["native_outputs"]["logits.f32"]["sha256"]="bad"
    if mutation not in ("missing-receipt", "changed-output"):
        (c/"execution.json").write_text(json.dumps(execution))
    if mutation != "missing-receipt":
        (c/"provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError): compare(r,c,tmp_path/"out")

@pytest.mark.parametrize("mutation", ["missing-config-both", "missing-index-both", "missing-source-both", "missing-configuration-source-both", "empty-shards-both", "malformed-hash-both", "size-both", "reference-status"])
def test_common_provenance_omissions_fail_closed(tmp_path, mutation):
    r,c=make(tmp_path)
    report=json.loads((r/"report.json").read_text())
    provenance=json.loads((c/"provenance.json").read_text())
    if mutation == "missing-config-both":
        del report["checkpoint"]["config"]; del provenance["checkpoint"]["config"]
    elif mutation == "missing-index-both":
        del report["checkpoint"]["index"]; del provenance["checkpoint"]["index"]
    elif mutation == "missing-source-both":
        del report["source"]["sha256"]; del provenance["original_source"]["modeling_bailing_moe_v2.py"]
    elif mutation == "missing-configuration-source-both":
        del report["source"]["configuration_sha256"]; del provenance["original_source"]["configuration_bailing_moe_v2.py"]
    elif mutation == "empty-shards-both":
        report["checkpoint"]["shards"]={}; provenance["checkpoint"]["shards"]={}
    elif mutation == "malformed-hash-both":
        report["checkpoint"]["config"]["sha256"]="bad"; provenance["checkpoint"]["config"]["sha256"]="bad"
    elif mutation == "size-both":
        report["checkpoint"]["config"]["size"]=2
    else:
        report["status"]="failed"
    (r/"report.json").write_text(json.dumps(report)); (c/"provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError): compare(r,c,tmp_path/"out")

@pytest.mark.parametrize("mutation", ["trailing", "truncated", "nan", "route", "weight", "duplicate", "missingrow", "tokens", "layerid"])
def test_independent_failures(tmp_path, mutation):
    r,c=make(tmp_path)
    if mutation in ("trailing", "truncated", "nan"):
        p=c/"logits.f32"
        if mutation=="trailing": p.write_bytes(p.read_bytes()+b"x")
        elif mutation=="truncated": p.write_bytes(p.read_bytes()[:-1])
        else:
            x=np.fromfile(p,dtype=np.float32); x[0]=np.nan; x.tofile(p)
    elif mutation in ("route","weight","duplicate","missingrow"):
        p=c/"routes.tsv"; rows=p.read_text().splitlines()
        if mutation=="route": rows[0]=rows[0].replace("0:0.1","9:0.1")
        elif mutation=="weight":
            rows[0]=rows[0].replace("0:0.1","0:0.101")
            # Large, identical hidden states must not relax route-weight limits.
            x=np.full(24,1e6,dtype=np.float32); x.tofile(r/"hidden_layers.fp32")
            x[:12].tofile(c/"layer-0.f32"); x[12:].tofile(c/"layer-1.f32")
        elif mutation=="duplicate": rows.append(rows[0])
        else: rows.pop()
        p.write_text("\n".join(rows)+"\n")
    elif mutation=="tokens":
        (c/"execution.json").write_text(json.dumps({"tokens":3,"token_ids":[1,2,4]}))
    else:
        p=r/"layers.json"; x=json.loads(p.read_text()); x[1]["layer"]=0; p.write_text(json.dumps(x))
    with pytest.raises(ValueError): compare(r,c,tmp_path/"out")
