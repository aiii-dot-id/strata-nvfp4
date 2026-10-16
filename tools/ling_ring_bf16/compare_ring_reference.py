#!/usr/bin/env python3
import argparse, hashlib, json, math, re
from pathlib import Path
import numpy as np

ATOL=0.001; RTOL=0.001; WATOL=1e-5; WRTOL=1e-4
NATIVE_OUTPUTS = {"routes.tsv", "final-hidden.f32", "logits.f32", "execution.json"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
def fail(s): raise ValueError(s)
def finite(a,n):
    if a.size != n or not np.isfinite(a).all(): fail("invalid, short, or nonfinite output")
def load(path,n):
    p=Path(path)
    if not p.is_file(): fail("missing output: "+p.name)
    if p.stat().st_size != n*4: fail("invalid or trailing output bytes: "+p.name)
    a=np.fromfile(p,dtype=np.float32); finite(a,n); return a
def verify_hashes(root, report):
    outputs=report.get("outputs")
    expected={"final_hidden.fp32", "logits.fp32", "hidden_layers.fp32", "layers.json", "router_meta.json"}
    if (root/"generation.json").is_file(): expected.add("generation.json")
    if (root/"route-probe.json").is_file(): expected.add("route-probe.json")
    if not isinstance(outputs, dict): fail("reference output hashes are missing")
    for name in expected:
        meta=outputs.get(name)
        if not isinstance(meta, dict) or not isinstance(meta.get("sha256"), str):
            fail("reference output hash is missing: "+name)
        if not HEX64.fullmatch(meta["sha256"]): fail("invalid reference output hash: "+name)
        p=root/name
        if not p.is_file(): fail("reference output is missing: "+name)
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        if h != meta["sha256"]: fail("reference output hash mismatch: "+name)

def pin(meta, label, require_size=True):
    if not isinstance(meta, dict) or not isinstance(meta.get("sha256"), str) or not HEX64.fullmatch(meta["sha256"]):
        fail("invalid provenance hash: "+label)
    if require_size and (type(meta.get("size")) is not int or meta["size"] <= 0):
        fail("invalid provenance size: "+label)
    return meta["sha256"]

def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _strict_ids(ids):
    return len(ids)==8 and all(type(x) is int and 0 <= x < 256 for x in ids) and len(set(ids))==8

def generation_trace(path):
    try: value=json.loads(Path(path).read_text())
    except Exception as e: fail("invalid generation trace: "+str(e))
    prompt=value.get("prompt_token_ids"); generated=value.get("generated_token_ids")
    if not isinstance(prompt,list) or not prompt or not all(type(x) is int and 0 <= x < 157184 for x in prompt): fail("invalid generation prompt trace")
    if not isinstance(generated,list) or not generated or not all(type(x) is int and 0 <= x < 157184 for x in generated): fail("invalid generation output trace")
    eos=value.get("eos_token_ids"); maximum=value.get("max_new_tokens")
    if not isinstance(eos,list) or not eos or any(type(x) is not int or x < 0 or x >= 157184 for x in eos) or len(set(eos)) != len(eos): fail("invalid generation EOS configuration")
    if type(maximum) is not int or maximum < 1 or maximum > 4096 or len(generated) > maximum: fail("invalid generation budget")
    if value.get("finish_reason") not in {"eos","length"}: fail("invalid generation finish reason")
    if any(x in eos for x in generated[:-1]): fail("generation continued after EOS")
    if value["finish_reason"] == "eos" and generated[-1] not in eos: fail("EOS generation does not end in EOS")
    if value["finish_reason"] == "length" and (len(generated) != maximum or generated[-1] in eos): fail("length generation does not consume the budget")
    return value

def verify_provenance(cand, execution, layers, tokens, ref_report):
    path=cand/"provenance.json"
    if not path.is_file(): fail("missing candidate provenance.json")
    try: provenance=json.loads(path.read_text())
    except Exception as e: fail("invalid candidate provenance: "+str(e))
    if provenance.get("schema") != "strata-ring-verified-run-v1" or provenance.get("status") != "verified":
        fail("candidate provenance is not a successful verified receipt")
    if ref_report.get("status") != "executed": fail("reference report is not an executed receipt")
    required={"status":"executed","model":"Ring-mini-2.0","activation_dtype":"FP32","quantized":False,"tokens":tokens,"layers":layers}
    for key,value in required.items():
        if execution.get(key) != value: fail("candidate execution receipt mismatch: "+key)
    if type(execution.get("quantized")) is not bool or execution["quantized"] is not False:
        fail("candidate execution receipt mismatch: quantized")
    if type(execution.get("tokens")) is not int or any(type(x) is not int for x in execution.get("token_ids", [])):
        fail("candidate execution receipt has non-integer token identity")
    if type(execution.get("layers")) is not int: fail("candidate execution receipt has non-integer layers")
    if execution.get("token_ids") != ref_report.get("tokens"): fail("candidate token identity mismatch")
    if execution.get("checkpoint_id") != provenance.get("checkpoint_id") or not isinstance(execution.get("checkpoint_id"),str) or not HEX64.fullmatch(execution["checkpoint_id"]):
        fail("candidate checkpoint identity is not bound")
    if provenance.get("checkpoint_id") != provenance.get("manifest",{}).get("sha256"):
        fail("candidate checkpoint identity is not the manifest digest")
    if provenance.get("execution") != execution: fail("provenance execution receipt differs from execution.json")
    pin(provenance.get("manifest"), "manifest")
    pin(provenance.get("executable"), "executable")
    relevant=provenance.get("relevant_source")
    if not isinstance(relevant, dict) or not relevant: fail("relevant source provenance is missing")
    for name, meta in relevant.items(): pin(meta, "relevant source "+str(name))
    original=provenance.get("original_source")
    if not isinstance(original, dict) or not original: fail("original source provenance is missing")
    for name, meta in original.items(): pin(meta, "original source "+str(name))
    if (cand/"generation.json").is_file():
        frontend=provenance.get("frontend")
        if not isinstance(frontend,dict): fail("generation frontend bindings are missing")
        for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            pin(frontend.get(name), "generation frontend "+name)
    checkpoint=provenance.get("checkpoint")
    ref_checkpoint=ref_report.get("checkpoint")
    if not isinstance(checkpoint,dict) or not isinstance(ref_checkpoint,dict): fail("checkpoint provenance is missing")
    if not isinstance(checkpoint.get("shards"),dict) or not checkpoint["shards"]: fail("candidate shard provenance is empty")
    if not isinstance(ref_checkpoint.get("shards"),dict) or not ref_checkpoint["shards"]: fail("reference shard provenance is empty")
    cfg=pin(checkpoint.get("config"), "candidate config"); idx=pin(checkpoint.get("index"), "candidate index")
    ref_cfg=pin(ref_checkpoint.get("config"), "reference config"); ref_idx=pin(ref_checkpoint.get("index"), "reference index")
    if cfg != ref_cfg:
        fail("candidate config hash does not match reference")
    if idx != ref_idx:
        fail("candidate index hash does not match reference")
    if checkpoint["config"]["size"] != ref_checkpoint["config"]["size"]:
        fail("candidate config size does not match reference")
    if checkpoint["index"]["size"] != ref_checkpoint["index"]["size"]:
        fail("candidate index size does not match reference")
    shards=checkpoint.get("shards"); ref_shards=ref_checkpoint.get("shards")
    if not isinstance(shards,dict) or not isinstance(ref_shards,dict) or set(shards) != set(ref_shards): fail("candidate shard set does not match reference")
    for name in shards:
        if pin(shards[name], "candidate shard "+name) != pin(ref_shards[name], "reference shard "+name): fail("candidate shard hash does not match reference: "+name)
        if shards[name]["size"] != ref_shards[name]["size"]: fail("candidate shard size does not match reference: "+name)
    ref_source=ref_report.get("source",{})
    # Older reports stored the modeling pin directly; current reports keep
    # runtime/source pins under named entries. Bind the candidate to both
    # original source files in either schema.
    ref_modeling = ref_source.get("modeling") if isinstance(ref_source, dict) and isinstance(ref_source.get("modeling"), dict) else ref_source
    if pin(original.get("modeling_bailing_moe_v2.py"), "original modeling source") != pin(ref_modeling, "reference modeling source", False):
        fail("candidate original modeling source does not match reference")
    ref_configuration = ref_source.get("configuration") if isinstance(ref_source, dict) and isinstance(ref_source.get("configuration"), dict) else None
    if ref_configuration is None and isinstance(ref_source, dict):
        legacy_configuration = ref_source.get("configuration_sha256")
        if isinstance(legacy_configuration, str):
            ref_configuration = {"sha256": legacy_configuration}
    if ref_configuration is None:
        fail("reference configuration source hash is missing")
    if pin(original.get("configuration_bailing_moe_v2.py"), "original configuration source") != pin(ref_configuration, "reference configuration source", False):
        fail("candidate original configuration source does not match reference")
    native=provenance.get("native_outputs")
    expected_native=NATIVE_OUTPUTS | {f"layer-{i}.f32" for i in range(layers)}
    if (cand/"generation.json").is_file(): expected_native.add("generation.json")
    if (cand/"route-probe.json").is_file(): expected_native.add("route-probe.json")
    if not isinstance(native,dict) or set(native) != expected_native: fail("candidate native output hashes are incomplete")
    for name in expected_native:
        meta=native.get(name)
        if not isinstance(meta,dict) or not isinstance(meta.get("sha256"),str): fail("candidate output hash is missing: "+name)
        pin(meta, "candidate output "+name)
        p=cand/name
        if not p.is_file() or p.stat().st_size != meta["size"] or _hash(p) != meta["sha256"]: fail("candidate output hash mismatch: "+name)
def routes_ref(path,layers,tokens,first_dense):
    try: rows=json.loads(Path(path).read_text())
    except Exception as e: fail("invalid layers.json: "+str(e))
    if not isinstance(rows,list) or len(rows)!=layers: fail("invalid reference route layers")
    out={}
    for layer,row in enumerate(rows):
        if row.get("layer") != layer: fail("reference layer identity mismatch")
        ids=row.get("router_ids"); weights=row.get("router_weights")
        if layer < first_dense:
            if ids != [] or weights != []: fail("dense reference layer has routes")
            continue
        if not isinstance(ids,list) or not isinstance(weights,list) or len(ids)!=tokens or len(weights)!=tokens: fail("invalid reference route rows")
        for pos in range(tokens):
            if not _strict_ids(ids[pos]) or len(weights[pos])!=8 or not all(math.isfinite(float(x)) for x in weights[pos]): fail("invalid reference route values")
            key=(pos,layer)
            if key in out: fail("duplicate reference route")
            out[key]=(ids[pos],weights[pos])
    return out
def routes_candidate(path,layers,tokens,first_dense):
    out={}
    for line in Path(path).read_text().splitlines():
        if not line: continue
        f=line.split("\t")
        if len(f)!=10: fail("invalid candidate route row")
        pos=int(f[0]); layer=int(f[1]); vals=f[2:]
        if layer < first_dense: fail("candidate route emitted for dense layer")
        if not (0<=pos<tokens and 0<=layer<layers) or len(vals)!=8: fail("invalid candidate route index/count")
        key=(pos,layer)
        if key in out: fail("duplicate candidate route")
        ids=[]; ws=[]
        for x in vals:
            q=x.split(":")
            if len(q)!=2: fail("invalid candidate route value")
            ids.append(int(q[0])); ws.append(float(q[1]))
        if not _strict_ids(ids) or not all(math.isfinite(x) for x in ws): fail("invalid candidate route values")
        out[key]=(ids,ws)
    if len(out)!=(layers-first_dense)*tokens: fail("missing candidate route row")
    return out
def compare(reference,candidate,output):
    ref=Path(reference); cand=Path(candidate); output=Path(output)
    if output.exists(): fail("refusing existing output")
    report=json.loads((ref/"report.json").read_text())
    verify_hashes(ref, report)
    hs=report.get("hidden_layers_shape"); toks=report.get("tokens")
    if not isinstance(hs,list) or len(hs)!=3 or not isinstance(toks,list): fail("invalid reference report")
    layers,tokens,hidden=map(int,hs); vocab=int(report.get("config",{}).get("vocab_size",157184))
    if layers<1 or tokens<1 or hidden<1 or len(toks)!=tokens: fail("invalid reference dimensions")
    ref_layers=load(ref/"hidden_layers.fp32",layers*tokens*hidden).reshape(layers,tokens,hidden)
    ref_final=load(ref/"final_hidden.fp32",tokens*hidden).reshape(tokens,hidden)
    all_layers=[]
    for i in range(layers): all_layers.append(load(cand/f"layer-{i}.f32",tokens*hidden).reshape(tokens,hidden))
    cand_layers=np.stack(all_layers); cand_final=load(cand/"final-hidden.f32",tokens*hidden).reshape(tokens,hidden)
    execution=json.loads((cand/"execution.json").read_text())
    if execution.get("tokens")!=tokens or execution.get("token_ids")!=toks: fail("candidate token identity mismatch")
    mode=execution.get("logits_mode", "all")
    if mode not in {"all", "last"}: fail("invalid candidate logits mode")
    declared_rows=execution.get("logits_tokens")
    if declared_rows is not None and declared_rows != (1 if mode == "last" else tokens): fail("candidate logits row count mismatch")
    logits_shape=report.get("logits_shape")
    if isinstance(logits_shape,list) and len(logits_shape)==2:
        ref_logit_tokens, ref_vocab = map(int, logits_shape)
        if ref_vocab != vocab: fail("reference logits vocabulary mismatch")
    elif isinstance(logits_shape,list) and len(logits_shape)==3:
        batch, ref_logit_tokens, ref_vocab = map(int, logits_shape)
        if batch != 1 or ref_vocab != vocab: fail("reference logits shape must be [1,tokens,vocab]")
    else:
        ref_logit_tokens = int(report.get("logits_tokens", tokens))
    if ref_logit_tokens < 1: fail("invalid reference logits row count")
    cand_logit_tokens = 1 if mode == "last" else tokens
    if cand_logit_tokens != ref_logit_tokens: fail("reference and candidate logits modes differ")
    ref_logits=load(ref/"logits.fp32",ref_logit_tokens*vocab).reshape(ref_logit_tokens,vocab)
    cand_logits=load(cand/"logits.f32",cand_logit_tokens*vocab).reshape(cand_logit_tokens,vocab)
    first_dense=int(report.get("config",{}).get("first_k_dense_replace",1))
    if not 0 <= first_dense <= layers: fail("invalid dense layer count")
    verify_provenance(cand, execution, layers, tokens, report)
    cand_generation=(cand/"generation.json").is_file()
    if cand_generation:
        cg=generation_trace(cand/"generation.json")
        prompt=cg["prompt_token_ids"]; generated=cg["generated_token_ids"]
        if toks[:len(prompt)] != prompt: fail("candidate generation prompt is outside evaluated token trace")
        if toks != prompt + generated[:-1]: fail("candidate generation trace does not match evaluated token trace")
        if execution.get("logits_mode") != "all" or execution.get("logits_tokens") != tokens or ref_logit_tokens != tokens:
            fail("generation comparison requires all logits for the evaluated trace")
        start=len(prompt)-1
        if start < 0 or start + len(generated) > ref_logits.shape[0]: fail("generation trace exceeds reference logits")
        expected=np.argmax(ref_logits[start:start+len(generated)], axis=1).astype(int).tolist()
        if generated != expected: fail("candidate generation is not the original greedy trace")
    rr=routes_ref(ref/"layers.json",layers,tokens,first_dense); cr=routes_candidate(cand/"routes.tsv",layers,tokens,first_dense)
    def maxerr(a,b): return float(np.max(np.abs(a-b)))
    errors={"hidden_layers_max_abs":maxerr(ref_layers,cand_layers),"final_hidden_max_abs":maxerr(ref_final,cand_final),"logits_max_abs":maxerr(ref_logits,cand_logits),"route_weight_max_abs":0.0}
    route_mismatches=[]; route_set_mismatches=[]
    for key,(ri,rw) in rr.items():
        ci,cw=cr[key]
        if ci!=ri: route_mismatches.append(key)
        if set(ci)!=set(ri): route_set_mismatches.append(key)
        errors["route_weight_max_abs"]=max(errors["route_weight_max_abs"],max(abs(float(a)-float(b)) for a,b in zip(rw,cw)))
    top_exact=bool(np.array_equal(np.argmax(ref_logits,axis=1),np.argmax(cand_logits,axis=1)))
    weight_pass=all(abs(float(a)-float(b)) <= WATOL+WRTOL*abs(float(a)) for key,(ri,rw) in rr.items() for a,b in zip(rw,cr[key][1]))
    passed=bool(not route_mismatches and np.all(np.abs(ref_layers-cand_layers)<=ATOL+RTOL*np.abs(ref_layers)) and np.all(np.abs(ref_final-cand_final)<=ATOL+RTOL*np.abs(ref_final)) and np.all(np.abs(ref_logits-cand_logits)<=ATOL+RTOL*np.abs(ref_logits)) and weight_pass and top_exact)
    result={"schema":"ring-reference-compare-v1","passed":passed,"layers":layers,"tokens":tokens,"vocab":vocab,"logits_tokens":ref_logit_tokens,"route_count":len(rr),"top1_exact":top_exact,"max_errors":errors,"tolerances":{"output":"0.001 + 0.001 * abs(reference)","weights":"1e-5 + 1e-4 * abs(reference)"}}
    result["reference_decoding_mode"]=report.get("decoding_mode", "full")
    result["route_mismatch_count"]=len(route_mismatches)
    result["route_set_mismatch_count"]=len(route_set_mismatches)
    result["first_route_mismatches"]=[{"position":key[0],"layer":key[1],"reference":rr[key][0],"candidate":cr[key][0]} for key in route_mismatches[:8]]
    output.mkdir(parents=True); (output/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    if not passed: fail("comparison failed")
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument("--reference",required=True);p.add_argument("--candidate",required=True);p.add_argument("--output",required=True)
    args=p.parse_args()
    try: print(json.dumps(compare(args.reference,args.candidate,args.output),indent=2))
    except Exception as e:
        out=Path(args.output)
        if not out.exists():
            out.mkdir(parents=True); (out/"result.json").write_text(json.dumps({"schema":"ring-reference-compare-v1","passed":False,"error":str(e)},indent=2)+"\n")
        p.error(str(e))
if __name__=="__main__": main()
