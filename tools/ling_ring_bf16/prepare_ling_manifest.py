#!/usr/bin/env python3
"""CPU-only structural admission for the Ling-3.0-tiny safetensors package.

This reads JSON and safetensors headers only. Payload bytes are never read or
hashed; the resulting manifest is a source binding and schema check, not a
weights-integrity or runtime-readiness claim.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, struct
from pathlib import Path

class AdmissionError(ValueError): pass

HEADER_LIMIT = 64 * 1024 * 1024
DTYPE_BYTES = {"BF16": 2, "F32": 4}

EXPECTED_CONFIG = {
    "model_type":"bailing_hybrid", "num_hidden_layers":24, "hidden_size":1536,
    "vocab_size":157184, "num_experts":128, "num_experts_per_tok":8,
    "n_group":8, "topk_group":4, "first_k_dense_replace":1,
    "moe_intermediate_size":512, "intermediate_size":4608,
    "num_attention_heads":16, "head_dim":128, "kv_lora_rank":512,
    "q_lora_rank":256, "qk_head_dim":192, "qk_nope_head_dim":128,
    "qk_rope_head_dim":64, "v_head_dim":128, "num_key_value_heads":16,
    "gated_attention_proj_granularity_type":"head_wise",
    "kda_safe_gate":True, "kda_lower_bound":-5, "layer_group_size":4,
    "use_qk_norm":True, "score_function":"sigmoid", "scoring_func":"sigmoid",
    "routed_scaling_factor":2.5, "torch_dtype":"bfloat16",
    "no_kda_lora":True, "use_kda_lora":False, "mtp_use_kda":False,
    "mtp_loss_scaling_factor":0, "num_nextn_predict_layers":0,
    "num_shared_experts":1, "moe_shared_expert_intermediate_size":512,
    "short_conv_kernel_size":4, "rope_interleave":True,
}


def _pairs(pairs):
    d={}
    for k,v in pairs:
        if k in d: raise AdmissionError(f"duplicate JSON key: {k}")
        d[k]=v
    return d


def _json(path):
    try: return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as e: raise AdmissionError(f"invalid JSON {path}: {e}")


def _sha_bytes(b): return hashlib.sha256(b).hexdigest()
def _sha_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def _safe_shard(name):
    if not isinstance(name,str) or not name or Path(name).is_absolute() or "\\" in name:
        raise AdmissionError(f"unsafe shard path: {name!r}")
    p=Path(name)
    if any(x in ("",".","..") for x in p.parts) or str(p)!=name:
        raise AdmissionError(f"unsafe shard path: {name!r}")
    if len(p.parts)!=1 or not name.endswith(".safetensors"):
        raise AdmissionError(f"unsafe shard path: {name!r}")
    return p


def _header(path):
    try:
        with path.open("rb") as f:
            raw=f.read(8)
            if len(raw)!=8: raise AdmissionError(f"short safetensors header: {path.name}")
            n=struct.unpack("<Q",raw)[0]
            if n>HEADER_LIMIT: raise AdmissionError(f"header exceeds limit: {path.name}")
            h=f.read(n)
            if len(h)!=n: raise AdmissionError(f"truncated safetensors header: {path.name}")
            data=json.loads(h.decode("utf-8"), object_pairs_hook=_pairs)
            if not isinstance(data,dict): raise AdmissionError(f"header is not object: {path.name}")
            return n,h,data
    except (OSError, UnicodeError, json.JSONDecodeError, struct.error) as e:
        if isinstance(e,AdmissionError): raise
        raise AdmissionError(f"invalid safetensors header {path}: {e}")


def _shape(x):
    if not isinstance(x,list) or not x or any(type(v) is not int or v<=0 for v in x):
        raise AdmissionError(f"invalid tensor shape: {x!r}")
    return tuple(x)


def _records(root,index):
    allr={}; files=[]
    for shard_name in sorted(set(index.values())):
        shard=_safe_shard(shard_name); path=root/shard
        spans=[]
        if not path.is_file() or path.is_symlink(): raise AdmissionError(f"missing or symlink shard: {shard_name}")
        size=path.stat().st_size; n,h,header=_header(path)
        files.append({"name":shard_name,"size":size,"header_bytes":8+n,"header_sha256":_sha_bytes(h)})
        for name,rec in header.items():
            if name=="__metadata__": continue
            if not isinstance(rec,dict) or name in allr: raise AdmissionError(f"duplicate tensor header: {name}")
            dtype=rec.get("dtype"); shape=_shape(rec.get("shape")); offsets=rec.get("data_offsets")
            if dtype not in DTYPE_BYTES or not isinstance(offsets,list) or len(offsets)!=2 or any(type(x) is not int for x in offsets):
                raise AdmissionError(f"invalid tensor record: {name}")
            start,end=offsets
            if start<0 or end<=start or end>size-(8+n): raise AdmissionError(f"offset bounds: {name}")
            elems=1
            for x in shape: elems*=x
            if end-start != elems*DTYPE_BYTES[dtype]: raise AdmissionError(f"dtype/shape bytes mismatch: {name}")
            spans.append((start,end,name))
            allr[name]={"dtype":dtype,"shape":list(shape),"data_offsets":[start,end],"shard":shard_name,"header_bytes":8+n,"shard_size":size}
        spans.sort()
        payload_size=size-(8+n)
        if spans and (spans[0][0] != 0 or spans[-1][1] != payload_size):
            raise AdmissionError(f"unindexed payload gap: {shard_name}")
        for left,right in zip(spans,spans[1:]):
            if left[1] != right[0]:
                raise AdmissionError(f"overlapping or gapped payload: {shard_name}: {left[2]}")
    return allr,files


def expected_records(config):
    h=1536; a=2048; q=1536; f=4608; e=512; ex=128
    r={"model.word_embeddings.weight":("BF16",(157184,h)),"model.norm.weight":("BF16",(h,)),"lm_head.weight":("BF16",(157184,h))}
    mla={3,7,11,15,19,23}
    for i in range(24):
        p=f"model.layers.{i}."; r[p+"input_layernorm.weight"]=("BF16",(h,)); r[p+"post_attention_layernorm.weight"]=("BF16",(h,))
        if i in mla:
            for n,s in [("attention.dense.weight",(h,a)),("attention.g_proj.weight",(16,h)),("attention.kv_a_layernorm.weight",(512,)),("attention.kv_a_proj_with_mqa.weight",(576,h)),("attention.kv_b_proj.weight",(4096,512)),("attention.q_a_layernorm.weight",(256,)),("attention.q_a_proj.weight",(256,h)),("attention.q_b_proj.weight",(3072,256))]: r[p+n]=("BF16",s)
        else:
            for n,s in [("attention.A_log",(16,)),("attention.dt_bias",(a,)),("attention.b_proj.weight",(16,h)),("attention.f_proj.weight",(a,h)),("attention.g_proj.weight",(a,h)),("attention.k_conv1d.weight",(a,1,4)),("attention.k_proj.weight",(a,h)),("attention.o_norm.weight",(128,)),("attention.o_proj.weight",(h,a)),("attention.q_conv1d.weight",(a,1,4)),("attention.q_proj.weight",(a,h)),("attention.v_conv1d.weight",(a,1,4)),("attention.v_proj.weight",(a,h))]: r[p+n]=("F32" if n in ("attention.A_log","attention.dt_bias") else "BF16",s)
        if i==0:
            for n,s in [("mlp.down_proj.weight",(h,f)),("mlp.gate_proj.weight",(f,h)),("mlp.up_proj.weight",(f,h))]: r[p+n]=("BF16",s)
        else:
            r[p+"mlp.gate.weight"]=("BF16",(ex,h)); r[p+"mlp.gate.expert_bias"]=("F32",(ex,))
            for x,sh in [("down_proj.weight",(h,e)),("gate_proj.weight",(e,h)),("up_proj.weight",(e,h))]:
                for j in range(ex): r[f"{p}mlp.experts.{j}.{x}"]=("BF16",sh)
            for x,sh in [("down_proj.weight",(h,e)),("gate_proj.weight",(e,h)),("up_proj.weight",(e,h))]: r[p+"mlp.shared_experts."+x]=("BF16",sh)
    return r


def validate_config(c):
    if not isinstance(c,dict): raise AdmissionError("config is not an object")
    for k,v in EXPECTED_CONFIG.items():
        if c.get(k) != v: raise AdmissionError(f"Ling config mismatch: {k}={c.get(k)!r}, expected {v!r}")
    if c.get("architectures") != ["BailingMoeV3ForCausalLM"]: raise AdmissionError("unexpected Ling architecture")
    if c.get("num_hidden_layers") % c.get("layer_group_size") != 0: raise AdmissionError("layer grouping mismatch")


def validate_index(weight, records):
    if not isinstance(weight,dict) or not weight: raise AdmissionError("empty or missing weight_map")
    for name, shard in weight.items():
        if name not in records: raise AdmissionError(f"index tensor missing from headers: {name}")
        if records[name]["shard"] != shard: raise AdmissionError(f"index/header shard mismatch: {name}")
    if set(records) != set(weight):
        extra=sorted(set(records)-set(weight)); raise AdmissionError(f"unindexed header tensor: {extra[:3]}")


def validate_records(records, config):
    exp=expected_records(config)
    if set(records)!=set(exp):
        missing=sorted(set(exp)-set(records)); extra=sorted(set(records)-set(exp)); raise AdmissionError(f"Ling tensor schema mismatch: missing={missing[:3]} extra={extra[:3]}")
    for n,(dtype,shape) in exp.items():
        r=records[n]
        if r["dtype"]!=dtype or tuple(r["shape"])!=shape: raise AdmissionError(f"Ling tensor shape/dtype mismatch: {n}")


def prepare(checkpoint,output):
    root=Path(checkpoint); out=Path(output)
    if out.exists(): raise AdmissionError(f"refusing existing output: {out}")
    cfgp=root/"config.json"; idxp=root/"model.safetensors.index.json"
    config=_json(cfgp); validate_config(config); index=_json(idxp)
    weight=index.get("weight_map")
    if not isinstance(weight,dict) or not weight: raise AdmissionError("empty or missing weight_map")
    for n,s in weight.items(): _safe_shard(s)
    records,files=_records(root,weight)
    validate_index(weight, records)
    validate_records(records,config)
    out.mkdir(parents=False)
    manifest={"schema":"ling-bf16-structural-admission-v1","scope":"headers/config/index only; payloads not read or hashed","checkpoint":str(root.resolve()),"config_sha256":_sha_file(cfgp),"index_sha256":_sha_file(idxp),"files":files,"tensor_count":len(records),"tensors":dict(sorted(records.items())),"assumptions":{"layers":24,"dense_layers":[0],"kda_layers":[i for i in range(24) if i not in [3,7,11,15,19,23]],"mla_layers":[3,7,11,15,19,23],"weights":"BF16; router expert_bias and attention A_log/dt_bias F32"}}
    rows=[]
    for name,rec in sorted(records.items()):
        start,end=rec["data_offsets"]
        rows.append("\t".join([name,rec["dtype"],",".join(map(str,rec["shape"])),rec["shard"],str(rec["header_bytes"]+start),str(end-start),str(rec["shard_size"])]))
    tsv=("\n".join(rows)+"\n").encode(); (out/"manifest.tsv").write_bytes(tsv)
    manifest["manifest_sha256"]=_sha_bytes(tsv); manifest["tsv_format"]="name\tdtype\tshape\tshard\tabs_offset\tbytes\tshard_size"
    raw=json.dumps(manifest,sort_keys=True,indent=2).encode()+b"\n"; (out/"manifest.json").write_bytes(raw)
    return manifest

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("checkpoint",type=Path); ap.add_argument("output",type=Path); a=ap.parse_args()
    try: prepare(a.checkpoint,a.output)
    except (OSError,AdmissionError) as e: ap.error(str(e))
if __name__=="__main__": main()
