#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
from inspect_checkpoint import CheckpointError, inspect_checkpoint

def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise CheckpointError(f"duplicate JSON key: {key}")
        out[key] = value
    return out

EXPECTED = {
    "model_type": "bailing_moe", "num_hidden_layers": 20, "vocab_size": 157184,
    "hidden_size": 2048, "intermediate_size": 5120,
    "moe_intermediate_size": 512, "num_attention_heads": 16, "num_key_value_heads": 4,
    "head_dim": 128, "partial_rotary_factor": 0.5, "rope_theta": 600000,
    "rms_norm_eps": 1e-6, "first_k_dense_replace": 1, "num_experts": 256,
    "num_experts_per_tok": 8, "n_group": 8, "topk_group": 4,
    "num_shared_experts": 1, "routed_scaling_factor": 2.5,
    "use_bias": False, "use_qkv_bias": False, "use_qk_norm": True,
    "tie_word_embeddings": False, "hidden_act": "silu", "attention_dropout": 0.0,
    "score_function": "sigmoid", "rope_scaling": None,
}

def _same(a, b):
    return type(a) is type(b) and a == b

def _guard(config):
    for key, expected in EXPECTED.items():
        if key in config and not _same(config[key], expected):
            raise CheckpointError(f"Ring config mismatch: {key}={config[key]!r}, expected {expected!r}")
        if key not in config and expected is not None:
            raise CheckpointError(f"Ring config missing required field: {key}")

def _json_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(checkpoint, output):
    checkpoint = Path(checkpoint); output = Path(output)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    _guard(config)
    inspection = inspect_checkpoint(checkpoint)
    rows=[]; header_hashes={}
    for name, rec in inspection["tensors"].items():
        if rec["dtype"] not in {"BF16", "F32"}:
            raise CheckpointError(f"unsupported Ring tensor dtype: {name}: {rec['dtype']}")
        start, end = rec["data_offsets"]
        rows.append("\t".join([name, rec["dtype"], ",".join(map(str, rec["shape"])), rec["shard"], str(rec["header_bytes"]+start), str(end-start), str((checkpoint/rec["shard"]).stat().st_size)]))
    rows.sort()
    manifest = "".join(x+"\n" for x in rows).encode()
    output.mkdir(parents=True, exist_ok=False)
    (output/"manifest.tsv").write_bytes(manifest)
    for rec in inspection["files"]:
        if rec.get("hash_scope")=="safetensors_header": header_hashes[rec["name"]]=rec["sha256"]
    admission={"schema":"strata-ring-bf16-manifest-v1","scope":"structural/source binding; tensor payloads not hashed","checkpoint":str(checkpoint.resolve()),"config_sha256":_json_hash(config_path),"index_sha256":_json_hash(checkpoint/"model.safetensors.index.json"),"source_sha256":{"prepare_ring_manifest.py":_json_hash(Path(__file__))},"inspector_sha256":_json_hash(Path(__file__).with_name("inspect_checkpoint.py")),"header_sha256":header_hashes,"manifest_sha256":hashlib.sha256(manifest).hexdigest(),"tensor_count":len(rows),"parameter_count":inspection["parameter_count"],"payload_bytes":inspection["payload_bytes"],"dtype_payload_bytes":inspection["dtype_payload_bytes"]}
    (output/"admission.json").write_text(json.dumps(admission,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return admission

def main():
    p=argparse.ArgumentParser();p.add_argument("checkpoint",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    try: prepare(a.checkpoint,a.output)
    except (OSError,ValueError,CheckpointError) as e: p.error(str(e))
if __name__=="__main__": main()
