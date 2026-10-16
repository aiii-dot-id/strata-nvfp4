#!/usr/bin/env python3
"""CPU FP32 reference runner for the original Ring implementation."""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, mmap, struct, sys, types
from pathlib import Path

class ReferenceError(RuntimeError): pass

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ReferenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def read_json(path):
    try: return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc: raise ReferenceError(f"invalid JSON {path}: {exc}") from exc

class SafeTensorIndex:
    def __init__(self, root):
        self.root = Path(root); self.index = read_json(self.root / "model.safetensors.index.json")
        self.weight_map = self.index.get("weight_map", {})
        if not self.weight_map: raise ReferenceError("empty safetensors weight map")
        self.files, self.headers = {}, {}
        for name in sorted(set(self.weight_map.values())):
            path = self.root / name
            if Path(name).is_absolute() or ".." in Path(name).parts or not path.is_file() or path.is_symlink(): raise ReferenceError(f"unsafe or missing shard: {name}")
            with path.open("rb") as f:
                raw_len = f.read(8)
                if len(raw_len) != 8: raise ReferenceError(f"truncated shard: {name}")
                header_len = struct.unpack("<Q", raw_len)[0]
                if header_len > 64 * 1024 * 1024 or header_len + 8 > path.stat().st_size: raise ReferenceError(f"invalid header span: {name}")
                raw = raw_len + f.read(header_len)
            if len(raw) != header_len + 8: raise ReferenceError(f"truncated header: {name}")
            self.files[name] = {"path": path, "size": path.stat().st_size, "sha256": sha(path), "header_sha256": hashlib.sha256(raw).hexdigest(), "header_bytes": len(raw)}
            self.headers[name] = json.loads(raw[8:].decode("utf-8"), object_pairs_hook=unique)
        for tensor, shard in self.weight_map.items():
            if tensor not in self.headers[shard]: raise ReferenceError(f"index tensor absent from header: {tensor}")

    def pin(self):
        config = self.root / "config.json"; index = self.root / "model.safetensors.index.json"
        return {"config": {"path": str(config), "size": config.stat().st_size, "sha256": sha(config)}, "index": {"size": index.stat().st_size, "sha256": sha(index)}, "shards": {name: {k: value[k] for k in ("size", "sha256", "header_sha256", "header_bytes")} for name, value in self.files.items()}}

    def raw(self, name, torch):
        if name not in self.weight_map: raise ReferenceError(f"tensor not indexed: {name}")
        shard = self.weight_map[name]; entry = self.headers[shard][name]; start, end = entry.get("data_offsets", [None, None]); info = self.files[shard]
        if entry.get("dtype") not in {"BF16", "F32"}: raise ReferenceError(f"unsupported dtype for {name}: {entry.get('dtype')}")
        shape = entry.get("shape")
        if not isinstance(shape, list) or not all(type(x) is int and x >= 0 for x in shape): raise ReferenceError(f"invalid shape for {name}")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start: raise ReferenceError(f"invalid offsets for {name}")
        width = 2 if entry["dtype"] == "BF16" else 4; count = 1
        for dim in shape: count *= dim
        if end - start != count * width or info["header_bytes"] + end > info["size"]: raise ReferenceError(f"tensor bounds/length mismatch for {name}")
        with info["path"].open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            payload = bytes(mm[info["header_bytes"] + start:info["header_bytes"] + end])
        dtype = torch.bfloat16 if entry["dtype"] == "BF16" else torch.float32
        return torch.frombuffer(payload, dtype=dtype).clone().to(torch.float32).reshape(tuple(shape))

def load_original(source):
    package = types.ModuleType("ring_reference"); package.__path__ = [str(source.parent)]; sys.modules[package.__name__] = package
    config_source = source.parent / "configuration_bailing_moe_v2.py"
    if config_source.exists():
        spec = importlib.util.spec_from_file_location("ring_reference.configuration_bailing_moe_v2", config_source); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location("ring_reference.modeling_bailing_moe_v2", source); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source)); exec(compile(tree, str(source), "exec"), mod.__dict__); return mod

def run(args):
    try: import torch
    except ImportError as exc: raise ReferenceError("CPU torch is required; installation is not attempted") from exc
    torch.set_num_threads(args.threads); root = Path(args.checkpoint); source = Path(args.source); config_data = read_json(root / "config.json"); index = SafeTensorIndex(root); module = load_original(source)
    config = module.BailingMoeV2Config.from_pretrained(str(root)); config._attn_implementation = "eager"; tokens = [int(x) for x in args.tokens.split(",") if x]
    if not tokens or any(x < 0 or x >= config.vocab_size for x in tokens): raise ReferenceError("token outside vocabulary")
    ids = torch.tensor([tokens], dtype=torch.long); hidden = index.raw("model.word_embeddings.weight", torch)[ids]; positions = torch.arange(len(tokens), dtype=torch.long).unsqueeze(0); causal = torch.triu(torch.full((1, 1, len(tokens), len(tokens)), float("-inf")), diagonal=1); rotary = module.BailingMoeV2RotaryEmbedding(config); pos_emb = rotary(hidden, positions); layers = []; hidden_outputs = []; router_meta = []
    with torch.no_grad():
        for layer_no in range(config.num_hidden_layers):
            with torch.device("meta"): layer = module.BailingMoeV2DecoderLayer(config, layer_no)
            layer = layer.to_empty(device="cpu"); prefix = f"model.layers.{layer_no}."; state = {key[len(prefix):]: index.raw(key, torch) for key in index.weight_map if key.startswith(prefix)}; missing, extra = layer.load_state_dict(state, strict=False, assign=True); layer.eval()
            if missing or extra: raise ReferenceError(f"layer {layer_no} state mismatch: missing={missing} unexpected={extra}")
            captured = {}
            if hasattr(layer.mlp, "gate"):
                original = layer.mlp.gate.forward
                def capture(x, _original=original):
                    value = _original(x); captured["ids"], captured["weights"] = value[0].detach().cpu(), value[1].detach().cpu(); return value
                layer.mlp.gate.forward = capture
            result = layer(hidden, attention_mask=causal, position_ids=positions, position_embeddings=pos_emb, output_router_logits=False)[0]; hidden = result
            hidden_outputs.append(result.detach().cpu().float()); router_meta.append({"layer": layer_no, "router_ids": bool("ids" in captured), "tokens": len(tokens), "top_k": int(captured["ids"].shape[-1]) if "ids" in captured else 0}); layers.append({"layer": layer_no, "router_ids": captured.get("ids", torch.empty(0, dtype=torch.long)).tolist(), "router_weights": captured.get("weights", torch.empty(0)).tolist()}); del layer, state
    norm = module.BailingMoeV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps); norm.load_state_dict({"weight": index.raw("model.norm.weight", torch)}); hidden = norm(hidden); logits = torch.matmul(hidden.float(), index.raw("lm_head.weight", torch).transpose(0, 1).float())
    output = Path(args.output)
    if output.exists(): raise ReferenceError(f"refusing existing output: {output}")
    output.mkdir(parents=True); hidden.detach().cpu().float().numpy().tofile(output / "final_hidden.fp32"); logits.detach().cpu().float().numpy().tofile(output / "logits.fp32"); torch.stack(hidden_outputs).numpy().tofile(output / "hidden_layers.fp32"); (output / "layers.json").write_text(json.dumps(layers), encoding="utf-8"); (output / "router_meta.json").write_text(json.dumps(router_meta, indent=2), encoding="utf-8")
    report = {"status": "executed", "reference": "original Ring checkpoint implementation; CPU FP32 widened BF16", "hidden_layers_shape": [len(hidden_outputs), len(tokens), config.hidden_size], "tokens": tokens, "threads": args.threads, "source": {"path": str(source), "sha256": sha(source)}, "checkpoint": index.pin(), "config": config_data, "outputs": {name: {"bytes": (output / name).stat().st_size, "sha256": sha(output / name)} for name in ("final_hidden.fp32", "logits.fp32", "hidden_layers.fp32", "layers.json", "router_meta.json")}}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"); return report

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--source", required=True, type=Path); parser.add_argument("--tokens", required=True); parser.add_argument("--output", required=True, type=Path); parser.add_argument("--threads", default=4, type=int); args = parser.parse_args()
    if args.threads < 1: parser.error("threads must be positive")
    try: print(json.dumps(run(args), indent=2, sort_keys=True))
    except (ReferenceError, OSError, ValueError) as exc: parser.error(str(exc))

if __name__ == "__main__": main()
