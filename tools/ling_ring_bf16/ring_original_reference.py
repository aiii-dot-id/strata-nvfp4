#!/usr/bin/env python3
"""CPU FP32 reference runner for the original Ring implementation."""
from __future__ import annotations
import argparse, ast, gc, hashlib, importlib.util, json, mmap, resource, struct, sys, time, types
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
        # Re-read every file so the post-run pin detects payload drift, not
        # just changes to construction-time metadata.
        config = self.root / "config.json"
        index = self.root / "model.safetensors.index.json"
        shards = {}
        for name in sorted(self.files):
            path = self.files[name]["path"]
            if not path.is_file() or path.is_symlink():
                raise ReferenceError(f"shard changed or disappeared: {name}")
            with path.open("rb") as f:
                raw_len = f.read(8)
                if len(raw_len) != 8:
                    raise ReferenceError(f"shard changed or truncated: {name}")
                header_len = struct.unpack("<Q", raw_len)[0]
                if header_len > 64 * 1024 * 1024 or header_len + 8 > path.stat().st_size:
                    raise ReferenceError(f"invalid changed header span: {name}")
                raw_header = raw_len + f.read(header_len)
            if len(raw_header) != header_len + 8:
                raise ReferenceError(f"shard header changed or truncated: {name}")
            shards[name] = {"path": str(path), "size": path.stat().st_size,
                            "sha256": sha(path),
                            "header_sha256": hashlib.sha256(raw_header).hexdigest(),
                            "header_bytes": len(raw_header)}
        return {"config": {"path": str(config), "size": config.stat().st_size, "sha256": sha(config)},
                "index": {"path": str(index), "size": index.stat().st_size, "sha256": sha(index)},
                "shards": shards}

    def raw(self, name, torch, device="cpu"):
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
        return torch.frombuffer(payload, dtype=dtype).clone().to(torch.float32).reshape(tuple(shape)).to(device)

def file_pin(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ReferenceError(f"required source file missing or unsafe: {path}")
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha(path)}

def runtime_pins(root, source):
    source = Path(source)
    config_source = source.parent / "configuration_bailing_moe_v2.py"
    return {"runner": file_pin(Path(__file__).resolve()),
            "modeling": file_pin(source),
            "configuration": file_pin(config_source),
            "config": file_pin(Path(root) / "config.json")}

def load_original(source):
    package = types.ModuleType("ring_reference"); package.__path__ = [str(source.parent)]; sys.modules[package.__name__] = package
    config_source = source.parent / "configuration_bailing_moe_v2.py"
    if config_source.exists():
        spec = importlib.util.spec_from_file_location("ring_reference.configuration_bailing_moe_v2", config_source); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location("ring_reference.modeling_bailing_moe_v2", source); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source)); exec(compile(tree, str(source), "exec"), mod.__dict__); return mod

def select_logits_hidden(hidden, logits_mode):
    if logits_mode == "all":
        return hidden
    if logits_mode == "last":
        return hidden[:, -1:, :]
    raise ReferenceError(f"unsupported logits mode: {logits_mode}")

def _parse_route_probe(text, token_count, layer_count):
    if not isinstance(text, str) or text.count(",") != 1:
        raise ReferenceError("route_probe must be POS,LAYER")
    pos_text, layer_text = text.split(",")
    if not pos_text.isdigit() or not layer_text.isdigit():
        raise ReferenceError("route_probe must be POS,LAYER")
    position, layer = int(pos_text), int(layer_text)
    if position < 0 or position >= token_count:
        raise ReferenceError("route_probe position exceeds evaluated tokens")
    if layer < 1 or layer >= layer_count:
        raise ReferenceError("route_probe layer must be a sparse layer")
    return position, layer

def decode_layer(layer, hidden, positions, position_embeddings, causal,
                 decoding_mode, cache_factory, layer_no, torch):
    """Use the original decoder and cache; keep only one layer resident.

    Cached mode evaluates the same fixed causal token trace one token at a
    time. It does not infer future tokens or replace the original attention.
    """
    if decoding_mode == "full":
        return layer(hidden, attention_mask=causal, position_ids=positions,
                     position_embeddings=position_embeddings,
                     output_router_logits=False)[0]
    if decoding_mode != "cached":
        raise ReferenceError("unsupported decoding mode")
    cache = cache_factory()
    rows = []
    for pos in range(hidden.shape[1]):
        result = layer(hidden[:, pos:pos + 1], attention_mask=None,
                       position_ids=positions[:, pos:pos + 1],
                       position_embeddings=tuple(x[:, pos:pos + 1] for x in position_embeddings),
                       past_key_value=cache, use_cache=True,
                       output_router_logits=False)
        if len(result) != 2 or result[-1] is not cache:
            raise ReferenceError("original decoder did not return its cache")
        if cache.get_seq_length(layer_no) != pos + 1:
            raise ReferenceError("original cache did not advance exactly once")
        rows.append(result[0])
    return torch.cat(rows, dim=1)

def decode_head(hidden, norm, head, logits_mode, decoding_mode, torch):
    if decoding_mode == "cached":
        normalized = torch.cat([norm(hidden[:, pos:pos + 1]) for pos in range(hidden.shape[1])], dim=1)
        selected = select_logits_hidden(normalized, logits_mode)
        logits = torch.cat([torch.matmul(selected[:, pos:pos + 1].float(), head.transpose(0, 1).float()) for pos in range(selected.shape[1])], dim=1)
        return normalized, logits
    normalized = norm(hidden)
    return normalized, torch.matmul(select_logits_hidden(normalized, logits_mode).float(), head.transpose(0, 1).float())

def run(args):
    try: import torch
    except ImportError as exc: raise ReferenceError("CPU torch is required; installation is not attempted") from exc
    if args.device == "cuda" and not torch.cuda.is_available(): raise ReferenceError("CUDA requested but unavailable in reference environment")
    if args.device == "cuda": torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device); started = time.monotonic(); output = Path(args.output)
    decoding_mode = getattr(args, "decoding_mode", "full")
    if decoding_mode not in {"full", "cached"}: raise ReferenceError("unsupported decoding mode")
    if output.exists(): raise ReferenceError(f"refusing existing output: {output}")
    if args.device == "cuda" and torch.cuda.mem_get_info()[0] < 6 * 1024**3: raise ReferenceError("insufficient free GPU memory for one FP32 decoder layer")
    torch.set_num_threads(args.threads); root = Path(args.checkpoint); source = Path(args.source); config_data = read_json(root / "config.json"); index = SafeTensorIndex(root); pins_before = index.pin(); source_pins_before = runtime_pins(root, source); module = load_original(source)
    config = module.BailingMoeV2Config.from_pretrained(str(root)); config._attn_implementation = "eager"; tokens = [int(x) for x in args.tokens.split(",") if x]
    if not tokens or any(x < 0 or x >= config.vocab_size for x in tokens): raise ReferenceError("token outside vocabulary")
    if not tokens or len(tokens) > 2048: raise ReferenceError("tokens length must be between 1 and 2048")
    route_probe = getattr(args, "route_probe", None)
    parsed_probe = _parse_route_probe(route_probe, len(tokens), config.num_hidden_layers) if route_probe is not None else None
    ids = torch.tensor([tokens], dtype=torch.long, device=device); hidden = index.raw("model.word_embeddings.weight", torch, device)[ids]; positions = torch.arange(len(tokens), dtype=torch.long, device=device).unsqueeze(0); causal = torch.triu(torch.full((1, 1, len(tokens), len(tokens)), float("-inf"), device=device), diagonal=1); rotary = module.BailingMoeV2RotaryEmbedding(config).to(device); pos_emb = rotary(hidden, positions); layers = []; hidden_outputs = []; router_meta = []
    route_probe_data = [None]
    with torch.no_grad():
        for layer_no in range(config.num_hidden_layers):
            with torch.device("meta"): layer = module.BailingMoeV2DecoderLayer(config, layer_no)
            prefix = f"model.layers.{layer_no}."; state = {key[len(prefix):]: index.raw(key, torch, device) for key in index.weight_map if key.startswith(prefix)}; missing, extra = layer.load_state_dict(state, strict=False, assign=True); layer.eval()
            if any(value.is_meta for value in list(layer.parameters()) + list(layer.buffers())): raise ReferenceError(f"layer {layer_no} retained meta tensors")
            if missing or extra: raise ReferenceError(f"layer {layer_no} state mismatch: missing={missing} unexpected={extra}")
            captured = {}; original = None; capture = None
            probe_calls = [0]
            if hasattr(layer.mlp, "gate"):
                original = layer.mlp.gate.forward
                def capture(x, _original=original, _layer_no=layer_no):
                    value = _original(x)
                    captured.setdefault("ids", []).append(value[0].detach().cpu())
                    captured.setdefault("weights", []).append(value[1].detach().cpu())
                    if parsed_probe is not None and _layer_no == parsed_probe[1]:
                        flat_x = x.reshape(-1, x.shape[-1])
                        flat_logits = value[2].reshape(-1, value[2].shape[-1])
                        if flat_x.shape[0] == len(tokens):
                            row, should_capture = parsed_probe[0], True
                        elif flat_x.shape[0] == 1:
                            should_capture = probe_calls[0] == parsed_probe[0]
                            row = 0
                            probe_calls[0] += 1
                        else:
                            raise ReferenceError("route probe row is unavailable")
                        if should_capture:
                            logits_device = flat_logits[row].detach().float()
                            scores_device = torch.sigmoid(logits_device)
                            bias_device = layer.mlp.gate.expert_bias.detach().float().reshape(-1)
                            corrected_device = scores_device + bias_device
                            logits_row = logits_device.cpu()
                            scores = scores_device.cpu()
                            bias = bias_device.cpu()
                            corrected = corrected_device.cpu()
                            route_probe_data[0] = {
                                "position": parsed_probe[0], "layer": parsed_probe[1],
                                "score_reconstruction": "torch.sigmoid(router_logits) and bias correction recomputed on the original logits device from the same gate output",
                                "gate_input": flat_x[row].detach().float().cpu().tolist(),
                                "router_logits": logits_row.tolist(),
                                "sigmoid_scores": scores.tolist(),
                                "expert_bias": bias.tolist(),
                                "bias_corrected_scores": corrected.tolist(),
                                "selected_expert_ids": value[0].reshape(-1, value[0].shape[-1])[row].detach().cpu().tolist(),
                                "selected_weights": value[1].reshape(-1, value[1].shape[-1])[row].detach().float().cpu().tolist(),
                            }
                    return value
                layer.mlp.gate.forward = capture
            try:
                result = decode_layer(layer, hidden, positions, pos_emb, causal,
                                      decoding_mode, module.DynamicCache, layer_no, torch)
            finally:
                # Restore the bound method while the layer is still alive.
                if original is not None:
                    layer.mlp.gate.forward = original
            hidden = result
            captured = {name: torch.cat(values, dim=0) for name, values in captured.items()}
            hidden_outputs.append(result.detach().cpu().float()); router_meta.append({"layer": layer_no, "router_ids": bool("ids" in captured), "tokens": len(tokens), "top_k": int(captured["ids"].shape[-1]) if "ids" in captured else 0}); layers.append({"layer": layer_no, "router_ids": captured.get("ids", torch.empty(0, dtype=torch.long)).tolist(), "router_weights": captured.get("weights", torch.empty(0)).tolist()}); del layer, state, captured, original, capture
            gc.collect()
            if args.device == "cuda": torch.cuda.empty_cache()
    with torch.no_grad():
        norm = module.BailingMoeV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device); norm.load_state_dict({"weight": index.raw("model.norm.weight", torch, device)})
        hidden, logits = decode_head(hidden, norm, index.raw("lm_head.weight", torch, device), args.logits_mode, decoding_mode, torch)
    if args.device == "cuda" and torch.cuda.max_memory_reserved() > 5 * 1024**3:
        raise ReferenceError("reference GPU peak exceeded 5 GiB safety limit")
    pins_after = index.pin()
    source_pins_after = runtime_pins(root, source)
    if pins_before != pins_after: raise ReferenceError("checkpoint pins changed during execution")
    if source_pins_before != source_pins_after: raise ReferenceError("source/config/script pins changed during execution")
    if parsed_probe is not None and route_probe_data[0] is None:
        raise ReferenceError("route probe position was not captured")
    output.mkdir(parents=True); hidden.detach().cpu().float().numpy().tofile(output / "final_hidden.fp32"); logits.detach().cpu().float().numpy().tofile(output / "logits.fp32"); torch.stack(hidden_outputs).cpu().numpy().tofile(output / "hidden_layers.fp32"); (output / "layers.json").write_text(json.dumps(layers), encoding="utf-8"); (output / "router_meta.json").write_text(json.dumps(router_meta, indent=2), encoding="utf-8")
    if route_probe_data[0] is not None:
        (output / "route-probe.json").write_text(json.dumps(route_probe_data[0], sort_keys=True) + "\n", encoding="utf-8")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != "darwin" else 1); peak_gpu = {"allocated": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0, "reserved": int(torch.cuda.max_memory_reserved()) if args.device == "cuda" else 0}; legacy_source = {"path": str(source), "size": source.stat().st_size, "sha256": source_pins_after["modeling"]["sha256"], "configuration_sha256": source_pins_after["configuration"]["sha256"]}; report = {"status": "executed", "reference": f"original Ring checkpoint implementation; {args.device.upper()} FP32 widened BF16", "device": args.device, "logits_mode": args.logits_mode, "hidden_layers_shape": [len(hidden_outputs), len(tokens), config.hidden_size], "logits_shape": list(logits.shape), "tokens": tokens, "threads": args.threads, "runtime_seconds": time.monotonic() - started, "peak_rss_bytes": peak_rss, "peak_gpu_bytes": peak_gpu, "source": legacy_source, "source_details": source_pins_after, "checkpoint": pins_after, "config": config_data, "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda, "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cuda.matmul.allow_tf32)}, "outputs": {name: {"bytes": (output / name).stat().st_size, "sha256": sha(output / name)} for name in (("final_hidden.fp32", "logits.fp32", "hidden_layers.fp32", "layers.json", "router_meta.json") + (("route-probe.json",) if route_probe_data[0] is not None else ()))}}
    report["runtime"]["tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    report["decoding_mode"] = decoding_mode
    report["cache_protocol"] = "original DynamicCache, one token at a time, one decoder layer resident" if decoding_mode == "cached" else "full causal sequence without cache"
    if parsed_probe is not None:
        report["route_probe"] = {"position": parsed_probe[0], "layer": parsed_probe[1], "output": "route-probe.json"}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"); return report

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--source", required=True, type=Path); parser.add_argument("--tokens", required=True); parser.add_argument("--output", required=True, type=Path); parser.add_argument("--threads", default=4, type=int); parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu"); parser.add_argument("--logits-mode", choices=("all", "last"), default="all")
    parser.add_argument("--decoding-mode", choices=("full", "cached"), default="full")
    parser.add_argument("--route-probe", help="capture one POS,LAYER router state")
    args = parser.parse_args()
    if args.threads < 1: parser.error("threads must be positive")
    try: print(json.dumps(run(args), indent=2, sort_keys=True))
    except (ReferenceError, OSError, ValueError) as exc: parser.error(str(exc))

if __name__ == "__main__": main()
