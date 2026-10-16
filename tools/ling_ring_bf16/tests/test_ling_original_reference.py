from pathlib import Path
import importlib.util, json, os, struct, subprocess, sys, types
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[3]
CK = Path(os.environ.get("LING_CHECKPOINT", "/work/AI/checkpoints/Ling-3.0-tiny"))
ENV = Path(os.environ.get("LING_REFERENCE_PYTHON", sys.executable))
RUNNER = ROOT / "tools/ling_ring_bf16/ling_original_reference.py"
SOURCE = CK / "modeling_bailing_moe_v3.py"
def require_checkpoint():
    if not CK.is_dir() or not SOURCE.is_file():
        pytest.skip(f"Ling checkpoint fixture absent: {CK}")
def load_runner():
    spec = importlib.util.spec_from_file_location("ling_original_reference_test", RUNNER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def test_source_import_and_meta_model():
    require_checkpoint(); mod = load_runner()
    try:
        model = mod.load_source(SOURCE); cfg = model.BailingMoeV3Config.from_pretrained(str(CK))
        import torch
        with torch.device("meta"): model.BailingMoeV3DecoderLayer(cfg, 7)
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"Ling reference dependencies unavailable: {exc}")
    assert cfg.num_hidden_layers == 24; assert cfg.hidden_size == 1536
def test_index_pins_checkpoint_without_tensor_materialization():
    require_checkpoint(); index = load_runner().SafeTensorIndex(CK)
    assert len(index.files) == 32
    assert "model.embed_tokens.weight" in index.weight_map or "model.word_embeddings.weight" in index.weight_map
    assert len(index.pin()["shards"]) == 32
def test_invalid_token_rejected_without_output(tmp_path):
    require_checkpoint(); output = tmp_path / "invalid"
    result = subprocess.run([str(ENV), str(RUNNER), "--checkpoint", str(CK), "--source", str(SOURCE), "--tokens", "157184", "--output", str(output), "--device", "cpu"], capture_output=True, text=True)
    assert result.returncode != 0; assert not output.exists()
def test_report_writer_and_comparator_contract(tmp_path):
    mod = load_runner(); cmp_path = ROOT / "tools/ling_ring_bf16/compare_ling_reference.py"
    spec = importlib.util.spec_from_file_location("ling_compare_contract", cmp_path); cmp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cmp)
    out = tmp_path / "reference"; out.mkdir(); tokens = 2
    (out / "final_hidden.fp32").write_bytes(np.zeros(tokens * 1536, dtype=np.float32).tobytes())
    (out / "logits.fp32").write_bytes(np.zeros(tokens * 157184, dtype=np.float32).tobytes())
    (out / "hidden_layers.fp32").write_bytes(np.zeros(24 * tokens * 1536, dtype=np.float32).tobytes())
    rows = [{"layer": 0, "router_ids": [], "router_weights": []}]
    rows += [{"layer": layer, "router_ids": [[i for i in range(8)] for _ in range(tokens)], "router_weights": [[1.0 / 8] * 8 for _ in range(tokens)]} for layer in range(1, 24)]
    (out / "layers.json").write_text(json.dumps(rows)); (out / "router_meta.json").write_text("[]")
    class Args:
        source = str(tmp_path / "modeling.py"); checkpoint = str(tmp_path); device = "cpu"; logits_mode = "all"
    (tmp_path / "modeling.py").write_text("source")
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 24, "hidden_size": 1536, "vocab_size": 157184, "num_experts": 128, "num_experts_per_tok": 8}))
    srcpins = {"modeling": {"sha256": "a" * 64}, "configuration": {"sha256": "b" * 64}}
    fake_torch = types.SimpleNamespace(__version__="test", version=types.SimpleNamespace(cuda=None), backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)), cudnn=types.SimpleNamespace(allow_tf32=False)))
    report = mod.build_report(Args(), types.SimpleNamespace(num_hidden_layers=24, hidden_size=1536, vocab_size=157184), [1, 2], np.zeros((1, tokens, 1536), dtype=np.float32), np.zeros((1, tokens, 157184), dtype=np.float32), srcpins, {"config": {"sha256": "c" * 64}}, fake_torch, out, 0.0)
    mod.write_report(report, out); written = json.loads((out / "report.json").read_text())
    assert written["hidden_layers_shape"] == [24, tokens, 1536]; assert written["logits_shape"] == [1, tokens, 157184]; assert written["logits_tokens"] == tokens
    cmp._reference_identity(written); cmp.verify_reference_hashes(out, written); cmp._reference_source(written)
    assert len(cmp.routes_reference(out / "layers.json", tokens)) == 23 * tokens


def test_cached_mode_metadata_and_cli_contract(tmp_path):
    mod = load_runner()
    source_text = Path(RUNNER).read_text()
    assert "DynamicCache" in source_text
    assert "finals.append(final.detach()" in source_text
    assert "finals.append(hidden.detach()" not in source_text
    help_result = __import__("subprocess").run([str(ENV), str(RUNNER), "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "--decoding-mode" in help_result.stdout
    class Args:
        source = str(tmp_path / "modeling.py")
        checkpoint = str(tmp_path)
        device = "cuda"
        logits_mode = "all"
    (tmp_path / "modeling.py").write_text("source")
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 24, "hidden_size": 1536, "vocab_size": 157184, "num_experts": 128, "num_experts_per_tok": 8}))
    fake_torch = types.SimpleNamespace(__version__="test", version=types.SimpleNamespace(cuda=None), backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)), cudnn=types.SimpleNamespace(allow_tf32=False)))
    report = mod.build_report(Args(), types.SimpleNamespace(num_hidden_layers=24, hidden_size=1536, vocab_size=157184), [1], np.zeros((1, 1, 1536), dtype=np.float32), np.zeros((1, 1, 157184), dtype=np.float32), {"modeling": {"sha256": "a" * 64}, "configuration": {"sha256": "b" * 64}}, {"config": {"sha256": "c" * 64}}, fake_torch, tmp_path, 0.0, "cached", {"cache_type": "DynamicCache", "tokenwise": True})
    assert report["decoding_mode"] == "cached"
    assert report["cache_metadata"] == {"cache_type": "DynamicCache", "tokenwise": True}


def test_cached_route_probe_parser_is_strict():
    mod = load_runner()
    assert mod._parse_route_probe("337,19", 358, 24) == (337, 19)
    for text in ("337", "337,19,1", "abc,19", "358,19", "337,0", "337,24"):
        with pytest.raises(mod.ReferenceError):
            mod._parse_route_probe(text, 358, 24)


def test_cached_trace_state_control():
    mod = load_runner()
    rows = [[np.zeros(3, dtype=np.float32)] for _ in range(2)]
    routes = [{"layer": 0, "router_ids": [], "router_weights": []}, {"layer": 1, "router_ids": [[0] * 8], "router_weights": [[0.125] * 8]}]
    assert mod.validate_cached_trace(rows, routes, 1, 2)
    with pytest.raises(mod.ReferenceError, match="token count"):
        mod.validate_cached_trace([rows[0], []], routes, 1, 2)


def _write_safetensor_fixture(tmp_path, entry, raw_payload=b"\x00\x00\x00\x00"):
    root = tmp_path / "checkpoint"; root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model.safetensors"}}))
    header = json.dumps({"x": entry}, separators=(",", ":")).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + raw_payload)
    return root


def test_safetensor_duplicate_json_key_rejected(tmp_path):
    mod = load_runner()
    path = tmp_path / "index.json"
    path.write_text('{"weight_map":{"x":"model.safetensors"},"weight_map":{"y":"model.safetensors"}}')
    with pytest.raises(mod.ReferenceError, match="duplicate JSON key"):
        mod.readj(path)


@pytest.mark.parametrize("entry,pattern", [
    ({"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}, "unsupported dtype"),
    ({"dtype": "BF16", "shape": [-1], "data_offsets": [0, 2]}, "invalid shape"),
    ({"dtype": "BF16", "shape": [2], "data_offsets": [4, 2]}, "invalid offsets"),
    ({"dtype": "BF16", "shape": [2], "data_offsets": [0, 2]}, "bounds/length"),
])
def test_safetensor_header_validation_rejects_invalid_metadata(tmp_path, entry, pattern):
    mod = load_runner(); root = _write_safetensor_fixture(tmp_path, entry)
    index = mod.SafeTensorIndex(root)
    with pytest.raises(mod.ReferenceError, match=pattern):
        index.raw("x", None, "cpu")


def test_safetensor_truncated_header_rejected(tmp_path):
    mod = load_runner()
    root = tmp_path / "checkpoint"; root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model.safetensors"}}))
    (root / "model.safetensors").write_bytes(b"\x08\x00\x00\x00")
    mod = load_runner()
    with pytest.raises(mod.ReferenceError, match="truncated shard"):
        mod.SafeTensorIndex(root)
