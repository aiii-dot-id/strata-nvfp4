from pathlib import Path
import importlib.util, json, os, subprocess, sys, types
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
