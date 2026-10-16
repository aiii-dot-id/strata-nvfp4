import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from prepare_ring_manifest import EXPECTED, prepare
import run_ring_verified as verified


def checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    config = dict(EXPECTED)
    config["architectures"] = ["BailingMoeV2ForCausalLM"]
    (root / "config.json").write_text(json.dumps(config))
    header = json.dumps({"x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model.safetensors"}}))
    (root / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"abcd")
    (root / "modeling_bailing_moe_v2.py").write_text("model source")
    (root / "configuration_bailing_moe_v2.py").write_text("config source")
    out = tmp_path / "admission"
    prepare(root, out)
    return root, out / "manifest.tsv"


def fake_engine(monkeypatch, mutate=None):
    def run(argv, **kwargs):
        output = Path(argv[4])
        output.mkdir()
        (output / "routes.tsv").write_text("\n".join(f"0\t{layer}\t" + "\t".join(f"{x}:0.1" for x in range(8)) for layer in range(1, 20)) + "\n")
        for layer in range(20):
            (output / f"layer-{layer}.f32").write_bytes(b"\0\0\0\0")
        (output / "final-hidden.f32").write_bytes(b"\0\0\0\0")
        (output / "logits.f32").write_bytes(b"\0\0\0\0")
        (output / "execution.json").write_text(json.dumps({
            "status": "executed", "model": "Ring-mini-2.0", "activation_dtype": "FP32",
            "quantized": False, "tokens": 1, "token_ids": [1], "layers": 20,
            "checkpoint_id": argv[6],
        }))
        if mutate:
            mutate(Path(argv[1]))
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(verified.subprocess, "run", run)


def test_manifest_mismatch_refused(tmp_path):
    root, manifest = checkpoint(tmp_path)
    bad = tmp_path / "bad.tsv"
    bad.write_bytes(manifest.read_bytes() + b"x")
    with pytest.raises(verified.VerificationError, match="manifest differs"):
        verified.run_verified(root, bad, Path("/bin/true"), "1", tmp_path / "out", 6)


def test_source_drift_refused(tmp_path, monkeypatch):
    root, manifest = checkpoint(tmp_path)
    fake_engine(monkeypatch, lambda path: (path / "modeling_bailing_moe_v2.py").write_text("changed"))
    with pytest.raises(verified.VerificationError, match="drifted"):
        verified.run_verified(root, manifest, Path("/bin/true"), "1", tmp_path / "out", 6)


def test_verified_receipt_binds_manifest_and_outputs(tmp_path, monkeypatch):
    root, manifest = checkpoint(tmp_path)
    fake_engine(monkeypatch)
    receipt = verified.run_verified(root, manifest, Path("/bin/true"), "1", tmp_path / "out", 6)
    assert receipt["status"] == "verified"
    assert receipt["checkpoint_id"] == receipt["manifest"]["sha256"]
    assert set(receipt["native_outputs"]) == set(verified.EXPECTED_NATIVE)
