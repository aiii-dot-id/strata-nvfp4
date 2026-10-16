import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest
import run_ling_verified as v


def fixture(tmp_path, monkeypatch, variant="good"):
    ck = tmp_path / "checkpoint"; ck.mkdir()
    shards = [f"model-{i:05d}-of-00032.safetensors" for i in range(1, 33)]
    (ck / "config.json").write_text("{}")
    (ck / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"x{i}": s for i, s in enumerate(shards)}}))
    for s in shards: (ck / s).write_bytes(bytes([len(s)]))
    for n in ("modeling_bailing_moe_v3.py", "configuration_bailing_moe_v3.py"): (ck / n).write_text(n)
    manifest = tmp_path / "manifest.tsv"; manifest.write_bytes(b"manifest\n")
    def prepare(_ck, out):
        Path(out).mkdir()
        (Path(out) / "manifest.tsv").write_bytes(manifest.read_bytes())
        return {}
    monkeypatch.setattr(v, "prepare", prepare)
    def sources(checkpoint, repo):
        return ({n: v._pin(checkpoint / n, n) for n in ("modeling_bailing_moe_v3.py", "configuration_bailing_moe_v3.py")},
                {"src/ling/engine.cu": {"sha256": "1" * 64, "size": 1}})
    monkeypatch.setattr(v, "_source_files", sources)
    exe = tmp_path / "engine.py"
    exe.write_text(f'''#!/usr/bin/env python3
import json, pathlib, struct, sys, time
out=pathlib.Path(sys.argv[4]); out.mkdir()
if {variant!r} == "sleep":
    pathlib.Path(sys.argv[7]).write_text(str(__import__("os").getpid())); time.sleep(60)
def f(path, n): path.write_bytes(struct.pack("<"+"f"*n, *([0.0]*n)))
for i in range(24): f(out/f"layer-{{i}}.f32", {v.HIDDEN})
f(out/"final-hidden.f32", {v.HIDDEN}); f(out/"logits.f32", {v.VOCAB})
rows=[]
for layer in {list(v.EXPECTED_ROUTE_LAYERS)!r}: rows.append("0\\t"+str(layer)+"\\t"+"\\t".join(str(i)+":0.1" for i in range(8)))
(out/"routes.tsv").write_text("\\n".join(rows)+"\\n")
model = "Wrong" if {variant!r} == "model" else "Ling-3.0-tiny"
layers = 23 if {variant!r} == "layers" else 24
token_ids = [1]
(out/"execution.json").write_text(json.dumps({{"status":"executed","model":model,"activation_dtype":"FP32","quantized":False,"layers":layers,"tokens":1,"token_ids":token_ids,"logits_mode":"all","logits_tokens":1,"checkpoint_id":sys.argv[6]}}))
''')
    exe.chmod(0o755)
    return ck, manifest, exe


def test_verified_receipt_and_exact_outputs(tmp_path, monkeypatch):
    ck, manifest, exe = fixture(tmp_path, monkeypatch)
    receipt = v.run_verified(ck, manifest, exe, "1", tmp_path / "out", 5)
    assert receipt["status"] == "verified"
    assert receipt["checkpoint_id"] == v._sha(manifest)
    assert len(receipt["checkpoint"]["shards"]) == 32
    assert (tmp_path / "out" / "provenance.json").is_file()


@pytest.mark.parametrize("variant, message", [("model", "model"), ("layers", "layers")])
def test_execution_identity_controls(tmp_path, monkeypatch, variant, message):
    ck, manifest, exe = fixture(tmp_path, monkeypatch, variant)
    with pytest.raises(v.VerificationError, match=message):
        v.run_verified(ck, manifest, exe, "1", tmp_path / "out", 5)
    assert not (tmp_path / "out" / "provenance.json").exists()


def test_manifest_and_checkpoint_drift_refused(tmp_path, monkeypatch):
    ck, manifest, exe = fixture(tmp_path, monkeypatch)
    bad = tmp_path / "bad.tsv"; bad.write_bytes(b"different\n")
    with pytest.raises(v.VerificationError, match="manifest differs"):
        v.run_verified(ck, bad, exe, "1", tmp_path / "bad", 5)
    original = exe.read_text()
    exe.write_text(original.replace('out=pathlib.Path(sys.argv[4]); out.mkdir()', 'out=pathlib.Path(sys.argv[4]); out.mkdir(); pathlib.Path(sys.argv[1], "model-00001-of-00032.safetensors").write_bytes(b"drift")'))
    with pytest.raises(v.VerificationError, match="drifted"):
        v.run_verified(ck, manifest, exe, "1", tmp_path / "drift", 5)


def test_route_bounds_and_truncated_float_refused(tmp_path, monkeypatch):
    ck, manifest, exe = fixture(tmp_path, monkeypatch)
    original = exe.read_text()
    exe.write_text(original.replace('str(i)+":0.1" for i in range(8)', 'str(128)+":0.1" if i == 0 else str(i)+":0.1" for i in range(8)'))
    with pytest.raises(v.VerificationError, match="expert ID"):
        v.run_verified(ck, manifest, exe, "1", tmp_path / "route", 5)
    exe.write_text(original.replace('f(out/"final-hidden.f32", 1536)', '(out/"final-hidden.f32").write_bytes(b"\\0\\0")'))
    with pytest.raises(v.VerificationError, match="size mismatch"):
        v.run_verified(ck, manifest, exe, "1", tmp_path / "short", 5)


def test_owned_child_cancellation_writes_no_receipt(tmp_path, monkeypatch):
    ck, manifest, exe = fixture(tmp_path, monkeypatch, "sleep")
    out = tmp_path / "out"
    started = tmp_path / "started"
    # The driver patches only structural admission/source enumeration, then
    # invokes the real wrapper and its real owned-child subprocess path.
    driver = tmp_path / "driver.py"
    driver.write_text("""import pathlib, run_ling_verified as v
ck=pathlib.Path(%r); manifest=pathlib.Path(%r); exe=pathlib.Path(%r); out=pathlib.Path(%r)
def prepare(_ck, target):
    target=pathlib.Path(target); target.mkdir(); (target/\"manifest.tsv\").write_bytes(manifest.read_bytes())
v.prepare=prepare
v._source_files=lambda checkpoint, repo: ({}, {})
v.run_verified(ck, manifest, exe, \"1\", out, 5, timeout=30)
""" % (str(ck), str(manifest), str(exe), str(out)))
    exe.write_text(exe.read_text().replace('pathlib.Path(sys.argv[7]).write_text', 'pathlib.Path(%r).write_text' % str(started)))
    env = dict(os.environ, PYTHONPATH=str(Path(v.__file__).resolve().parent))
    proc = subprocess.Popen([sys.executable, str(driver)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 10
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert started.exists()
    child_pid = int(started.read_text())
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) != 0
    for _ in range(100):
        try: os.kill(child_pid, 0)
        except ProcessLookupError: break
        time.sleep(0.02)
    else: pytest.fail("owned Ling child survived wrapper cancellation")
    assert not (out / "provenance.json").exists()

@pytest.mark.parametrize("text", [" 1", "+1", "1 ", "1,,2", "1,-2"])
def test_token_grammar_matches_native_unsigned_decimal(tmp_path, monkeypatch, text):
    ck, manifest, exe = fixture(tmp_path, monkeypatch)
    with pytest.raises(v.VerificationError, match="unsigned decimal"):
        v.run_verified(ck, manifest, exe, text, tmp_path / "out", 5)
