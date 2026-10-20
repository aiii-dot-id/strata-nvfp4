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


def test_ling_source_closure_matches_linked_ring_units(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("modeling_bailing_moe_v3.py", "configuration_bailing_moe_v3.py"):
        (checkpoint / name).write_text(name)
    repo = tmp_path / "repo"
    required = [
        "include/strata/ling/mla.hpp", "include/strata/ring/generation.hpp",
        "include/strata/ring/expert_cache.hpp", "include/strata/ring/numerics.hpp",
        "include/strata/ring/routing.hpp", "include/strata/ring/tensor_store.hpp",
        "include/strata/ring/expert_cache.hpp", "include/strata/ring/numerics.hpp",
        "include/strata/ring/routing.hpp", "include/strata/ring/tensor_store.hpp",
        "src/ling/engine.cu", "src/ling/kda.cpp",
        "src/ring/routing.cpp", "src/ring/expert_cache.cpp", "src/ring/tensor_store.cpp",
        "include/strata/kernels/bf16_gemv.hpp", "include/strata/kernels/bf16_bits.hpp",
        "include/strata/kernels/pdl.hpp", "src/kernels/cuda/native_bf16.cu",
        "src/kernels/cuda/s26_tsum.cuh", "tools/ling_ring_bf16/prepare_ling_manifest.py",
        "tools/ling_ring_bf16/ling/CMakeLists.txt", "tools/ling_ring_bf16/run_ling_verified.py",
        "tools/ling_ring_bf16/ling/ling_torch_ops.hpp", "tools/ling_ring_bf16/ling/ling_torch_ops.cpp",
        "tools/ling_ring_bf16/generate_ling.py", "tools/ling_ring_bf16/verified_hash.py",
    ]
    for name in required:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    unrelated = repo / "src/ring/engine.cu"
    unrelated.write_text("unrelated")
    _, sources = v._source_files(checkpoint, repo)
    assert "src/ring/routing.cpp" in sources
    assert "src/ring/expert_cache.cpp" in sources
    assert "src/ring/tensor_store.cpp" in sources
    assert "src/ring/engine.cu" not in sources


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
(out/"execution.json").write_text(json.dumps({{"status":"executed","model":model,"activation_dtype":"FP32","quantized":False,"layers":layers,"tokens":1,"token_ids":token_ids,"logits_mode":"all","logits_tokens":1,"checkpoint_id":sys.argv[6],"rope_frequency":sys.argv[sys.argv.index("--rope-frequency")+1],"rms_mode":sys.argv[sys.argv.index("--rms-mode")+1],"matmul_backend":sys.argv[sys.argv.index("--matmul")+1],"matmul_fp32_weight_scratch_bytes":(157184*1536*4 if sys.argv[sys.argv.index("--matmul")+1]=="fp32-cublas" else 0),"workspace_bytes":157184*1536*2+4608*4+157184*4+(157184*1536*4 if sys.argv[sys.argv.index("--matmul")+1]=="fp32-cublas" else 0)}}))
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


def test_original_rope_frequency_is_bound_in_argv_and_execution(tmp_path, monkeypatch):
    ck, manifest, exe = fixture(tmp_path, monkeypatch)
    receipt = v.run_verified(ck, manifest, exe, "1", tmp_path / "out", 5,
                             rope_frequency="original")
    assert receipt["execution"]["rope_frequency"] == "original"
    assert receipt["argv"][-8:] == ["--rope-frequency", "original", "--rms-mode", "native", "--matmul", "bf16-gemv", "--pointwise", "native"]


def test_invalid_rope_frequency_fails_closed(tmp_path, monkeypatch):
    ck, manifest, exe = fixture(tmp_path, monkeypatch)
    with pytest.raises(v.VerificationError, match="rope_frequency"):
        v.run_verified(ck, manifest, exe, "1", tmp_path / "out", 5,
                       rope_frequency="other")


def test_torch_cuda_rms_mode_is_bound_in_argv_and_execution(tmp_path, monkeypatch):
    ck, manifest, exe = fixture(tmp_path, monkeypatch)
    receipt = v.run_verified(ck, manifest, exe, "1", tmp_path / "out", 5,
                             rms_mode="torch-cuda-order")
    assert receipt["execution"]["rms_mode"] == "torch-cuda-order"
    assert receipt["argv"][-6:] == ["--rms-mode", "torch-cuda-order", "--matmul", "bf16-gemv", "--pointwise", "native"]


def test_invalid_rms_mode_fails_closed(tmp_path, monkeypatch):
    ck, manifest, exe = fixture(tmp_path, monkeypatch)
    with pytest.raises(v.VerificationError, match="rms_mode"):
        v.run_verified(ck, manifest, exe, "1", tmp_path / "out", 5,
                       rms_mode="other")

def test_pointwise_receipt_binding_rejects_cross_mode_and_missing_optin(tmp_path, monkeypatch):
    ck, manifest, exe = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(v, "_source_files", lambda checkpoint, repo, *args: ({}, {}))
    original = exe.read_text()
    exe.write_text(original.replace('"matmul_backend":sys.argv[sys.argv.index("--matmul")+1]', '"matmul_backend":sys.argv[sys.argv.index("--matmul")+1],"pointwise_mode":"torch-cuda"'))
    with pytest.raises(v.VerificationError, match="pointwise_mode"):
        v.run_verified(ck, manifest, exe, "1", tmp_path / "native", 5)
    exe.write_text(original)
    with pytest.raises(v.VerificationError, match="pointwise_mode"):
        v.run_verified(ck, manifest, exe, "1", tmp_path / "optin", 5, pointwise="torch-cuda")


def test_route_probe_is_forwarded_and_bound(tmp_path, monkeypatch):
    ck, manifest, exe = fixture(tmp_path, monkeypatch)
    original = exe.read_text()
    marker = '(out/"routes.tsv").write_text("\\n".join(rows)+"\\n")'
    probe = '''if "--route-probe" in sys.argv:
    (out/"route-probe.json").write_text(json.dumps({"position":3,"layer":2,
      "input_hidden":[0.0]*1536,"post_attention_hidden":[0.0]*1536,
      "gate_input":[0.0]*1536,"router_logits":[0.0]*128,
      "sigmoid_scores":[0.5]*128,"expert_bias":[0.0]*128,
      "bias_corrected_scores":[0.5]*128,"selected_expert_ids":list(range(8)),
      "selected_weights":[0.1]*8}))
'''
    exe.write_text(original.replace(marker, marker + "\n" + probe))
    receipt = v.run_verified(ck, manifest, exe, "1", tmp_path / "out", 5, route_probe="3,2")
    assert receipt["route_probe"] == {"position": 3, "layer": 2, "output": "route-probe.json"}
    assert "route-probe.json" in receipt["native_outputs"]


@pytest.mark.parametrize("probe", ["3", "3,2,1", "x,2", "1024,2", "3,0", "3,24"])
def test_route_probe_input_fails_closed(tmp_path, monkeypatch, probe):
    ck, manifest, exe = fixture(tmp_path, monkeypatch)
    with pytest.raises(v.VerificationError, match="route probe"):
        v.run_verified(ck, manifest, exe, "1", tmp_path / "out", 5, route_probe=probe)


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


def test_workspace_receipt_rejects_underreported_base(tmp_path, monkeypatch):
    ck, manifest, exe = fixture(tmp_path, monkeypatch)
    original = exe.read_text()
    exe.write_text(original.replace('"workspace_bytes":157184*1536*2+4608*4+157184*4+', '"workspace_bytes":0+'))
    with pytest.raises(v.VerificationError, match="workspace"):
        v.run_verified(ck, manifest, exe, "1", tmp_path / "out", 5)

def test_cublas_dynamic_closure_positive_and_canonical(tmp_path, monkeypatch):
    exe = tmp_path / "engine"; exe.write_bytes(b"exe")
    cublas = tmp_path / "libcublas.so.13"; cublas.write_bytes(b"cublas")
    libc = tmp_path / "libc.so.6"; libc.write_bytes(b"libc")
    outputs = {
        str(exe.resolve()): f"{cublas} => {cublas} (0x0)\n",
        str(cublas.resolve()): f"{libc} => {libc} (0x0)\n",
        str(libc.resolve()): "\n",
    }
    monkeypatch.setattr(v.subprocess, "check_output", lambda argv, **kw: outputs[str(Path(argv[-1]).resolve())])
    got = v._dynamic_library_closure(exe, True)
    assert "external:" + str(cublas.resolve()) in got
    assert got["external:" + str(cublas.resolve())]["sha256"] == v._sha(cublas)


def test_cublas_dynamic_closure_rejects_missing_library(tmp_path, monkeypatch):
    exe = tmp_path / "engine"; exe.write_bytes(b"exe")
    monkeypatch.setattr(v.subprocess, "check_output", lambda argv, **kw: "libcublas.so.13 => not found\n")
    with pytest.raises(v.VerificationError, match="not found"):
        v._dynamic_library_closure(exe, True)


def test_cublas_dynamic_closure_detects_after_run_drift(tmp_path, monkeypatch):
    exe = tmp_path / "engine"; exe.write_bytes(b"exe")
    cublas = tmp_path / "libcublas.so.13"; cublas.write_bytes(b"one")
    state = {"value": b"one"}
    def ldd(argv, **kw):
        cublas.write_bytes(state["value"])
        return f"{cublas} => {cublas} (0x0)\n"
    monkeypatch.setattr(v.subprocess, "check_output", ldd)
    before = v._dynamic_library_closure(exe, True)
    state["value"] = b"two"
    after = v._dynamic_library_closure(exe, True)
    assert before != after

def test_ling_source_closure_excludes_unlinked_ring_headers(tmp_path):
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir()
    for name in ("modeling_bailing_moe_v3.py", "configuration_bailing_moe_v3.py"): (checkpoint / name).write_text(name)
    repo = tmp_path / "repo"
    required = ["include/strata/ling/x.hpp", "src/ling/engine.cu", "include/strata/kernels/bf16_gemv.hpp", "include/strata/kernels/bf16_bits.hpp", "include/strata/kernels/pdl.hpp", "src/kernels/cuda/native_bf16.cu", "src/kernels/cuda/s26_tsum.cuh", "tools/ling_ring_bf16/prepare_ling_manifest.py", "tools/ling_ring_bf16/ling/CMakeLists.txt", "tools/ling_ring_bf16/run_ling_verified.py", "tools/ling_ring_bf16/generate_ling.py", "tools/ling_ring_bf16/verified_hash.py"]
    ring = ["expert_cache.hpp", "generation.hpp", "numerics.hpp", "routing.hpp", "tensor_store.hpp"]
    for name in required + ["include/strata/ring/" + x for x in ring] + ["src/ring/routing.cpp", "src/ring/expert_cache.cpp", "src/ring/tensor_store.cpp"]:
        q=repo/name; q.parent.mkdir(parents=True,exist_ok=True); q.write_text(name)
    (repo / "include/strata/ring/attention.hpp").write_text("unlinked")
    _, sources = v._source_files(checkpoint, repo)
    assert "include/strata/ring/attention.hpp" not in sources
    assert all("include/strata/ring/" + x in sources for x in ring)
