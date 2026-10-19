import json
import os
import signal
import shutil
import subprocess
import sys
import time
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
    def launch(argv, **kwargs):
        output = Path(argv[4])
        output.mkdir()
        matmul = argv[argv.index("--matmul") + 1]
        (output / "routes.tsv").write_text("\n".join(f"0\t{layer}\t" + "\t".join(f"{x}:0.1" for x in range(8)) for layer in range(1, 20)) + "\n")
        for layer in range(20):
            (output / f"layer-{layer}.f32").write_bytes(b"\0\0\0\0")
        (output / "final-hidden.f32").write_bytes(b"\0\0\0\0")
        (output / "logits.f32").write_bytes(b"\0\0\0\0")
        if "--route-probe" in argv:
            (output / "route-probe.json").write_text(json.dumps({
                "position": 0, "layer": 1, "gate_input": [0.0] * 2048,
                "router_logits": [0.0] * 256, "sigmoid_scores": [0.5] * 256,
                "expert_bias": [0.0] * 256, "bias_corrected_scores": [0.5] * 256,
                "selected_expert_ids": list(range(8)), "selected_weights": [0.125] * 8}))
        (output / "execution.json").write_text(json.dumps({
            "status": "executed", "model": "Ring-mini-2.0", "activation_dtype": "FP32",
            "quantized": False, "tokens": 1, "token_ids": [1], "layers": 20,
            "checkpoint_id": argv[6], "matmul_backend": matmul,
            "router_backend": argv[argv.index("--router") + 1] if "--router" in argv else "native",
            "rms_backend": argv[argv.index("--rms-mode") + 1] if "--rms-mode" in argv else "native",
            "attention_backend": argv[argv.index("--attention-backend") + 1] if "--attention-backend" in argv else "native",
            "pointwise_backend": argv[argv.index("--pointwise") + 1] if "--pointwise" in argv else "native",
            "attention_value_order": argv[argv.index("--attention-order") + 1] if "--attention-order" in argv else "native",
            "attention_score_scale": argv[argv.index("--attention-scale") + 1] if "--attention-scale" in argv else "native",
            "rope_frequency": argv[argv.index("--rope-frequency") + 1] if "--rope-frequency" in argv else "native",
        }))
        if mutate:
            mutate(Path(argv[1]))
        class Finished:
            pid = os.getpid()
            returncode = 0
            def communicate(self, timeout=None): return "", ""
            def wait(self, timeout=None): return 0
        return Finished()
    monkeypatch.setattr(verified.subprocess, "Popen", launch)


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


def test_route_probe_is_forwarded_and_bound(tmp_path, monkeypatch):
    root, manifest = checkpoint(tmp_path)
    fake_engine(monkeypatch)
    receipt = verified.run_verified(root, manifest, Path("/bin/true"), "1", tmp_path / "out", 6,
                                   route_probe="0,1")
    assert receipt["route_probe"] == {"position": 0, "layer": 1, "output": "route-probe.json"}
    assert "route-probe.json" in receipt["native_outputs"]


def test_fp32_cublas_backend_is_forwarded_and_bound(tmp_path, monkeypatch):
    root, manifest = checkpoint(tmp_path)
    fake_engine(monkeypatch)
    receipt = verified.run_verified(root, manifest, Path("/bin/true"), "1", tmp_path / "out", 6,
                                    matmul="fp32-cublas")
    assert receipt["matmul_backend"] == "fp32-cublas"
    assert receipt["execution"]["matmul_backend"] == "fp32-cublas"
    assert receipt["argv"][-2:] == ["--matmul", "fp32-cublas"]


def test_attention_profiles_are_forwarded_and_bound(tmp_path, monkeypatch):
    root, manifest = checkpoint(tmp_path)
    fake_engine(monkeypatch)
    receipt = verified.run_verified(root, manifest, Path("/bin/true"), "1", tmp_path / "out", 6,
                                    attention_order="normalized-first", attention_scale="divide")
    assert receipt["attention_value_order"] == "normalized-first"
    assert receipt["attention_score_scale"] == "divide"
    assert receipt["execution"]["attention_value_order"] == "normalized-first"
    assert receipt["execution"]["attention_score_scale"] == "divide"
    assert "--attention-order" in receipt["argv"]
    assert "--attention-scale" in receipt["argv"]


def test_original_rope_frequency_is_forwarded_and_bound(tmp_path, monkeypatch):
    root, manifest = checkpoint(tmp_path)
    fake_engine(monkeypatch)
    receipt = verified.run_verified(
        root, manifest, Path("/bin/true"), "1", tmp_path / "out", 6,
        rope_frequency="original")
    assert receipt["rope_frequency"] == "original"
    assert receipt["execution"]["rope_frequency"] == "original"
    assert receipt["argv"][-2:] == ["--rope-frequency", "original"]


def test_attention_profile_receipt_mismatch_is_rejected(tmp_path, monkeypatch):
    root, manifest = checkpoint(tmp_path)
    def mutate(_checkpoint):
        path = tmp_path / "out"
        receipt = json.loads((path / "execution.json").read_text())
        receipt["attention_value_order"] = "native"
        (path / "execution.json").write_text(json.dumps(receipt))
    fake_engine(monkeypatch, mutate)
    with pytest.raises(verified.VerificationError, match="attention arithmetic profile"):
        verified.run_verified(root, manifest, Path("/bin/true"), "1", tmp_path / "out", 6,
                              attention_order="normalized-first")


@pytest.mark.parametrize("backend", ["", "cuda", "fp32-cublas "])
def test_invalid_matmul_backend_refused(tmp_path, backend):
    root, manifest = checkpoint(tmp_path)
    with pytest.raises(verified.VerificationError, match="matmul"):
        verified.run_verified(root, manifest, Path("/bin/true"), "1", tmp_path / "out", 6,
                              matmul=backend)


@pytest.mark.parametrize("probe,match", [("0,0", "sparse layer"), ("1,20", "sparse layer"),
                                          ("1", "POS,LAYER"), ("2,1", "evaluation window")])
def test_route_probe_invalid_input_refused(tmp_path, probe, match):
    root, manifest = checkpoint(tmp_path)
    with pytest.raises(verified.VerificationError, match=match):
        verified.run_verified(root, manifest, Path("/bin/true"), "1", tmp_path / "out", 6,
                              route_probe=probe)


def test_cli_repeated_bind_file_is_recorded(tmp_path):
    root, manifest = checkpoint(tmp_path)
    first = root / "tokenizer.json"; second = root / "tokenizer_config.json"
    first.write_text("{}")
    second.write_text("{}")
    executable = tmp_path / "engine.py"
    executable.write_text("""#!/usr/bin/env python3
import json, pathlib, sys
o = pathlib.Path(sys.argv[4]); o.mkdir()
(o / \"routes.tsv\").write_text(\"\".join(f\"0\\t{layer}\\t\" + \"\\t\".join(f\"{x}:0.1\" for x in range(8)) + \"\\n\" for layer in range(1, 20)))
for layer in range(20): (o / f\"layer-{layer}.f32\").write_bytes(bytes(4))
(o / \"final-hidden.f32\").write_bytes(bytes(4)); (o / \"logits.f32\").write_bytes(bytes(4))
(o / \"execution.json\").write_text(json.dumps({\"status\":\"executed\",\"model\":\"Ring-mini-2.0\",\"activation_dtype\":\"FP32\",\"quantized\":False,\"tokens\":1,\"token_ids\":[1],\"layers\":20,\"checkpoint_id\":sys.argv[6],\"matmul_backend\":sys.argv[sys.argv.index(\"--matmul\")+1]}))
""")
    executable.chmod(0o755)
    output = tmp_path / "out"
    command = [sys.executable, "tools/ling_ring_bf16/run_ring_verified.py", "--checkpoint", str(root),
               "--manifest", str(manifest), "--executable", str(executable), "--tokens", "1",
               "--output", str(output), "--cache-mib", "6", "--bind-file", str(first),
               "--bind-file", str(second)]
    completed = subprocess.run(command, cwd=Path(__file__).parents[3], check=True, capture_output=True, text=True)
    receipt = json.loads((output / "provenance.json").read_text())
    assert set(receipt["frontend"]) == {"tokenizer.json", "tokenizer_config.json"}
    assert json.loads(completed.stdout)["status"] == "verified"


def test_verified_generation_requires_eos_and_binds_trace(tmp_path, monkeypatch):
    root, manifest = checkpoint(tmp_path)
    binding=[root / name for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json")]
    for path in binding: path.write_text("{}")
    def run(argv, **kwargs):
        output=Path(argv[4]); output.mkdir()
        (output / "routes.tsv").write_text("\n".join(f"0\t{layer}\t" + "\t".join(f"{x}:0.1" for x in range(8)) for layer in range(1,20)) + "\n")
        for layer in range(20): (output / f"layer-{layer}.f32").write_bytes(b"\0\0\0\0")
        (output / "final-hidden.f32").write_bytes(b"\0\0\0\0"); (output / "logits.f32").write_bytes(b"\0\0\0\0")
        (output / "execution.json").write_text(json.dumps({"status":"executed","model":"Ring-mini-2.0","activation_dtype":"FP32","quantized":False,"tokens":2,"token_ids":[1,3],"layers":20,"checkpoint_id":argv[6],"matmul_backend":argv[argv.index("--matmul")+1],"logits_mode":"all","logits_tokens":2}))
        (output / "generation.json").write_text(json.dumps({"prompt_token_ids":[1],"generated_token_ids":[3,9],"eos_token_ids":[9],"max_new_tokens":2,"finish_reason":"eos"}))
        class Finished:
            pid = os.getpid()
            returncode = 0
            def communicate(self, timeout=None): return "", ""
            def wait(self, timeout=None): return 0
        return Finished()
    monkeypatch.setattr(verified.subprocess, "Popen", run)
    with pytest.raises(verified.VerificationError, match="EOS"):
        verified.run_verified(root, manifest, Path("/bin/true"), "1", tmp_path / "bad", 6, generate=2)
    receipt=verified.run_verified(root, manifest, Path("/bin/true"), "1", tmp_path / "good", 6, generate=2, eos_ids="9", binding_files=binding)
    assert receipt["generation"]["generated_token_ids"] == [3, 9]


def test_cancellation_writes_no_receipt(tmp_path, monkeypatch):
    root, manifest = checkpoint(tmp_path)
    started = tmp_path / "started"
    executable = tmp_path / "sleep-engine.py"
    executable.write_text("#!/usr/bin/env python3\nimport os, time\nopen(os.environ[\"RING_STARTED\"], \"w\").write(str(os.getpid()))\ntime.sleep(60)\n")
    executable.chmod(0o755)
    output = tmp_path / "out"
    env = dict(os.environ, RING_STARTED=str(started))
    command = [sys.executable, "tools/ling_ring_bf16/run_ring_verified.py", "--checkpoint", str(root),
               "--manifest", str(manifest), "--executable", str(executable), "--tokens", "1",
               "--output", str(output), "--cache-mib", "6", "--timeout", "30"]
    runner = subprocess.Popen(command, cwd=Path(__file__).parents[3], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 10
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert started.exists()
    child_pid = int(started.read_text())
    runner.send_signal(signal.SIGTERM)
    assert runner.wait(timeout=10) != 0
    for _ in range(100):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("owned child survived wrapper cancellation")
    assert not (output / "provenance.json").exists()


def test_signal_during_spawn_terminates_new_owned_child(monkeypatch):
    terminated = []

    class Finished:
        pid = 424242
        returncode = 143

        def communicate(self, timeout=None):
            return "", ""

    def launch(argv, **kwargs):
        # Exercise the exact race window: the handler is installed, but proc
        # has not yet been assigned by _run_owned.
        signal.raise_signal(signal.SIGTERM)
        return Finished()

    monkeypatch.setattr(verified.subprocess, "Popen", launch)
    monkeypatch.setattr(verified, "_terminate_owned", lambda proc: terminated.append(proc))
    with pytest.raises(verified.VerificationError, match="cancelled by SIGTERM"):
        verified._run_owned(["fake"], 1)
    assert len(terminated) == 1
    assert isinstance(terminated[0], Finished)


def test_spawn_failure_restores_signal_handlers(monkeypatch):
    previous = {name: signal.getsignal(getattr(signal, name)) for name in ("SIGINT", "SIGTERM")}

    def launch(argv, **kwargs):
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(verified.subprocess, "Popen", launch)
    with pytest.raises(OSError, match="synthetic spawn failure"):
        verified._run_owned(["fake"], 1)
    assert signal.getsignal(signal.SIGINT) is previous["SIGINT"]
    assert signal.getsignal(signal.SIGTERM) is previous["SIGTERM"]

def test_torch_router_backend_is_forwarded_and_external_closure_bound(tmp_path, monkeypatch):
    root, manifest = checkpoint(tmp_path)
    libs = tmp_path / "torch-libs"; libs.mkdir()
    for name in ("libtorch_cpu.so", "libtorch_cuda.so", "libc10.so"):
        (libs / name).write_bytes(b"synthetic")
    def fake_ldd(cmd, **kwargs):
        if str(cmd[1]).endswith("/true") or str(cmd[1]).endswith("/gnutrue"):
            return "\n".join(f"{name} => {libs / name} (0x0)" for name in ("libtorch_cpu.so", "libtorch_cuda.so", "libc10.so")) + "\n"
        return ""
    monkeypatch.setattr(verified.subprocess, "check_output", fake_ldd)
    real_popen = __import__("subprocess").Popen
    fake_engine(monkeypatch)
    engine_popen = verified.subprocess.Popen
    def dispatch(argv, **kwargs):
        if argv and argv[0] == "ldd":
            return real_popen(argv, **kwargs)
        return engine_popen(argv, **kwargs)
    monkeypatch.setattr(verified.subprocess, "Popen", dispatch)
    receipt = verified.run_verified(
        root, manifest, Path("/bin/true"), "1", tmp_path / "out", 6,
        router="torch-cuda")
    assert receipt["execution"]["router_backend"] == "torch-cuda"
    assert "--router" in receipt["argv"]
    assert receipt["argv"][receipt["argv"].index("--router") + 1] == "torch-cuda"
    assert any(key.startswith("external:") for key in receipt["relevant_source"].keys())


def test_torch_backend_rejects_non_torch_executable_closure(tmp_path):
    root, _manifest = checkpoint(tmp_path)
    with pytest.raises(verified.VerificationError, match="Torch backend executable closure missing"):
        verified._source_files(root, Path(__file__).parents[3], "torch-cuda", Path("/bin/true"))

def test_torch_router_missing_dynamic_dependency_fails_closed(tmp_path, monkeypatch):
    root, _manifest = checkpoint(tmp_path)
    def fake_ldd(*_args, **_kwargs):
        return "libmissing.so => not found\n"
    monkeypatch.setattr(verified.subprocess, "check_output", fake_ldd)
    with pytest.raises(verified.VerificationError, match="unresolved dynamic dependency"):
        verified._source_files(root, Path(__file__).parents[3], "torch-cuda", Path("/bin/true"))


def test_native_router_omits_optional_flag_for_legacy_engine(tmp_path, monkeypatch):
    root, manifest = checkpoint(tmp_path)
    seen = []
    original = verified._run_owned
    def capture(argv, timeout):
        seen.extend(argv)
        return original(argv, timeout)
    fake_engine(monkeypatch)
    monkeypatch.setattr(verified, "_run_owned", capture)
    verified.run_verified(root, manifest, Path("/bin/true"), "1", tmp_path / "out", 6)
    assert "--router" not in [x for x in seen]

def test_torch_rms_mode_is_forwarded_and_bound(tmp_path, monkeypatch):
    root, manifest = checkpoint(tmp_path)
    libs = tmp_path / "torch-libs"; libs.mkdir()
    for name in ("libtorch_cpu.so", "libtorch_cuda.so", "libc10.so"):
        (libs / name).write_bytes(b"synthetic")
    def fake_ldd(cmd, **kwargs):
        if str(cmd[1]).endswith("/true") or str(cmd[1]).endswith("/gnutrue"):
            return "\n".join(f"{name} => {libs / name} (0x0)" for name in ("libtorch_cpu.so", "libtorch_cuda.so", "libc10.so")) + "\n"
        return ""
    monkeypatch.setattr(verified.subprocess, "check_output", fake_ldd)
    real_popen = __import__("subprocess").Popen
    fake_engine(monkeypatch)
    engine_popen = verified.subprocess.Popen
    def dispatch(argv, **kwargs):
        if argv and argv[0] == "ldd":
            return real_popen(argv, **kwargs)
        return engine_popen(argv, **kwargs)
    monkeypatch.setattr(verified.subprocess, "Popen", dispatch)
    receipt = verified.run_verified(root, manifest, Path("/bin/true"), "1", tmp_path / "out", 6,
                                    rms_mode="torch-cuda")
    assert receipt["rms_mode"] == "torch-cuda"
    assert receipt["execution"]["rms_backend"] == "torch-cuda"
    assert receipt["argv"][receipt["argv"].index("--rms-mode") + 1] == "torch-cuda"
    assert any(key.endswith("libtorch_cpu.so") for key in receipt["relevant_source"])

def test_invalid_rms_mode_refused(tmp_path):
    root, manifest = checkpoint(tmp_path)
    with pytest.raises(verified.VerificationError, match="rms mode"):
        verified.run_verified(root, manifest, Path("/bin/true"), "1", tmp_path / "out", 6,
                              rms_mode="invalid")

def test_torch_attention_and_pointwise_are_forwarded_and_bound(tmp_path, monkeypatch):
    root, manifest = checkpoint(tmp_path)
    libs = tmp_path / "torch-libs"; libs.mkdir()
    for name in ("libtorch_cpu.so", "libtorch_cuda.so", "libc10.so"):
        (libs / name).write_bytes(b"x")
    monkeypatch.setattr(verified, "_source_files", lambda checkpoint, repo, router="native", executable=None, rms_mode="native", attention_backend="native", pointwise="native": ({}, {"torch": {"sha256": "x", "size": 1}}))
    fake_engine(monkeypatch)
    receipt = verified.run_verified(root, manifest, Path("/bin/true"), "1", tmp_path / "out", 6,
                                    attention_backend="torch-cuda", pointwise="torch-cuda")
    assert receipt["attention_backend"] == "torch-cuda"
    assert receipt["pointwise_backend"] == "torch-cuda"
    assert receipt["execution"]["attention_backend"] == "torch-cuda"
    assert receipt["execution"]["pointwise_backend"] == "torch-cuda"
    assert receipt["argv"][receipt["argv"].index("--attention-backend") + 1] == "torch-cuda"
    assert receipt["argv"][receipt["argv"].index("--pointwise") + 1] == "torch-cuda"
