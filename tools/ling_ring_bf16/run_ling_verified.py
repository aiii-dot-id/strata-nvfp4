#!/usr/bin/env python3
"""Run the opt-in Ling forward executable with checkpoint/output binding.

This wrapper is a CPU-side admission and provenance boundary.  It does not
claim model quality or equivalence to the original implementation.
Attribution: AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations

import argparse
import hashlib

from verified_hash import require_hash_workers, sha256_file, sha256_files
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import tempfile

from prepare_ling_manifest import AdmissionError, prepare


class VerificationError(RuntimeError):
    pass

_CANCEL_GRACE_SECONDS = 5.0
MODEL = "Ling-3.0-tiny"
LAYERS = 24
HIDDEN = 1536
VOCAB = 157184
EXPERTS = 128
TOPK = 8
MAX_TOKENS = 1024
EXPECTED_ROUTE_LAYERS = tuple(range(1, LAYERS))
EXPECTED_NATIVE = tuple([*(f"layer-{i}.f32" for i in range(LAYERS)), "final-hidden.f32", "logits.f32", "routes.tsv", "execution.json"])
HEX64 = __import__("re").compile(r"[0-9a-f]{64}\Z")


def _terminate_owned(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=_CANCEL_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=_CANCEL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def _run_owned(argv: list[str], timeout: float) -> tuple[int, str, str]:
    proc = None
    interrupted = {"signal": None}

    def cancel(signum, _frame):
        interrupted["signal"] = signum
        if proc is not None:
            _terminate_owned(proc)

    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        if interrupted["signal"] is not None:
            _terminate_owned(proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_owned(proc)
            try:
                proc.communicate(timeout=_CANCEL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _terminate_owned(proc)
            raise VerificationError(f"Ling execution timed out after {timeout:g} seconds; no verification receipt was written")
    finally:
        signal.signal(signal.SIGINT, previous[signal.SIGINT])
        signal.signal(signal.SIGTERM, previous[signal.SIGTERM])
    if interrupted["signal"] is not None:
        raise VerificationError(f"Ling execution cancelled by {signal.Signals(interrupted['signal']).name}; no verification receipt was written")
    return proc.returncode, stdout, stderr


def _sha(path: Path) -> str:
    return sha256_file(path)


def _file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise VerificationError(f"{label} must be a regular file: {path}")
    return path


def _unique(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            raise VerificationError(f"duplicate JSON key: {key}")
        values[key] = value
    return values


def _read_json(path: Path, label: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid {label}: {exc}") from exc


def _pin(path: Path, label: str) -> dict:
    path = _file(path, label)
    return {"path": str(path), "sha256": _sha(path), "size": path.stat().st_size}


def _checkpoint_files(checkpoint: Path, hash_workers: int = 1) -> dict:
    config = _file(checkpoint / "config.json", "config")
    index = _file(checkpoint / "model.safetensors.index.json", "index")
    data = _read_json(index, "checkpoint index")
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise VerificationError("checkpoint index has no weight_map")
    names = set()
    for name in weight_map.values():
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts or len(Path(name).parts) != 1:
            raise VerificationError(f"unsafe shard path: {name!r}")
        names.add(name)
    if len(names) != 32:
        raise VerificationError(f"Ling checkpoint must contain exactly 32 shards, found {len(names)}")
    files = {"config": _pin(config, "config"), "index": _pin(index, "index"), "shards": {}}
    shard_paths = {name: _file(checkpoint / name, f"shard {name}") for name in sorted(names)}
    digests = sha256_files((shard_paths[name] for name in sorted(names)), hash_workers)
    for name, digest in zip(sorted(names), digests):
        shard = shard_paths[name]
        files["shards"][name] = {"path": str(shard), "sha256": digest, "size": shard.stat().st_size}
    return files


def _dynamic_library_closure(executable: Path, require_cublas: bool) -> dict:
    """Pin canonical ldd closure for the opt-in cuBLAS backend."""
    pending = [Path(executable).resolve(strict=True)]
    seen: set[Path] = set()
    paths: set[Path] = set()
    cublas = False
    while pending:
        lib = pending.pop().resolve(strict=True)
        if lib in seen:
            continue
        seen.add(lib)
        try:
            text = subprocess.check_output(["ldd", str(lib)], text=True, stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise VerificationError(f"cannot inspect dynamic library closure: {lib}") from exc
        for line in text.splitlines():
            if "not found" in line:
                raise VerificationError(f"unresolved dynamic dependency for {lib}: {line.strip()}")
            candidate = None
            if "=>" in line:
                rhs = line.split("=>", 1)[1].strip().split(None, 1)[0]
                if rhs.startswith("/"):
                    candidate = Path(rhs)
            elif line.strip().startswith("/"):
                candidate = Path(line.strip().split(None, 1)[0])
            if candidate is None:
                continue
            try:
                candidate = candidate.resolve(strict=True)
            except OSError as exc:
                raise VerificationError(f"unresolved dynamic dependency: {candidate}") from exc
            paths.add(candidate)
            if "libcublas.so" in candidate.name:
                cublas = True
            if candidate not in seen:
                pending.append(candidate)
    if require_cublas and not cublas:
        raise VerificationError("fp32-cublas executable closure has no libcublas.so")
    return {"external:" + str(path): _pin(path, "dynamic library") for path in sorted(paths)}

def _source_files(checkpoint: Path, repo: Path, executable: Path | None = None, matmul: str = "bf16-gemv", pointwise: str = "native") -> tuple[dict, dict]:
    original = {name: _pin(checkpoint / name, f"original source {name}") for name in
                ("modeling_bailing_moe_v3.py", "configuration_bailing_moe_v3.py")}
    # Keep this closure aligned with ling/CMakeLists.txt.  The Ling CUDA
    # target links these three Ring translation units; unrelated Ring engine
    # and attention sources must not make an in-progress Ring change alter a
    # Ling receipt.
    paths = [
        *sorted((repo / "include/strata/ling").glob("*.hpp")),
        *sorted((repo / "src/ling").glob("*")),
        *(repo / "include/strata/ring" / name for name in
          ("expert_cache.hpp", "generation.hpp", "continuation.hpp", "numerics.hpp", "routing.hpp", "tensor_store.hpp")),
        *(repo / "src/ring" / name for name in
          ("routing.cpp", "expert_cache.cpp", "tensor_store.cpp")),
        repo / "include/strata/kernels/bf16_gemv.hpp",
        repo / "include/strata/kernels/bf16_bits.hpp",
        repo / "include/strata/kernels/pdl.hpp",
        repo / "src/kernels/cuda/native_bf16.cu",
        repo / "src/kernels/cuda/s26_tsum.cuh",
        repo / "tools/ling_ring_bf16/prepare_ling_manifest.py",
        repo / "tools/ling_ring_bf16/ling/CMakeLists.txt",
        *sorted((repo / "tools/ling_ring_bf16/ling").glob("ling_torch_ops.*")),
        repo / "tools/ling_ring_bf16/run_ling_verified.py",
        repo / "tools/ling_ring_bf16/verified_hash.py",
        repo / "tools/ling_ring_bf16/generate_ling.py",
    ]
    relevant = {}
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        rel = str(_file(path, "relevant source").relative_to(repo))
        relevant[rel] = _pin(path, "relevant source")
    if matmul == "fp32-cublas" or pointwise == "torch-cuda":
        if executable is None:
            raise VerificationError("opt-in backend requires an executable for library binding")
        relevant.update(_dynamic_library_closure(executable, matmul == "fp32-cublas"))
    return original, relevant


def _all_inputs(checkpoint: Path, manifest: Path, executable: Path, repo: Path, binding_files=(), matmul: str = "bf16-gemv", pointwise: str = "native", hash_workers: int = 1) -> dict:
    original, relevant = (_source_files(checkpoint, repo) if matmul == "bf16-gemv" and pointwise == "native" else _source_files(checkpoint, repo, executable, matmul, pointwise))
    frontend = {}
    for path in binding_files:
        path = _file(Path(path), "frontend binding")
        try: name = str(path.relative_to(checkpoint))
        except ValueError as exc: raise VerificationError("frontend binding must be inside checkpoint") from exc
        frontend[name] = _pin(path, "frontend binding")
    return {"checkpoint": _checkpoint_files(checkpoint, hash_workers), "original_source": original,
            "relevant_source": relevant,
            "runtime_libraries": {k: v for k, v in relevant.items() if k.startswith("external:")},
            "manifest": _pin(manifest, "manifest"),
            "executable": _pin(executable, "executable"), "frontend": frontend}


def _tokens(text: str) -> list[int]:
    parts = text.split(",")
    if not parts or any(not p or any(ch not in "0123456789" for ch in p) for p in parts):
        raise VerificationError("tokens must be comma-separated unsigned decimal integers")
    try:
        values = [int(p, 10) for p in parts]
    except ValueError as exc:
        raise VerificationError("tokens must be decimal integers") from exc
    if not 1 <= len(values) <= MAX_TOKENS or any(type(x) is not int or x < 0 or x >= VOCAB for x in values):
        raise VerificationError(f"tokens must contain 1..{MAX_TOKENS} vocabulary IDs")
    return values


def _parse_route_probe(text: str) -> tuple[int, int]:
    parts = text.split(",")
    if len(parts) != 2 or any(not part or any(ch not in "0123456789" for ch in part) for part in parts):
        raise VerificationError("route probe must be POS,LAYER unsigned decimals")
    position, layer = (int(part, 10) for part in parts)
    if position >= MAX_TOKENS or layer < 1 or layer >= LAYERS:
        raise VerificationError("route probe position/layer is out of range")
    return position, layer


def _require_execution(path: Path, token_ids: list[int], checkpoint_id: str,
                       generate: int = 0, rope_frequency: str = "native",
                       rms_mode: str = "native", matmul: str = "bf16-gemv", pointwise: str = "native", lm_head_residency: str = "off") -> dict:
    execution = _read_json(path, "execution receipt")
    expected = {"status": "executed", "model": MODEL, "activation_dtype": "FP32",
                "quantized": False, "layers": LAYERS, "checkpoint_id": checkpoint_id,
                "rope_frequency": rope_frequency, "rms_mode": rms_mode, "matmul_backend": matmul}
    if execution.get("pointwise_mode", "native") != pointwise:
        raise VerificationError("execution receipt mismatch: pointwise_mode")
    for key, value in expected.items():
        if execution.get(key) != value:
            raise VerificationError(f"execution receipt mismatch: {key}")
    if type(execution.get("quantized")) is not bool or execution.get("quantized") is not False:
        raise VerificationError("execution quantized must be false boolean")
    if execution.get("lm_head_residency", "off") != lm_head_residency:
        raise VerificationError("execution receipt mismatch: lm_head_residency")
    residency_fields = ("lm_head_resident_bytes", "lm_head_resident_uploads",
                        "lm_head_resident_apply_calls")
    if lm_head_residency == "cuda":
        expected = (157184 * 1536 * 2, 1)
        values = tuple(execution.get(key) for key in residency_fields)
        if any(type(value) is not int for value in values) or values[:2] != expected:
            raise VerificationError("execution lm-head residency accounting mismatch")
        logits_tokens = execution.get("logits_tokens")
        if type(logits_tokens) is not int or logits_tokens < 0 or values[2] != logits_tokens:
            raise VerificationError("execution lm-head residency call count mismatch")
    elif any(key in execution for key in residency_fields):
        values = tuple(execution.get(key) for key in residency_fields)
        if any(type(value) is not int for value in values) or values != (0, 0, 0):
            raise VerificationError("off lm-head residency accounting mismatch")
    scratch = execution.get("matmul_fp32_weight_scratch_bytes")
    workspace = execution.get("workspace_bytes")
    if type(scratch) is not int or isinstance(scratch, bool) or scratch < 0 or type(workspace) is not int or isinstance(workspace, bool) or workspace < 0:
        raise VerificationError("execution workspace accounting is invalid")
    expected_scratch = 157184 * 1536 * 4 if matmul == "fp32-cublas" else 0
    expected_workspace = 157184 * 1536 * 2 + 4608 * 4 + 157184 * 4 + expected_scratch
    if scratch != expected_scratch or workspace != expected_workspace:
        raise VerificationError("execution workspace does not match matmul backend")
    evaluated = execution.get("token_ids")
    if not isinstance(evaluated, list) or any(type(x) is not int or isinstance(x, bool) or x < 0 or x >= VOCAB for x in evaluated):
        raise VerificationError("execution token identity is invalid")
    if type(execution.get("tokens")) is not int or isinstance(execution.get("tokens"), bool):
        raise VerificationError("execution token count is invalid")
    if generate:
        if execution.get("logits_mode") != "all" or execution.get("logits_tokens") != len(execution.get("token_ids", [])):
            raise VerificationError("generation execution must retain all logits")
        evaluated = execution.get("token_ids")
        if not isinstance(evaluated, list) or evaluated[:len(token_ids)] != token_ids or not len(token_ids) <= len(evaluated) <= len(token_ids) + generate:
            raise VerificationError("generated execution token trace is invalid")
        if execution.get("tokens") != len(evaluated):
            raise VerificationError("generated execution token count mismatch")
    else:
        if execution.get("tokens") != len(token_ids) or execution.get("token_ids") != token_ids:
            raise VerificationError("execution token identity mismatch")
        if execution.get("logits_mode") != "all" or execution.get("logits_tokens") != len(token_ids):
            raise VerificationError("Ling execution must emit all logits rows")
    return execution
def _parse_eos(text: str) -> list[int]:
    try: values=[int(x,10) for x in text.split(",")]
    except ValueError as exc: raise VerificationError("EOS IDs must be decimal integers") from exc
    if not values or any(type(x) is not int or x < 0 or x >= VOCAB for x in values) or len(set(values)) != len(values):
        raise VerificationError("EOS IDs must be unique vocabulary IDs")
    return values
def _require_generation(path: Path, token_ids: list[int], maximum: int, eos_ids: list[int], schema: str = "strata-ling-generation-v1") -> dict:
    value=_read_json(path, "generation receipt")
    if value.get("schema") != schema:
        raise VerificationError("generation receipt schema mismatch")
    prompt=value.get("prompt_token_ids"); generated=value.get("generated_token_ids")
    configured=value.get("eos_token_ids"); declared=value.get("max_new_tokens")
    if (not isinstance(prompt,list) or any(type(x) is not int or isinstance(x,bool) or x < 0 or x >= VOCAB for x in prompt)
        or prompt != token_ids or configured != eos_ids or type(declared) is not int or isinstance(declared,bool)
        or declared != maximum or not isinstance(generated,list) or not generated
        or any(type(x) is not int or isinstance(x,bool) or x < 0 or x >= VOCAB for x in generated) or len(generated)>maximum):
        raise VerificationError("generation token trace is invalid")
    if any(x in eos_ids for x in generated[:-1]): raise VerificationError("generation continued after EOS")
    reason=value.get("finish_reason")
    if reason not in {"eos","length"}: raise VerificationError("generation finish reason is invalid")
    if reason=="eos" and generated[-1] not in eos_ids: raise VerificationError("EOS trace does not end in configured EOS")
    if reason=="length" and (len(generated)!=maximum or generated[-1] in eos_ids): raise VerificationError("length trace does not consume budget")
    return value


def _finite_f32(path: Path, expected_bytes: int, label: str) -> None:
    if path.stat().st_size != expected_bytes:
        raise VerificationError(f"{label} size mismatch: expected {expected_bytes}, got {path.stat().st_size}")
    with path.open("rb") as stream:
        remaining = expected_bytes
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block or len(block) % 4:
                raise VerificationError(f"{label} is truncated")
            for (value,) in struct.iter_unpack("<f", block):
                if not math.isfinite(value):
                    raise VerificationError(f"{label} contains non-finite FP32")
            remaining -= len(block)


def _route_rows(path: Path, tokens: int) -> None:
    expected = {(token, layer) for token in range(tokens) for layer in EXPECTED_ROUTE_LAYERS}
    seen = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise VerificationError(f"invalid routes.tsv: {exc}") from exc
    for line in lines:
        cols = line.split("\t")
        if len(cols) != 2 + TOPK:
            raise VerificationError("route row must contain token, layer, and exactly eight experts")
        try:
            token, layer = int(cols[0], 10), int(cols[1], 10)
        except ValueError as exc:
            raise VerificationError("route token/layer must be decimal integers") from exc
        key = (token, layer)
        if key not in expected or key in seen:
            raise VerificationError("route token/layer is out of range or duplicated")
        seen.add(key)
        ids = []
        for item in cols[2:]:
            pair = item.split(":")
            if len(pair) != 2:
                raise VerificationError("route expert must be id:weight")
            try:
                expert = int(pair[0], 10)
                weight = float(pair[1])
            except ValueError as exc:
                raise VerificationError("route expert and weight must be numeric") from exc
            if type(expert) is not int or not 0 <= expert < EXPERTS or expert in ids or not math.isfinite(weight):
                raise VerificationError("route expert ID/weight is invalid")
            ids.append(expert)
    if seen != expected:
        raise VerificationError(f"route rows incomplete: expected {len(expected)}, got {len(seen)}")


def _route_probe(path: Path, expected: tuple[int, int]) -> None:
    value = _read_json(path, "route probe")
    if value.get("position") != expected[0] or value.get("layer") != expected[1]:
        raise VerificationError("route probe position/layer mismatch")
    for name, count in (("input_hidden", HIDDEN), ("post_attention_hidden", HIDDEN),
                        ("gate_input", HIDDEN), ("router_logits", EXPERTS),
                        ("sigmoid_scores", EXPERTS), ("expert_bias", EXPERTS),
                        ("bias_corrected_scores", EXPERTS), ("selected_weights", TOPK)):
        values = value.get(name)
        if not isinstance(values, list) or len(values) != count or any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(float(x)) for x in values):
            raise VerificationError(f"route probe {name} is invalid")
    ids = value.get("selected_expert_ids")
    if not isinstance(ids, list) or len(ids) != TOPK or any(type(x) is not int or isinstance(x, bool) or not 0 <= x < EXPERTS for x in ids) or len(set(ids)) != TOPK:
        raise VerificationError("route probe selected expert IDs are invalid")


def _native_outputs(output: Path, tokens: int, generation: bool = False, route_probe: tuple[int, int] | None = None) -> dict:
    result = {}
    names = EXPECTED_NATIVE + (("generation.json",) if generation else ()) + (("route-probe.json",) if route_probe else ())
    for name in names:
        path = _file(output / name, f"native output {name}")
        if name.startswith("layer-") or name == "final-hidden.f32":
            _finite_f32(path, tokens * HIDDEN * 4, name)
        elif name == "logits.f32":
            _finite_f32(path, tokens * VOCAB * 4, name)
        elif name == "routes.tsv":
            _route_rows(path, tokens)
        elif name == "route-probe.json":
            _route_probe(path, route_probe)
        result[name] = {"sha256": _sha(path), "size": path.stat().st_size}
    return result


def run_verified(checkpoint: Path, manifest: Path, executable: Path, tokens: str, output: Path,
                 cache_mib: int, timeout: float = 1800, generate: int = 0, eos_ids=None,
                 binding_files=(), route_probe=None, rope_frequency: str = "native",
                 rms_mode: str = "native", matmul: str = "bf16-gemv", pointwise: str = "native", lm_head_residency: str = "off", hash_workers: int = 1) -> dict:
    checkpoint, manifest, executable, output = (Path(x).resolve() for x in (checkpoint, manifest, executable, output))
    repo = Path(__file__).resolve().parents[2]
    token_ids = _tokens(tokens)
    parsed_probe = _parse_route_probe(route_probe) if route_probe is not None else None
    if rope_frequency not in {"native", "original"}:
        raise VerificationError("rope_frequency must be native or original")
    if rms_mode not in {"native", "torch-cuda-order"}:
        raise VerificationError("rms_mode must be native or torch-cuda-order")
    if matmul not in {"bf16-gemv", "fp32-cublas"}:
        raise VerificationError("matmul must be bf16-gemv or fp32-cublas")
    if pointwise not in {"native", "torch-cuda"}:
        raise VerificationError("pointwise must be native or torch-cuda")
    if lm_head_residency not in {"off", "cuda"}:
        raise VerificationError("lm_head_residency must be off or cuda")
    try:
        require_hash_workers(hash_workers)
    except ValueError as exc:
        raise VerificationError(str(exc)) from exc
    if output.exists():
        raise VerificationError(f"refusing existing output: {output}")
    if type(cache_mib) is not int or not 5 <= cache_mib <= 2048:
        raise VerificationError("cache budget must be 5..2048 MiB")
    if type(generate) is not int or generate < 0 or generate > MAX_TOKENS - len(token_ids):
        raise VerificationError("generate must fit within the 1024-token evaluation window")
    if generate:
        if isinstance(eos_ids,str): parsed_eos=_parse_eos(eos_ids)
        else: parsed_eos=list(eos_ids or [])
        if not parsed_eos or len(set(parsed_eos)) != len(parsed_eos) or any(type(x) is not int or x < 0 or x >= VOCAB for x in parsed_eos):
            raise VerificationError("EOS IDs are mandatory and must be unique vocabulary IDs")
    else: parsed_eos=[]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 86400:
        raise VerificationError("timeout must be between 0 and 86400 seconds")
    _file(executable, "executable")
    if not os.access(executable, os.X_OK):
        raise VerificationError("executable is not executable")
    with tempfile.TemporaryDirectory(prefix="ling-manifest-") as temp:
        try:
            prepare(checkpoint, Path(temp) / "admission")
        except (OSError, ValueError, AdmissionError) as exc:
            raise VerificationError(f"checkpoint admission failed: {exc}") from exc
        generated = Path(temp) / "admission" / "manifest.tsv"
        requested = _file(manifest, "manifest")
        if requested.read_bytes() != generated.read_bytes():
            raise VerificationError("requested manifest differs from freshly prepared manifest")
    checkpoint_id = _sha(manifest)
    if generate and not {"tokenizer.json", "tokenizer_config.json", "generation_config.json"} <= {Path(x).name for x in binding_files}:
        raise VerificationError("generation requires tokenizer and generation-config bindings")
    before = _all_inputs(checkpoint, manifest, executable, repo, binding_files, matmul, pointwise, hash_workers)
    argv = [str(executable), str(checkpoint), str(manifest), tokens, str(output), str(cache_mib), checkpoint_id]
    if generate:
        argv += ["--generate", str(generate), "--eos", ",".join(map(str, parsed_eos))]
    if parsed_probe:
        argv += ["--route-probe", f"{parsed_probe[0]},{parsed_probe[1]}"]
    argv += ["--rope-frequency", rope_frequency]
    argv += ["--rms-mode", rms_mode, "--matmul", matmul, "--pointwise", pointwise]
    if lm_head_residency != "off": argv += ["--lm-head-residency", lm_head_residency]
    returncode, _stdout, stderr = _run_owned(argv, timeout)
    if returncode != 0:
        raise VerificationError(f"Ling executable failed ({returncode}): {stderr.strip()[-2000:]}")
    if not output.is_dir():
        raise VerificationError("Ling executable did not create output directory")
    after = _all_inputs(checkpoint, manifest, executable, repo, binding_files, matmul, pointwise, hash_workers)
    if before != after:
        raise VerificationError("checkpoint, executable, manifest, or source drifted during run")
    execution = _require_execution(output / "execution.json", token_ids, checkpoint_id, generate, rope_frequency, rms_mode, matmul, pointwise, lm_head_residency)
    if lm_head_residency == "cuda" and execution.get("lm_head_resident_apply_calls") != execution.get("logits_tokens"):
        raise VerificationError("execution lm-head residency call count mismatch")
    evaluated_tokens = len(execution.get("token_ids", token_ids))
    native = _native_outputs(output, evaluated_tokens, bool(generate), parsed_probe)
    receipt = {"schema": "strata-ling-verified-run-v1", "status": "verified",
               "model": MODEL, "checkpoint_id": checkpoint_id, "checkpoint": before["checkpoint"],
               "original_source": before["original_source"], "relevant_source": before["relevant_source"],
               "runtime_libraries": before["runtime_libraries"],
               "manifest": before["manifest"], "executable": before["executable"], "frontend": before["frontend"], "argv": argv,
               "cache_mib": cache_mib, "execution": execution, "native_outputs": native,
               "pointwise_mode": pointwise, "lm_head_residency": lm_head_residency,
               "verification_hash_workers": hash_workers,
               "pointwise_scope": ("native operations" if pointwise == "native" else
                                   "ATen CUDA general RMS, router selection/weights, KDA beta and composed output norm/gate, SwiGLU, and MoE weighted reduction; KDA recurrence remains native C++"),
               "scope": {"checkpoint_files": "config, index, all 32 safetensors shards, original model/configuration sources, and the Ling build source closure (all Ling sources/headers, shared headers, the three linked Ring translation units, CUDA kernels, CMake, wrapper, and frontend) hashed before and after execution",
                         "reference": "native Ling forward receipt and structural outputs only; no claim of original equivalence, activation quality, or production readiness"}}
    if parsed_probe:
        receipt["route_probe"] = {"position": parsed_probe[0], "layer": parsed_probe[1], "output": "route-probe.json"}
    if generate:
        receipt["generation"] = _require_generation(output / "generation.json", token_ids, generate, parsed_eos)
        expected = token_ids + receipt["generation"]["generated_token_ids"][:-1]
        if execution.get("token_ids") != expected:
            raise VerificationError("generated execution trace must omit only the last sampled token")
    (output / "provenance.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-mib", required=True, type=int)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--generate", type=int, default=0)
    parser.add_argument("--eos", help="comma-separated EOS IDs; required with --generate")
    parser.add_argument("--bind-file", action="append", type=Path, default=[], help="checkpoint-local frontend file to hash and bind")
    parser.add_argument("--route-probe", help="capture one POS,LAYER router state")
    parser.add_argument("--rope-frequency", choices=("native", "original"), default="native")
    parser.add_argument("--rms-mode", choices=("native", "torch-cuda-order"), default="native")
    parser.add_argument("--matmul", choices=("bf16-gemv", "fp32-cublas"), default="bf16-gemv")
    parser.add_argument("--pointwise", choices=("native", "torch-cuda"), default="native")
    parser.add_argument("--lm-head-residency", choices=("off", "cuda"), default="off")
    parser.add_argument("--hash-workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    try:
        print(json.dumps(run_verified(args.checkpoint, args.manifest, args.executable, args.tokens, args.output, args.cache_mib, args.timeout, args.generate, args.eos, binding_files=args.bind_file, route_probe=args.route_probe, rope_frequency=args.rope_frequency, rms_mode=args.rms_mode, matmul=args.matmul, pointwise=args.pointwise, lm_head_residency=args.lm_head_residency, hash_workers=args.hash_workers), indent=2, sort_keys=True))
    except (OSError, ValueError, VerificationError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
