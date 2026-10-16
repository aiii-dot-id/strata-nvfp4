#!/usr/bin/env python3
"""Run the opt-in Ling forward executable with checkpoint/output binding.

This wrapper is a CPU-side admission and provenance boundary.  It does not
claim model quality or equivalence to the original implementation.
Attribution: AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations

import argparse
import hashlib
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
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def _checkpoint_files(checkpoint: Path) -> dict:
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
    for name in sorted(names):
        files["shards"][name] = _pin(checkpoint / name, f"shard {name}")
    return files


def _source_files(checkpoint: Path, repo: Path) -> tuple[dict, dict]:
    original = {name: _pin(checkpoint / name, f"original source {name}") for name in
                ("modeling_bailing_moe_v3.py", "configuration_bailing_moe_v3.py")}
    paths = [
        *sorted((repo / "include/strata/ling").glob("*.hpp")),
        *sorted((repo / "src/ling").glob("*")),
        repo / "src/ling/engine.cu",
        *sorted((repo / "include/strata/ring").glob("*.hpp")),
        *sorted((repo / "src/ring").glob("*")),
        repo / "include/strata/kernels/bf16_gemv.hpp",
        repo / "include/strata/kernels/bf16_bits.hpp",
        repo / "include/strata/kernels/pdl.hpp",
        repo / "src/kernels/cuda/native_bf16.cu",
        repo / "src/kernels/cuda/s26_tsum.cuh",
        repo / "tools/ling_ring_bf16/prepare_ling_manifest.py",
        repo / "tools/ling_ring_bf16/ling/CMakeLists.txt",
        repo / "tools/ling_ring_bf16/run_ling_verified.py",
    ]
    relevant = {}
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        rel = str(_file(path, "relevant source").relative_to(repo))
        relevant[rel] = _pin(path, "relevant source")
    return original, relevant


def _all_inputs(checkpoint: Path, manifest: Path, executable: Path, repo: Path) -> dict:
    original, relevant = _source_files(checkpoint, repo)
    return {"checkpoint": _checkpoint_files(checkpoint), "original_source": original,
            "relevant_source": relevant, "manifest": _pin(manifest, "manifest"),
            "executable": _pin(executable, "executable")}


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


def _require_execution(path: Path, token_ids: list[int], checkpoint_id: str) -> dict:
    execution = _read_json(path, "execution receipt")
    expected = {"status": "executed", "model": MODEL, "activation_dtype": "FP32",
                "quantized": False, "layers": LAYERS, "tokens": len(token_ids),
                "token_ids": token_ids, "checkpoint_id": checkpoint_id}
    for key, value in expected.items():
        if execution.get(key) != value:
            raise VerificationError(f"execution receipt mismatch: {key}")
    if type(execution.get("quantized")) is not bool:
        raise VerificationError("execution quantized must be boolean")
    if execution.get("logits_mode") != "all" or execution.get("logits_tokens") != len(token_ids):
        raise VerificationError("Ling execution must emit all logits rows")
    return execution


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


def _native_outputs(output: Path, tokens: int) -> dict:
    result = {}
    for name in EXPECTED_NATIVE:
        path = _file(output / name, f"native output {name}")
        if name.startswith("layer-") or name == "final-hidden.f32":
            _finite_f32(path, tokens * HIDDEN * 4, name)
        elif name == "logits.f32":
            _finite_f32(path, tokens * VOCAB * 4, name)
        elif name == "routes.tsv":
            _route_rows(path, tokens)
        result[name] = {"sha256": _sha(path), "size": path.stat().st_size}
    return result


def run_verified(checkpoint: Path, manifest: Path, executable: Path, tokens: str, output: Path,
                 cache_mib: int, timeout: float = 1800) -> dict:
    checkpoint, manifest, executable, output = (Path(x).resolve() for x in (checkpoint, manifest, executable, output))
    repo = Path(__file__).resolve().parents[2]
    token_ids = _tokens(tokens)
    if output.exists():
        raise VerificationError(f"refusing existing output: {output}")
    if type(cache_mib) is not int or not 5 <= cache_mib <= 2048:
        raise VerificationError("cache budget must be 5..2048 MiB")
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
    before = _all_inputs(checkpoint, manifest, executable, repo)
    argv = [str(executable), str(checkpoint), str(manifest), tokens, str(output), str(cache_mib), checkpoint_id]
    returncode, _stdout, stderr = _run_owned(argv, timeout)
    if returncode != 0:
        raise VerificationError(f"Ling executable failed ({returncode}): {stderr.strip()[-2000:]}")
    if not output.is_dir():
        raise VerificationError("Ling executable did not create output directory")
    after = _all_inputs(checkpoint, manifest, executable, repo)
    if before != after:
        raise VerificationError("checkpoint, executable, manifest, or source drifted during run")
    execution = _require_execution(output / "execution.json", token_ids, checkpoint_id)
    native = _native_outputs(output, len(token_ids))
    receipt = {"schema": "strata-ling-verified-run-v1", "status": "verified",
               "model": MODEL, "checkpoint_id": checkpoint_id, "checkpoint": before["checkpoint"],
               "original_source": before["original_source"], "relevant_source": before["relevant_source"],
               "manifest": before["manifest"], "executable": before["executable"], "argv": argv,
               "cache_mib": cache_mib, "execution": execution, "native_outputs": native,
               "scope": {"checkpoint_files": "config, index, all 32 safetensors shards, and original model/configuration sources hashed before and after execution",
                         "reference": "native Ling forward receipt and structural outputs only; no claim of original equivalence, activation quality, or production readiness"}}
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
    args = parser.parse_args()
    try:
        print(json.dumps(run_verified(args.checkpoint, args.manifest, args.executable, args.tokens, args.output, args.cache_mib, args.timeout), indent=2, sort_keys=True))
    except (OSError, ValueError, VerificationError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
