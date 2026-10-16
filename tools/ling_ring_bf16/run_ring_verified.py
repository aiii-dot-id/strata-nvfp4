#!/usr/bin/env python3
"""Run the opt-in Ring executable with an auditable checkpoint binding."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from prepare_ring_manifest import CheckpointError, prepare


class VerificationError(RuntimeError):
    pass


EXPECTED_NATIVE = (
    "routes.tsv",
    "final-hidden.f32",
    "logits.f32",
    *(f"layer-{i}.f32" for i in range(20)),
    "execution.json",
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise VerificationError(f"{label} must be a regular file: {path}")
    return path


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise VerificationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json(path: Path, label: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid {label}: {exc}") from exc


def _checkpoint_files(checkpoint: Path) -> tuple[dict, dict]:
    config = _file(checkpoint / "config.json", "config")
    index = _file(checkpoint / "model.safetensors.index.json", "index")
    index_data = _read_json(index, "checkpoint index")
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise VerificationError("checkpoint index has no weight_map")
    names = set()
    for name in weight_map.values():
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise VerificationError(f"unsafe shard path: {name!r}")
        names.add(name)
    files = {
        "config": {"path": str(config), "sha256": _sha(config), "size": config.stat().st_size},
        "index": {"path": str(index), "sha256": _sha(index), "size": index.stat().st_size},
        "shards": {},
    }
    for name in sorted(names):
        shard = _file(checkpoint / name, f"shard {name}")
        files["shards"][name] = {"sha256": _sha(shard), "size": shard.stat().st_size}
    return files, index_data


def _source_files(checkpoint: Path, repo: Path) -> tuple[dict, dict]:
    original = {}
    for name in ("modeling_bailing_moe_v2.py", "configuration_bailing_moe_v2.py"):
        path = _file(checkpoint / name, f"original source {name}")
        original[name] = {"path": str(path), "sha256": _sha(path), "size": path.stat().st_size}

    relevant = {}
    paths = [
        *sorted((repo / "include/strata/ring").glob("*.hpp")),
        repo / "include/strata/kernels/bf16_gemv.hpp",
        repo / "include/strata/kernels/bf16_bits.hpp",
        repo / "include/strata/kernels/pdl.hpp",
        *sorted((repo / "src/ring").glob("*")),
        repo / "src/kernels/cuda/native_bf16.cu",
        repo / "src/kernels/cuda/s26_tsum.cuh",
        repo / "tools/ling_ring_bf16/CMakeLists.txt",
        repo / "tools/ling_ring_bf16/inspect_checkpoint.py",
        repo / "tools/ling_ring_bf16/prepare_ring_manifest.py",
        repo / "tools/ling_ring_bf16/ring_original_reference.py",
        repo / "tools/ling_ring_bf16/compare_ring_reference.py",
        repo / "tools/ling_ring_bf16/run_ring_verified.py",
    ]
    for path in paths:
        _file(path, "relevant source")
        rel = str(path.relative_to(repo))
        relevant[rel] = {"sha256": _sha(path), "size": path.stat().st_size}
    return original, relevant


def _all_inputs(checkpoint: Path, manifest: Path, executable: Path, repo: Path) -> dict:
    checkpoint_files, _ = _checkpoint_files(checkpoint)
    original, relevant = _source_files(checkpoint, repo)
    return {
        "checkpoint": checkpoint_files,
        "original_source": original,
        "relevant_source": relevant,
        "manifest": {"sha256": _sha(_file(manifest, "manifest")), "size": manifest.stat().st_size},
        "executable": {"sha256": _sha(_file(executable, "executable")), "size": executable.stat().st_size},
    }


def _tokens(text: str) -> list[int]:
    parts = text.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise VerificationError("tokens must be a comma-separated nonempty list")
    try:
        values = [int(part, 10) for part in parts]
    except ValueError as exc:
        raise VerificationError("tokens must be decimal integers") from exc
    if not 1 <= len(values) <= 4096 or any(value < 0 or value >= 157184 for value in values):
        raise VerificationError("tokens must contain 1..4096 vocabulary IDs")
    return values


def _require_execution(path: Path, token_ids: list[int], checkpoint_id: str) -> dict:
    execution = _read_json(path, "execution receipt")
    required = {
        "status": "executed",
        "model": "Ring-mini-2.0",
        "activation_dtype": "FP32",
        "quantized": False,
        "tokens": len(token_ids),
        "token_ids": token_ids,
        "layers": 20,
        "checkpoint_id": checkpoint_id,
    }
    for key, value in required.items():
        if execution.get(key) != value:
            raise VerificationError(f"execution receipt mismatch: {key}")
    if type(execution.get("quantized")) is not bool or execution["quantized"] is not False:
        raise VerificationError("execution receipt mismatch: quantized")
    if type(execution.get("tokens")) is not int or any(type(x) is not int for x in execution.get("token_ids", [])):
        raise VerificationError("execution receipt has non-integer token identity")
    return execution


def run_verified(checkpoint: Path, manifest: Path, executable: Path, tokens: str, output: Path, cache_mib: int) -> dict:
    checkpoint = checkpoint.resolve()
    manifest = manifest.resolve()
    executable = executable.resolve()
    output = output.resolve()
    repo = Path(__file__).resolve().parents[2]
    token_ids = _tokens(tokens)
    if output.exists():
        raise VerificationError(f"refusing existing output: {output}")
    if cache_mib < 6 or cache_mib > 2048:
        raise VerificationError("cache budget must be 6..2048 MiB")
    _file(executable, "executable")
    if not os.access(executable, os.X_OK):
        raise VerificationError("executable is not executable")

    with tempfile.TemporaryDirectory(prefix="ring-manifest-") as temp:
        try:
            admission = prepare(checkpoint, Path(temp) / "admission")
        except (OSError, ValueError, CheckpointError) as exc:
            raise VerificationError(f"checkpoint admission failed: {exc}") from exc
        generated = Path(temp) / "admission/manifest.tsv"
        requested = _file(manifest, "manifest")
        if requested.read_bytes() != generated.read_bytes():
            raise VerificationError("requested manifest differs from freshly prepared manifest")

    manifest_hash = _sha(manifest)
    before = _all_inputs(checkpoint, manifest, executable, repo)
    argv = [str(executable), str(checkpoint), str(manifest), tokens, str(output), str(cache_mib), manifest_hash]
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-2000:]
        raise VerificationError(f"Ring executable failed ({completed.returncode}): {detail}")
    if not output.is_dir():
        raise VerificationError("Ring executable did not create output directory")

    after = _all_inputs(checkpoint, manifest, executable, repo)
    if before != after:
        raise VerificationError("checkpoint, executable, manifest, or source drifted during run")
    execution = _require_execution(output / "execution.json", token_ids, manifest_hash)
    native = {}
    for name in EXPECTED_NATIVE:
        path = _file(output / name, f"native output {name}")
        native[name] = {"sha256": _sha(path), "size": path.stat().st_size}

    receipt = {
        "schema": "strata-ring-verified-run-v1",
        "status": "verified",
        "checkpoint_id": manifest_hash,
        "checkpoint": before["checkpoint"],
        "original_source": before["original_source"],
        "relevant_source": before["relevant_source"],
        "manifest": before["manifest"],
        "executable": before["executable"],
        "argv": argv,
        "cache_mib": cache_mib,
        "execution": execution,
        "native_outputs": native,
        "scope": {
            "checkpoint_files": "config, index, all indexed safetensors shards, and original model/configuration sources were hashed before and after execution",
            "reference": "this wrapper verifies the native Ring run and does not execute or independently validate the CPU original-reference outputs",
        },
    }
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
    args = parser.parse_args()
    try:
        print(json.dumps(run_verified(args.checkpoint, args.manifest, args.executable, args.tokens, args.output, args.cache_mib), indent=2, sort_keys=True))
    except (OSError, ValueError, VerificationError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
