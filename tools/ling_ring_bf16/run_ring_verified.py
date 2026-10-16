#!/usr/bin/env python3
"""Run the opt-in Ring executable with an auditable checkpoint binding."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path

from prepare_ring_manifest import CheckpointError, prepare


class VerificationError(RuntimeError):
    pass


_CANCEL_GRACE_SECONDS = 5.0


def _terminate_owned(proc: subprocess.Popen) -> None:
    """Stop only the process group created for this invocation."""
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
        # The caller still refuses a receipt; do not touch unrelated services.
        pass


def _run_owned(argv: list[str], timeout: float) -> tuple[int, str, str]:
    # Install handlers before spawning.  A signal can arrive in the Popen
    # window; remember it until the newly owned process exists, then terminate
    # only that process group and refuse the receipt.
    proc = None
    interrupted = {"signal": None}

    def cancel(signum, _frame):
        interrupted["signal"] = signum
        if proc is not None:
            _terminate_owned(proc)

    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    try:
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, start_new_session=True)
        except BaseException:
            # Restore the caller's handlers even when process creation fails.
            raise
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
            raise VerificationError(f"Ring execution timed out after {timeout:g} seconds; no verification receipt was written")
    finally:
        signal.signal(signal.SIGINT, previous[signal.SIGINT])
        signal.signal(signal.SIGTERM, previous[signal.SIGTERM])
    if interrupted["signal"] is not None:
        name = signal.Signals(interrupted["signal"]).name
        raise VerificationError(f"Ring execution cancelled by {name}; no verification receipt was written")
    return proc.returncode, stdout, stderr


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
        repo / "tools/ling_ring_bf16/generate_ring.py",
    ]
    for path in paths:
        _file(path, "relevant source")
        rel = str(path.relative_to(repo))
        relevant[rel] = {"sha256": _sha(path), "size": path.stat().st_size}
    return original, relevant


def _all_inputs(checkpoint: Path, manifest: Path, executable: Path, repo: Path, binding_files=()) -> dict:
    checkpoint_files, _ = _checkpoint_files(checkpoint)
    original, relevant = _source_files(checkpoint, repo)
    frontend = {}
    for path in binding_files:
        path = _file(Path(path), "frontend binding")
        try:
            name = str(path.relative_to(checkpoint))
        except ValueError as exc:
            raise VerificationError("frontend binding must be inside checkpoint") from exc
        frontend[name] = {"path": str(path), "sha256": _sha(path), "size": path.stat().st_size}
    return {
        "checkpoint": checkpoint_files,
        "original_source": original,
        "relevant_source": relevant,
        "manifest": {"sha256": _sha(_file(manifest, "manifest")), "size": manifest.stat().st_size},
        "executable": {"sha256": _sha(_file(executable, "executable")), "size": executable.stat().st_size},
        "frontend": frontend,
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


def _require_execution(path: Path, token_ids: list[int], checkpoint_id: str, generate: int = 0) -> dict:
    execution = _read_json(path, "execution receipt")
    required = {
        "status": "executed",
        "model": "Ring-mini-2.0",
        "activation_dtype": "FP32",
        "quantized": False,
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
    evaluated = execution.get("token_ids")
    if generate:
        if execution.get("logits_mode") != "all" or execution.get("logits_tokens") != len(execution.get("token_ids", [])):
            raise VerificationError("generation execution must retain all logits")
        if not isinstance(evaluated, list) or evaluated[:len(token_ids)] != token_ids or not len(token_ids) <= len(evaluated) <= len(token_ids) + generate:
            raise VerificationError("generated execution token trace is invalid")
        if execution["tokens"] != len(evaluated):
            raise VerificationError("generated execution token count mismatch")
    elif execution.get("tokens") != len(token_ids) or evaluated != token_ids:
        raise VerificationError("execution token identity mismatch")
    return execution


def _parse_eos(text: str) -> list[int]:
    parts = text.split(",")
    try:
        values = [int(part, 10) for part in parts]
    except ValueError as exc:
        raise VerificationError("EOS IDs must be decimal integers") from exc
    if not values or any(type(value) is not int or value < 0 or value >= 157184 for value in values) or len(set(values)) != len(values):
        raise VerificationError("EOS IDs must be unique vocabulary IDs")
    return values


def _require_generation(path: Path, token_ids: list[int], max_new: int, eos_ids: list[int]) -> dict:
    generation = _read_json(path, "generation receipt")
    prompt = generation.get("prompt_token_ids")
    generated = generation.get("generated_token_ids")
    if prompt != token_ids or generation.get("eos_token_ids") != eos_ids or generation.get("max_new_tokens") != max_new or not isinstance(generated, list) or any(type(x) is not int or x < 0 or x >= 157184 for x in generated) or not generated or len(generated) > max_new:
        raise VerificationError("generation token trace is invalid")
    if any(x in eos_ids for x in generated[:-1]):
        raise VerificationError("generation continued after EOS")
    reason = generation.get("finish_reason")
    if reason not in {"eos", "length"}:
        raise VerificationError("generation finish_reason is invalid")
    if reason == "eos" and (not generated or generated[-1] not in eos_ids):
        raise VerificationError("EOS generation trace does not end in a configured EOS token")
    if reason == "length" and (len(generated) != max_new or generated[-1] in eos_ids):
        raise VerificationError("length generation trace does not consume max_new_tokens")
    return generation


def run_verified(checkpoint: Path, manifest: Path, executable: Path, tokens: str, output: Path, cache_mib: int,
                 generate: int = 0, eos_ids=None, logits_mode: str = "all", binding_files=(), timeout: float = 1800) -> dict:
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
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 86400:
        raise VerificationError("timeout must be between 0 and 86400 seconds")
    if generate < 0 or generate > 4096 - len(token_ids):
        raise VerificationError("generate must fit within the 4096-token evaluation window")
    if logits_mode not in {"all", "last"}:
        raise VerificationError("logits_mode must be all or last")
    if generate and logits_mode != "all":
        raise VerificationError("generation requires logits_mode all")
    if not generate:
        parsed_eos = []
    elif isinstance(eos_ids, str):
        parsed_eos = _parse_eos(eos_ids)
    else:
        parsed_eos = list(eos_ids or [])
        if not parsed_eos or any(type(value) is not int or value < 0 or value >= 157184 for value in parsed_eos) or len(set(parsed_eos)) != len(parsed_eos):
            raise VerificationError("EOS IDs must be unique vocabulary IDs")
    if generate and not parsed_eos:
        raise VerificationError("EOS IDs are mandatory for generation")
    if generate and not {Path(path).name for path in binding_files} >= {"tokenizer.json", "tokenizer_config.json", "generation_config.json"}:
        raise VerificationError("generation requires tokenizer and generation-config bindings")
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
    before = _all_inputs(checkpoint, manifest, executable, repo, binding_files)
    argv = [str(executable), str(checkpoint), str(manifest), tokens, str(output), str(cache_mib), manifest_hash]
    if generate:
        argv += ["--generate", str(generate), "--eos", ",".join(map(str, parsed_eos)), "--logits-mode", logits_mode]
    elif logits_mode != "all":
        argv += ["--logits-mode", logits_mode]
    returncode, _stdout, stderr = _run_owned(argv, timeout)
    if returncode != 0:
        detail = stderr.strip()[-2000:]
        raise VerificationError(f"Ring executable failed ({returncode}): {detail}")
    if not output.is_dir():
        raise VerificationError("Ring executable did not create output directory")

    after = _all_inputs(checkpoint, manifest, executable, repo, binding_files)
    if before != after:
        raise VerificationError("checkpoint, executable, manifest, or source drifted during run")
    execution = _require_execution(output / "execution.json", token_ids, manifest_hash, generate)
    native = {}
    expected_native = EXPECTED_NATIVE + (("generation.json",) if generate else ())
    for name in expected_native:
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
        "frontend": before["frontend"],
        "argv": argv,
        "cache_mib": cache_mib,
        "execution": execution,
        "native_outputs": native,
        "scope": {
            "checkpoint_files": "config, index, all indexed safetensors shards, and original model/configuration sources were hashed before and after execution",
            "reference": "this wrapper verifies the native Ring run and does not execute or independently validate the CPU original-reference outputs",
        },
    }
    if generate:
        receipt["generation"] = _require_generation(output / "generation.json", token_ids, generate, parsed_eos)
        expected_evaluated = token_ids + receipt["generation"]["generated_token_ids"][:-1]
        if execution["token_ids"] != expected_evaluated:
            raise VerificationError("execution token trace must omit only the last sampled token")
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
    parser.add_argument("--generate", type=int, default=0)
    parser.add_argument("--eos", help="comma-separated EOS IDs; required with --generate")
    parser.add_argument("--logits-mode", choices=("all", "last"), default="all")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--bind-file", action="append", type=Path, default=[],
                        help="checkpoint-local frontend file to hash and bind; repeatable")
    args = parser.parse_args()
    try:
        print(json.dumps(run_verified(args.checkpoint, args.manifest, args.executable, args.tokens, args.output, args.cache_mib,
                                     args.generate, args.eos, args.logits_mode, binding_files=args.bind_file,
                                     timeout=args.timeout), indent=2, sort_keys=True))
    except (OSError, ValueError, VerificationError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
