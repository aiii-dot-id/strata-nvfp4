#!/usr/bin/env python3
"""Compare a verified native Ling forward against an original reference.

This is an internal FP32 widened-BF16 trace comparator.  It does not claim
activation or production quality.  Attribution: AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

MODEL = "Ling-3.0-tiny"
LAYERS = 24
HIDDEN = 1536
VOCAB = 157184
EXPERTS = 128
TOPK = 8
ATOL = 0.001
RTOL = 0.001
WATOL = 1e-5
WRTOL = 1e-4
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ROUTE_LAYERS = tuple(range(1, LAYERS))
NATIVE_OUTPUTS = {"routes.tsv", "final-hidden.f32", "logits.f32", "execution.json"} | {f"layer-{i}.f32" for i in range(LAYERS)}


class ComparisonError(ValueError):
    pass


def fail(message: str):
    raise ComparisonError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail("duplicate JSON key: " + key)
        result[key] = value
    return result


def _json(path: Path, label: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"invalid {label}: {exc}")


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _pin(meta, label: str, require_size: bool = True) -> str:
    if not isinstance(meta, dict) or not isinstance(meta.get("sha256"), str) or not HEX64.fullmatch(meta["sha256"]):
        fail(f"invalid provenance hash: {label}")
    if require_size and (type(meta.get("size")) is not int or meta["size"] <= 0):
        fail(f"invalid provenance size: {label}")
    return meta["sha256"]


def load(path: Path, count: int, label: str) -> np.ndarray:
    if not path.is_file() or path.stat().st_size != count * 4:
        fail(f"invalid or trailing output bytes: {label}")
    values = np.fromfile(path, dtype=np.float32)
    if values.size != count or not np.isfinite(values).all():
        fail(f"invalid, short, or nonfinite output: {label}")
    return values


def _reference_identity(report: dict) -> None:
    if report.get("status") != "executed":
        fail("reference report is not an executed receipt")
    cfg = report.get("config")
    if not isinstance(cfg, dict):
        fail("reference Ling config is missing")
    expected = {"num_hidden_layers": LAYERS, "hidden_size": HIDDEN, "vocab_size": VOCAB,
                "num_experts": EXPERTS, "num_experts_per_tok": TOPK}
    for key, value in expected.items():
        if cfg.get(key) != value:
            fail(f"reference Ling identity mismatch: {key}")
    receipt_fields = {"activation_dtype": "FP32", "quantized": False, "layers": LAYERS,
                      "hidden_size": HIDDEN, "vocab_size": VOCAB}
    for key, value in receipt_fields.items():
        if report.get(key) != value:
            fail(f"reference Ling receipt mismatch: {key}")
    if type(report.get("quantized")) is not bool:
        fail("reference quantized field must be boolean")
    model = report.get("model")
    reference = report.get("reference")
    if model != MODEL and not (isinstance(reference, str) and "Ling" in reference and "original" in reference.lower()):
        fail("reference report is not identified as original Ling-3.0-tiny")


def verify_reference_hashes(root: Path, report: dict) -> None:
    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        fail("reference output hashes are missing")
    expected = {"final_hidden.fp32", "logits.fp32", "hidden_layers.fp32", "layers.json", "router_meta.json"}
    if set(outputs) != expected:
        fail("reference output hashes are incomplete")
    for name in expected:
        meta = outputs.get(name)
        if not isinstance(meta, dict) or not isinstance(meta.get("sha256"), str) or not HEX64.fullmatch(meta["sha256"]):
            fail(f"invalid reference output hash: {name}")
        path = root / name
        if not path.is_file() or _hash(path) != meta["sha256"] or type(meta.get("bytes")) is not int or meta["bytes"] != path.stat().st_size:
            fail(f"reference output hash mismatch: {name}")


def _reference_source(report: dict) -> tuple[dict, str, str]:
    source = report.get("source")
    if not isinstance(source, dict):
        fail("reference source provenance is missing")
    modeling = source.get("modeling") if isinstance(source.get("modeling"), dict) else source
    modeling_sha = _pin(modeling, "reference modeling source", False)
    config_sha = source.get("configuration_sha256")
    if isinstance(source.get("configuration"), dict):
        config_sha = source["configuration"].get("sha256")
    if not isinstance(config_sha, str) or not HEX64.fullmatch(config_sha):
        fail("reference configuration source hash is missing")
    return source, modeling_sha, config_sha


def verify_provenance(candidate: Path, execution: dict, token_ids: list[int], report: dict) -> None:
    path = candidate / "provenance.json"
    if not path.is_file():
        fail("missing candidate provenance.json")
    provenance = _json(path, "candidate provenance")
    if provenance.get("schema") != "strata-ling-verified-run-v1" or provenance.get("status") != "verified" or provenance.get("model") != MODEL:
        fail("candidate provenance is not a successful verified Ling receipt")
    expected = {"status": "executed", "model": MODEL, "activation_dtype": "FP32", "quantized": False,
                "tokens": len(token_ids), "layers": LAYERS, "token_ids": token_ids}
    for key, value in expected.items():
        if execution.get(key) != value:
            fail(f"candidate execution receipt mismatch: {key}")
    if type(execution.get("quantized")) is not bool or execution["quantized"] is not False:
        fail("candidate execution receipt mismatch: quantized")
    if execution.get("logits_mode") != "all" or execution.get("logits_tokens") != len(token_ids):
        fail("candidate execution must contain all logits rows")
    checkpoint_id = execution.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not HEX64.fullmatch(checkpoint_id) or checkpoint_id != provenance.get("checkpoint_id"):
        fail("candidate checkpoint identity is not bound")
    if provenance.get("execution") != execution:
        fail("provenance execution receipt differs from execution.json")
    manifest = provenance.get("manifest")
    if checkpoint_id != (manifest or {}).get("sha256"):
        fail("candidate checkpoint identity is not the manifest digest")
    _pin(manifest, "candidate manifest")
    _pin(provenance.get("executable"), "candidate executable")
    for kind in ("original_source", "relevant_source"):
        values = provenance.get(kind)
        if not isinstance(values, dict) or not values:
            fail(f"candidate {kind} provenance is missing")
        for name, meta in values.items():
            _pin(meta, f"candidate {kind} {name}")
    checkpoint = provenance.get("checkpoint")
    reference_checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict) or not isinstance(reference_checkpoint, dict):
        fail("checkpoint provenance is missing")
    cfg = _pin(checkpoint.get("config"), "candidate config")
    idx = _pin(checkpoint.get("index"), "candidate index")
    rcfg = _pin(reference_checkpoint.get("config"), "reference config")
    ridx = _pin(reference_checkpoint.get("index"), "reference index")
    if cfg != rcfg or idx != ridx:
        fail("candidate config/index hash does not match reference")
    if checkpoint["config"]["size"] != reference_checkpoint["config"]["size"] or checkpoint["index"]["size"] != reference_checkpoint["index"]["size"]:
        fail("candidate config/index size does not match reference")
    shards = checkpoint.get("shards")
    reference_shards = reference_checkpoint.get("shards")
    if not isinstance(shards, dict) or len(shards) != 32 or not isinstance(reference_shards, dict) or len(reference_shards) != 32 or set(shards) != set(reference_shards):
        fail("candidate/reference shard set mismatch")
    for name in shards:
        if _pin(shards[name], f"candidate shard {name}") != _pin(reference_shards[name], f"reference shard {name}"):
            fail(f"candidate shard hash does not match reference: {name}")
        if shards[name]["size"] != reference_shards[name]["size"]:
            fail(f"candidate shard size does not match reference: {name}")
    _, reference_modeling, reference_configuration = _reference_source(report)
    original = provenance.get("original_source")
    if _pin(original.get("modeling_bailing_moe_v3.py"), "candidate modeling source") != reference_modeling:
        fail("candidate original modeling source does not match reference")
    config_meta = original.get("configuration_bailing_moe_v3.py")
    if _pin(config_meta, "candidate configuration source") != reference_configuration:
        fail("candidate original configuration source does not match reference")
    generated = (candidate / "generation.json").is_file()
    if generated != ("generation" in provenance):
        fail("generation receipt and provenance disagree")
    if generated:
        frontend = provenance.get("frontend")
        if not isinstance(frontend, dict):
            fail("generation frontend bindings are missing")
        for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            _pin(frontend.get(name), "generation frontend " + name)
        if provenance["generation"] != _json(candidate / "generation.json", "generation receipt"):
            fail("generation receipt differs from provenance")
    native = provenance.get("native_outputs")
    expected_native = NATIVE_OUTPUTS | ({"generation.json"} if generated else set())
    if not isinstance(native, dict) or set(native) != expected_native:
        fail("candidate native output hashes are incomplete")
    for name in expected_native:
        meta = native.get(name)
        path = candidate / name
        if not path.is_file():
            fail(f"candidate output is missing: {name}")
        if not isinstance(meta, dict) or _pin(meta, f"candidate output {name}") != _hash(path) or meta["size"] != path.stat().st_size:
            fail(f"candidate output hash mismatch: {name}")


def verify_generation(candidate: Path, token_ids: list[int], ref_logits: np.ndarray) -> dict | None:
    path = candidate / "generation.json"
    if not path.is_file():
        return None
    value = _json(path, "generation receipt")
    if value.get("schema") != "strata-ling-generation-v1":
        fail("invalid Ling generation schema")
    prompt, generated = value.get("prompt_token_ids"), value.get("generated_token_ids")
    for name, ids in (("prompt", prompt), ("generated", generated)):
        if not isinstance(ids, list) or not ids or any(type(x) is not int or not 0 <= x < VOCAB for x in ids):
            fail("invalid generation " + name + " IDs")
    maximum, eos, reason = value.get("max_new_tokens"), value.get("eos_token_ids"), value.get("finish_reason")
    if type(maximum) is not int or maximum < 1 or len(prompt) + maximum > 1024 or len(generated) > maximum:
        fail("invalid generation budget")
    if not isinstance(eos, list) or not eos or any(type(x) is not int or not 0 <= x < VOCAB for x in eos) or len(set(eos)) != len(eos):
        fail("invalid generation EOS IDs")
    if any(x in eos for x in generated[:-1]):
        fail("generation continued after EOS")
    if reason == "eos":
        if generated[-1] not in eos:
            fail("generation did not stop on EOS")
    elif reason == "length":
        if len(generated) != maximum or generated[-1] in eos:
            fail("generation did not consume its budget")
    else:
        fail("invalid generation finish reason")
    if token_ids != prompt + generated[:-1]:
        fail("generation trace differs from evaluated tokens")
    start = len(prompt) - 1
    expected = np.argmax(ref_logits[start:start + len(generated)], axis=1).astype(int).tolist()
    if generated != expected:
        fail("generation differs from original greedy cached trace")
    return {"greedy_choices_exact": True, "prompt_tokens": len(prompt),
            "generated_tokens": len(generated), "finish_reason": reason}


def _require_cached_reference(report: dict, token_count: int) -> None:
    """Generation qualification requires the original tokenwise cache trace."""
    if report.get("decoding_mode") != "cached":
        fail("generation requires a cached original reference report")
    metadata = report.get("cache_metadata")
    if not isinstance(metadata, dict):
        fail("generation reference cache metadata is missing")
    expected = {
        "cache_type": "DynamicCache",
        "prefill_tokens": 0,
        "decode_tokens": token_count,
        "tokenwise": True,
        "layer_major": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            fail(f"generation reference cache metadata mismatch: {key}")
    if report.get("logits_mode") != "all" or report.get("logits_tokens") != token_count:
        fail("generation reference must contain all logits rows")


def _strict_ids(ids) -> bool:
    return isinstance(ids, list) and len(ids) == TOPK and all(type(x) is int and 0 <= x < EXPERTS for x in ids) and len(set(ids)) == TOPK


def routes_reference(path: Path, tokens: int) -> dict:
    rows = _json(path, "reference layers")
    if not isinstance(rows, list) or len(rows) != LAYERS:
        fail("invalid reference route layers")
    result = {}
    for layer, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("layer") != layer:
            fail("reference layer identity mismatch")
        ids, weights = row.get("router_ids"), row.get("router_weights")
        if layer not in ROUTE_LAYERS:
            if ids != [] or weights != []:
                fail("non-MoE Ling layer has routes")
            continue
        if not isinstance(ids, list) or not isinstance(weights, list) or len(ids) != tokens or len(weights) != tokens:
            fail("invalid reference route rows")
        for pos in range(tokens):
            if not _strict_ids(ids[pos]) or not isinstance(weights[pos], list) or len(weights[pos]) != TOPK or not all(math.isfinite(float(x)) for x in weights[pos]):
                fail("invalid reference route values")
            result[(pos, layer)] = (ids[pos], [float(x) for x in weights[pos]])
    return result


def routes_candidate(path: Path, tokens: int) -> dict:
    result = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        fail(f"invalid candidate routes: {exc}")
    for line in lines:
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 2 + TOPK:
            fail("invalid candidate route row")
        try:
            pos, layer = int(fields[0], 10), int(fields[1], 10)
            values = [item.split(":") for item in fields[2:]]
            ids = [int(pair[0], 10) for pair in values]
            weights = [float(pair[1]) for pair in values]
        except (ValueError, IndexError) as exc:
            fail(f"invalid candidate route value: {exc}")
        if (pos, layer) in result or not (0 <= pos < tokens) or layer not in ROUTE_LAYERS:
            fail("invalid candidate route index or duplicate")
        if not _strict_ids(ids) or not all(math.isfinite(x) for x in weights):
            fail("invalid candidate route values")
        result[(pos, layer)] = (ids, weights)
    expected = {(pos, layer) for pos in range(tokens) for layer in ROUTE_LAYERS}
    if set(result) != expected:
        fail("missing candidate route row")
    return result


def _maxerr(a, b) -> float:
    return float(np.max(np.abs(a - b)))


def compare(reference: Path, candidate: Path, output: Path) -> dict:
    reference, candidate, output = Path(reference), Path(candidate), Path(output)
    if output.exists():
        fail("refusing existing output")
    report = _json(reference / "report.json", "reference report")
    _reference_identity(report)
    verify_reference_hashes(reference, report)
    hidden_shape = report.get("hidden_layers_shape")
    tokens = report.get("tokens")
    if not isinstance(tokens, list) or not tokens or any(type(x) is not int or x < 0 or x >= VOCAB for x in tokens):
        fail("invalid reference token trace")
    if hidden_shape != [LAYERS, len(tokens), HIDDEN]:
        fail("reference hidden shape must be [24,tokens,1536]")
    if any(type(x) is not int or x < 0 or x >= VOCAB for x in tokens):
        fail("invalid reference token trace")
    ref_layers = load(reference / "hidden_layers.fp32", LAYERS * len(tokens) * HIDDEN, "reference hidden layers").reshape(LAYERS, len(tokens), HIDDEN)
    ref_final = load(reference / "final_hidden.fp32", len(tokens) * HIDDEN, "reference final hidden").reshape(len(tokens), HIDDEN)
    candidate_layers = np.stack([load(candidate / f"layer-{i}.f32", len(tokens) * HIDDEN, f"candidate layer {i}").reshape(len(tokens), HIDDEN) for i in range(LAYERS)])
    candidate_final = load(candidate / "final-hidden.f32", len(tokens) * HIDDEN, "candidate final hidden").reshape(len(tokens), HIDDEN)
    execution = _json(candidate / "execution.json", "candidate execution")
    verify_provenance(candidate, execution, tokens, report)
    logits_shape = report.get("logits_shape")
    if isinstance(logits_shape, list) and len(logits_shape) == 2:
        logit_tokens, vocab = map(int, logits_shape)
    elif isinstance(logits_shape, list) and len(logits_shape) == 3:
        batch, logit_tokens, vocab = map(int, logits_shape)
        if batch != 1:
            fail("reference logits shape must have batch 1")
    else:
        fail("reference logits shape is missing")
    if logit_tokens != len(tokens) or vocab != VOCAB:
        fail("reference logits shape must be [tokens,157184]")
    ref_logits = load(reference / "logits.fp32", logit_tokens * VOCAB, "reference logits").reshape(logit_tokens, VOCAB)
    candidate_logits = load(candidate / "logits.f32", len(tokens) * VOCAB, "candidate logits").reshape(len(tokens), VOCAB)
    if (candidate / "generation.json").is_file():
        _require_cached_reference(report, len(tokens))
    generation = verify_generation(candidate, tokens, ref_logits)
    route_issues = []
    try:
        reference_routes = routes_reference(reference / "layers.json", len(tokens))
    except ComparisonError as exc:
        reference_routes, route_issues = {}, [str(exc)]
    try:
        candidate_routes = routes_candidate(candidate / "routes.tsv", len(tokens))
    except ComparisonError as exc:
        candidate_routes, route_issues = {}, route_issues + [str(exc)]
    route_pass = bool(reference_routes and candidate_routes and set(reference_routes) == set(candidate_routes))
    route_weight_max = 0.0
    route_mismatch_count = 0
    if reference_routes and candidate_routes:
        for key, (ref_ids, ref_weights) in reference_routes.items():
            if key not in candidate_routes:
                route_pass = False
                route_issues.append(f"missing candidate route at {key}")
                continue
            cand_ids, cand_weights = candidate_routes[key]
            if cand_ids != ref_ids:
                route_pass = False
                route_mismatch_count += 1
                route_issues.append(f"expert route mismatch at position/layer {key}: expected {ref_ids}, got {cand_ids}")
            route_weight_max = max(route_weight_max, max(abs(a - b) for a, b in zip(ref_weights, cand_weights)))
            if any(abs(a - b) > WATOL + WRTOL * abs(a) for a, b in zip(ref_weights, cand_weights)):
                route_pass = False
    top_exact = bool(np.array_equal(np.argmax(ref_logits, axis=1), np.argmax(candidate_logits, axis=1)))
    output_pass = bool(np.all(np.abs(ref_layers - candidate_layers) <= ATOL + RTOL * np.abs(ref_layers)) and
                       np.all(np.abs(ref_final - candidate_final) <= ATOL + RTOL * np.abs(ref_final)) and
                       np.all(np.abs(ref_logits - candidate_logits) <= ATOL + RTOL * np.abs(ref_logits)))
    issues = route_issues[:32]
    if not output_pass:
        issues.append("FP32 output tolerance failed")
    if not top_exact:
        issues.append("logit top1 mismatch")
    passed = bool(output_pass and route_pass and top_exact)
    result = {"schema": "ling-reference-compare-v1", "passed": passed, "model": MODEL,
              "layers": LAYERS, "tokens": len(tokens), "vocab": VOCAB,
              "route_count": len(reference_routes), "route_mismatch_count": route_mismatch_count,
              "top1_exact": top_exact, "max_errors": {"hidden_layers_max_abs": _maxerr(ref_layers, candidate_layers),
              "final_hidden_max_abs": _maxerr(ref_final, candidate_final), "logits_max_abs": _maxerr(ref_logits, candidate_logits),
              "route_weight_max_abs": route_weight_max}, "tolerances": {"output": "0.001 + 0.001 * abs(reference)", "weights": "1e-5 + 1e-4 * abs(reference)"}, "issues": issues}
    if generation is not None:
        result["generation"] = generation
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        fail("comparison failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(compare(Path(args.reference), Path(args.candidate), Path(args.output)), indent=2))
    except Exception as exc:
        out = Path(args.output)
        if not out.exists():
            out.mkdir(parents=True)
            (out / "result.json").write_text(json.dumps({"schema": "ling-reference-compare-v1", "passed": False, "error": str(exc)}, indent=2) + "\n")
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
