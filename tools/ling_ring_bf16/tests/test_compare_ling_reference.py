import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import compare_ling_reference as c


def _meta(path, field="size"):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), field: path.stat().st_size}


def fixture(tmp_path):
    ref, cand = tmp_path / "ref", tmp_path / "cand"
    ref.mkdir(parents=True); cand.mkdir()
    t = 1
    hidden = np.zeros((c.LAYERS, t, c.HIDDEN), dtype=np.float32)
    final = np.ones((t, c.HIDDEN), dtype=np.float32)
    logits = np.zeros((t, c.VOCAB), dtype=np.float32); logits[:, 0] = 2
    hidden.tofile(ref / "hidden_layers.fp32"); final.tofile(ref / "final_hidden.fp32"); logits.tofile(ref / "logits.fp32")
    layers = []
    for layer in range(c.LAYERS):
        if layer == 0:
            layers.append({"layer": layer, "router_ids": [], "router_weights": []})
        else:
            layers.append({"layer": layer, "router_ids": [list(range(8))], "router_weights": [[0.1] * 8]})
    (ref / "layers.json").write_text(json.dumps(layers)); (ref / "router_meta.json").write_text("[]")
    for i in range(c.LAYERS): hidden[i].tofile(cand / f"layer-{i}.f32")
    final.tofile(cand / "final-hidden.f32"); logits.tofile(cand / "logits.f32")
    rows = [f"0\t{layer}\t" + "\t".join(f"{i}:0.1" for i in range(8)) for layer in range(1, c.LAYERS)]
    (cand / "routes.tsv").write_text("\n".join(rows) + "\n")
    execution = {"status": "executed", "model": c.MODEL, "activation_dtype": "FP32", "quantized": False,
                 "tokens": 1, "token_ids": [1], "layers": c.LAYERS, "checkpoint_id": "a" * 64,
                 "logits_mode": "all", "logits_tokens": 1}
    (cand / "execution.json").write_text(json.dumps(execution))
    source_model = {"sha256": "f" * 64, "size": 1}
    source_config = {"sha256": "e" * 64, "size": 1}
    shards = {f"s{i}": {"sha256": f"{i:064x}", "size": 1} for i in range(32)}
    checkpoint = {"config": {"sha256": "b" * 64, "size": 1}, "index": {"sha256": "c" * 64, "size": 1}, "shards": shards}
    report_outputs = {name: _meta(ref / name, "bytes") for name in ("hidden_layers.fp32", "final_hidden.fp32", "logits.fp32", "layers.json", "router_meta.json")}
    report = {"status": "executed", "model": c.MODEL, "activation_dtype": "FP32", "quantized": False, "layers": c.LAYERS, "hidden_size": c.HIDDEN, "vocab_size": c.VOCAB, "reference": "original Ling-3.0-tiny checkpoint implementation",
              "hidden_layers_shape": [c.LAYERS, 1, c.HIDDEN], "logits_shape": [1, c.VOCAB], "tokens": [1],
              "config": {"num_hidden_layers": c.LAYERS, "hidden_size": c.HIDDEN, "vocab_size": c.VOCAB,
                         "num_experts": c.EXPERTS, "num_experts_per_tok": c.TOPK}, "checkpoint": checkpoint,
              "source": {"sha256": source_model["sha256"], "size": 1, "configuration_sha256": source_config["sha256"]},
              "outputs": report_outputs}
    (ref / "report.json").write_text(json.dumps(report))
    native = {name: _meta(cand / name) for name in c.NATIVE_OUTPUTS}
    prov = {"schema": "strata-ling-verified-run-v1", "status": "verified", "model": c.MODEL,
            "checkpoint_id": "a" * 64, "checkpoint": checkpoint, "original_source": {
                "modeling_bailing_moe_v3.py": source_model, "configuration_bailing_moe_v3.py": source_config},
            "relevant_source": {"src/ling/engine.cu": {"sha256": "d" * 64, "size": 1}},
            "manifest": {"sha256": "a" * 64, "size": 1}, "executable": {"sha256": "1" * 64, "size": 1},
            "execution": execution, "native_outputs": native}
    (cand / "provenance.json").write_text(json.dumps(prov))
    return ref, cand


def test_ling_comparison_passes(tmp_path):
    ref, cand = fixture(tmp_path)
    result = c.compare(ref, cand, tmp_path / "result")
    assert result["passed"] is True
    assert result["route_count"] == 23
    assert result["top1_exact"] is True


def test_route_mismatch_retains_numeric_diagnostics(tmp_path):
    ref, cand = fixture(tmp_path)
    lines = (cand / "routes.tsv").read_text().splitlines()
    lines[0] = lines[0].replace("0:0.1", "127:0.1", 1)
    (cand / "routes.tsv").write_text("\n".join(lines) + "\n")
    # Rebind the changed native output hash while preserving the valid receipt.
    p = json.loads((cand / "provenance.json").read_text()); p["native_outputs"]["routes.tsv"] = _meta(cand / "routes.tsv"); (cand / "provenance.json").write_text(json.dumps(p))
    out = tmp_path / "result"
    with pytest.raises(c.ComparisonError, match="comparison failed"):
        c.compare(ref, cand, out)
    result = json.loads((out / "result.json").read_text())
    assert result["passed"] is False
    assert result["route_mismatch_count"] == 1
    assert "hidden_layers_max_abs" in result["max_errors"]
    assert "logits_max_abs" in result["max_errors"]
    assert result["top1_exact"] is True


def test_wrong_reference_identity_and_missing_hash_fail(tmp_path):
    ref, cand = fixture(tmp_path)
    report = json.loads((ref / "report.json").read_text()); report["config"]["hidden_size"] = 2048; (ref / "report.json").write_text(json.dumps(report))
    with pytest.raises(c.ComparisonError, match="identity"):
        c.compare(ref, cand, tmp_path / "bad")
    ref, cand = fixture(tmp_path / "again")
    report = json.loads((ref / "report.json").read_text()); del report["outputs"]["logits.fp32"]; (ref / "report.json").write_text(json.dumps(report))
    with pytest.raises(c.ComparisonError, match="incomplete"):
        c.compare(ref, cand, tmp_path / "bad2")


def add_generation(cand, token=0):
    value = {"schema": "strata-ling-generation-v1", "prompt_token_ids": [1],
             "generated_token_ids": [token], "max_new_tokens": 1,
             "eos_token_ids": [token], "finish_reason": "eos"}
    (cand / "generation.json").write_text(json.dumps(value))
    reference = cand.parent / "ref" / "report.json"
    report = json.loads(reference.read_text())
    report.update({"decoding_mode": "cached", "logits_mode": "all", "logits_tokens": 1,
                   "cache_metadata": {"cache_type": "DynamicCache", "prefill_tokens": 0,
                                      "decode_tokens": 1, "tokenwise": True, "layer_major": True}})
    reference.write_text(json.dumps(report))
    prov = json.loads((cand / "provenance.json").read_text())
    prov["generation"] = value
    prov["frontend"] = {name: {"sha256": "9" * 64, "size": 1} for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json")}
    prov["native_outputs"]["generation.json"] = _meta(cand / "generation.json")
    (cand / "provenance.json").write_text(json.dumps(prov))


def test_generated_answer_requires_original_greedy_choice(tmp_path):
    ref, cand = fixture(tmp_path)
    add_generation(cand)
    result = c.compare(ref, cand, tmp_path / "good")
    assert result["generation"] == {"greedy_choices_exact": True, "prompt_tokens": 1, "generated_tokens": 1, "finish_reason": "eos"}
    ref, cand = fixture(tmp_path / "wrong")
    add_generation(cand, token=2)
    with pytest.raises(c.ComparisonError, match="original greedy"):
        c.compare(ref, cand, tmp_path / "wrong-choice")


@pytest.mark.parametrize("change, message", [
    (lambda report: report.pop("decoding_mode"), "cached original reference"),
    (lambda report: report.update({"decoding_mode": "full"}), "cached original reference"),
    (lambda report: report.pop("cache_metadata"), "cache metadata is missing"),
    (lambda report: report["cache_metadata"].update({"tokenwise": False}), "cache metadata mismatch"),
    (lambda report: report.update({"logits_mode": "last"}), "all logits rows"),
    (lambda report: report.update({"logits_tokens": 0}), "all logits rows"),
])
def test_generation_requires_cached_reference_receipt(tmp_path, change, message):
    ref, cand = fixture(tmp_path)
    add_generation(cand)
    report_path = ref / "report.json"
    report = json.loads(report_path.read_text())
    change(report)
    report_path.write_text(json.dumps(report))
    with pytest.raises(c.ComparisonError, match=message):
        c.compare(ref, cand, tmp_path / "bad")


def test_generation_receipt_cannot_be_changed_or_removed(tmp_path):
    ref, cand = fixture(tmp_path)
    add_generation(cand)
    (cand / "generation.json").write_text((cand / "generation.json").read_text() + " ")
    with pytest.raises(c.ComparisonError, match="hash mismatch"):
        c.compare(ref, cand, tmp_path / "changed")
    (cand / "generation.json").unlink()
    with pytest.raises(c.ComparisonError, match="provenance disagree"):
        c.compare(ref, cand, tmp_path / "removed")


def test_duplicate_receipt_keys_fail(tmp_path):
    ref, cand = fixture(tmp_path)
    report = ref / "report.json"
    report.write_text(report.read_text().replace('"status": "executed"', '"status": "rejected", "status": "executed"', 1))
    with pytest.raises(c.ComparisonError, match="duplicate JSON key"):
        c.compare(ref, cand, tmp_path / "duplicate")
