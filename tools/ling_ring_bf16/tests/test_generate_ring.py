import json
import sys
from pathlib import Path

import pytest

import generate_ring
from prepare_ring_manifest import EXPECTED
from run_ring_verified import VerificationError, _require_generation


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        assert messages[-1]["role"] == "user"
        return [[1, 2]]

    def decode(self, ids, **kwargs):
        assert kwargs["skip_special_tokens"] is False
        return "answer"


def checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    config = dict(EXPECTED); config["architectures"] = ["BailingMoeV2ForCausalLM"]
    (root / "config.json").write_text(json.dumps(config))
    (root / "generation_config.json").write_text(json.dumps({"eos_token_id": [9, 10]}))
    return root


def test_eos_config_is_required_and_strict(tmp_path):
    root = checkpoint(tmp_path)
    assert generate_ring._eos_ids(root) == [9, 10]
    (root / "generation_config.json").write_text(json.dumps({"eos_token_id": [9, True]}))
    with pytest.raises(VerificationError): generate_ring._eos_ids(root)
    (root / "generation_config.json").unlink()
    with pytest.raises(VerificationError): generate_ring._eos_ids(root)


def test_frontend_binds_template_and_writes_decoded_text(tmp_path, monkeypatch):
    root = checkpoint(tmp_path)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text("{}")
    monkeypatch.setattr(generate_ring, "_tokenizer", lambda _: Tokenizer())
    out = tmp_path / "out"; out.mkdir()
    def fake_run(*args, **kwargs):
        return {"generation": {"prompt_token_ids": [1, 2], "generated_token_ids": [3, 9], "finish_reason": "eos"}, "frontend": {"generation_config.json": {"sha256": "x"}}}
    monkeypatch.setattr(generate_ring, "run_verified", fake_run)
    result = generate_ring.generate(root, tmp_path / "manifest.tsv", Path("/bin/true"), "hello", out, 6, 4)
    assert result["frontend"]["generated_token_ids"] == [3, 9]
    assert (out / "generated.txt").read_text() == "answer"


def test_frontend_forwards_original_rope_frequency(tmp_path, monkeypatch):
    root = checkpoint(tmp_path)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text("{}")
    monkeypatch.setattr(generate_ring, "_tokenizer", lambda _: Tokenizer())
    out = tmp_path / "out"; out.mkdir()
    seen = {}

    def fake_run(*args, **kwargs):
        seen["rope_frequency"] = kwargs["rope_frequency"]
        return {
            "generation": {
                "prompt_token_ids": [1, 2],
                "generated_token_ids": [9],
                "finish_reason": "eos",
            },
            "frontend": {"generation_config.json": {"sha256": "x"}},
        }

    monkeypatch.setattr(generate_ring, "run_verified", fake_run)
    result = generate_ring.generate(
        root, tmp_path / "manifest.tsv", Path("/bin/true"), "hello",
        out, 6, 1, rope_frequency="original")
    assert seen["rope_frequency"] == "original"
    assert result["frontend"]["generated_token_ids"] == [9]


def test_frontend_defaults_and_forwards_matmul_router(tmp_path, monkeypatch):
    root = checkpoint(tmp_path)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text("{}")
    monkeypatch.setattr(generate_ring, "_tokenizer", lambda _: Tokenizer())
    seen = []

    def fake_run(*args, **kwargs):
        seen.append(kwargs)
        return {"generation": {"prompt_token_ids": [1, 2], "generated_token_ids": [9], "finish_reason": "eos"},
                "frontend": {"generation_config.json": {"sha256": "x"}}}

    monkeypatch.setattr(generate_ring, "run_verified", fake_run)
    (tmp_path / "out-default").mkdir()
    result = generate_ring.generate(root, tmp_path / "manifest.tsv", Path("/bin/true"), "hello", tmp_path / "out-default", 6, 1)
    assert seen[-1]["matmul"] == "bf16-gemv" and seen[-1]["router"] == "native"
    assert result["frontend"]["matmul_backend"] == "bf16-gemv"
    (tmp_path / "out-opt").mkdir()
    result = generate_ring.generate(root, tmp_path / "manifest.tsv", Path("/bin/true"), "hello", tmp_path / "out-opt", 6, 1,
                                    matmul="fp32-cublas", router="torch-cuda")
    assert seen[-1]["matmul"] == "fp32-cublas" and seen[-1]["router"] == "torch-cuda"
    assert result["frontend"]["router_backend"] == "torch-cuda"


@pytest.mark.parametrize("field,value", [("matmul", "bad"), ("router", "bad")])
def test_frontend_rejects_invalid_backend_before_tokenizer(tmp_path, monkeypatch, field, value):
    root = checkpoint(tmp_path)
    monkeypatch.setattr(generate_ring, "_tokenizer", lambda _: pytest.fail("tokenizer must not be loaded"))
    kwargs = {field: value}
    with pytest.raises(VerificationError, match=field):
        generate_ring.generate(root, tmp_path / "manifest.tsv", Path("/bin/true"), "hello", tmp_path / "out", 6, 1, **kwargs)


def test_main_forwards_matmul_router_flags(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_generate(*args, **kwargs):
        seen["args"] = args
        seen.update(kwargs)
        return {"frontend": {"decoded": "answer"}}

    monkeypatch.setattr(generate_ring, "generate", fake_generate)
    monkeypatch.setattr(sys, "argv", ["generate_ring.py", "--checkpoint", str(tmp_path / "c"), "--manifest", str(tmp_path / "m"),
                                        "--executable", "/bin/true", "--prompt", "hello", "--output", str(tmp_path / "o"),
                                        "--cache-mib", "6", "--max-new-tokens", "1", "--matmul", "fp32-cublas", "--router", "torch-cuda"])
    assert generate_ring.main() == 0
    assert seen["args"][-2:] == ("fp32-cublas", "torch-cuda")
    assert capsys.readouterr().out == "answer"


def test_frontend_rejects_binding_drift_during_tokenization(tmp_path, monkeypatch):
    root = checkpoint(tmp_path)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text("{}")

    class MutatingTokenizer(Tokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            (root / "tokenizer_config.json").write_text("changed")
            return super().apply_chat_template(messages, tokenize, add_generation_prompt)

    monkeypatch.setattr(generate_ring, "_tokenizer", lambda _: MutatingTokenizer())
    with pytest.raises(VerificationError, match="bindings drifted"):
        generate_ring.generate(root, tmp_path / "manifest.tsv", Path("/bin/true"), "hello", tmp_path / "out", 6, 4)


def test_prompt_budget_rejected(tmp_path, monkeypatch):
    root = checkpoint(tmp_path)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text("{}")
    monkeypatch.setattr(generate_ring, "_tokenizer", lambda _: Tokenizer())
    with pytest.raises(VerificationError, match="exceeds"):
        generate_ring.generate(root, tmp_path / "manifest.tsv", Path("/bin/true"), "hello", tmp_path / "out", 6, 4095)


@pytest.mark.parametrize("trace", [
    {"prompt_token_ids": [1], "generated_token_ids": [9, 10], "eos_token_ids": [9], "max_new_tokens": 2, "finish_reason": "eos"},
    {"prompt_token_ids": [1], "generated_token_ids": [3], "eos_token_ids": [9], "max_new_tokens": 1, "finish_reason": "eos"},
    {"prompt_token_ids": [1], "generated_token_ids": [3], "eos_token_ids": [9], "max_new_tokens": 1, "finish_reason": "length"},
])
def test_generation_eos_and_truncation_controls(tmp_path, trace):
    path = tmp_path / "generation.json"; path.write_text(json.dumps(trace))
    if trace["generated_token_ids"] == [3] and trace["finish_reason"] == "eos":
        with pytest.raises(VerificationError): _require_generation(path, [1], 1, [9])
    elif trace["finish_reason"] == "length":
        assert _require_generation(path, [1], 1, [9])["finish_reason"] == "length"
    else:
        with pytest.raises(VerificationError): _require_generation(path, [1], 1, [9])
