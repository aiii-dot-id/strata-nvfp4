"""CPU orchestration tests for bounded tool-training replay.
AI Identity Incorported <james@aiii.id>.
These tests isolate replay/snapshot orchestration with deterministic mocks; they
do not qualify a full model or native/GPU run.
"""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
import pytest
import torch
import tool_training

POLICY = {"learning_rates": {"rlvr": 0.01, "self_opd": 0.01}, "max_mean_kl": 1.0, "max_max_kl": 2.0, "max_selected_log_ratio": 3.0}

class TinyTrainer(torch.nn.Module):
    model = "ling"
    expert = 3
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor([0.0]))
        self.register_buffer("frozen", torch.tensor([7.0]))

def _setup(monkeypatch, tmp_path):
    trainer = TinyTrainer()
    opt = torch.optim.Adam(trainer.parameters(), lr=0.01, foreach=False)
    cp = tmp_path / "checkpoint"; cp.mkdir()
    (cp / "weights.safetensors").write_bytes(b"checkpoint")
    adapter = tmp_path / "parent.adapter"; adapter.write_bytes(b"parent")
    recdir = tmp_path / "row"; recdir.mkdir()
    (recdir / "ROLLOUT.json").write_text("{}")
    records = [{"path": recdir, "native_stages": [{"stage": "s", "rows": 1}], "policy_adapter_sha256":hashlib.sha256(b"parent").hexdigest()}]
    monkeypatch.setattr(tool_training.native, "MODELS", {"ling": {"checkpoint": cp}})
    monkeypatch.setattr(tool_training, "release", lambda t: None)
    states = {}
    def save_state(t, o, path, **kwargs):
        states[str(path)] = {"params": {k: v.detach().clone() for k, v in t.named_parameters()}, "opt": copy.deepcopy(o.state_dict()), "rng": torch.get_rng_state().clone(), "meta": kwargs}
        Path(path).mkdir(parents=True, exist_ok=True); (Path(path) / "snapshot.marker").write_text("cpu-test")
    def restore_state(t, o, path, **kwargs):
        s = states[str(path)]
        with torch.no_grad():
            for k, v in t.named_parameters(): v.copy_(s["params"][k])
        o.load_state_dict(copy.deepcopy(s["opt"])); torch.set_rng_state(s["rng"].clone())
    monkeypatch.setattr(tool_training.terminal, "save_training_state", save_state)
    monkeypatch.setattr(tool_training.terminal, "restore_training_state", restore_state)
    def save_adapter(t, path, checkpoint, **kwargs): Path(path).write_text(json.dumps({"p": float(t.p.detach())}, sort_keys=True))
    def load_adapter(t, path, **kwargs):
        with torch.no_grad(): t.p.fill_(json.loads(Path(path).read_text())["p"])
    monkeypatch.setattr(tool_training.terminal, "save_adapter", save_adapter)
    monkeypatch.setattr(tool_training.terminal, "load_adapter", load_adapter)
    monkeypatch.setattr(tool_training.native, "sha", lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    monkeypatch.setattr(tool_training.native, "dump", lambda path, value: Path(path).write_text(json.dumps(value, sort_keys=True)))
    return trainer, opt, records, adapter

def _decision(accepted=True, marker="same"):
    update = {"loss": 1.25, "gradient": {"p": {"shape": [1], "dtype": "torch.float32", "sha256": "grad"}}, "parameters": {"p": {"shape": [1], "dtype": "torch.float32", "sha256": marker}}}
    return {"accepted": accepted, "reason": "accepted" if accepted else "bounds", "update": update if accepted else {}, "metrics": {"mean_kl": 0.1, "max_kl": 0.2, "selected_log_ratio": 0.0}, "bounds": {"max_mean_kl": 1.0}, "restored": True}

def _run(monkeypatch, trainer, opt, records, adapter, output, fn):
    calls = []
    def mocked(*args, **kwargs):
        calls.append(len(calls)); return fn(len(calls))
    monkeypatch.setattr(tool_training, "perform_update", mocked)
    return tool_training.replay_qualified_update(trainer, opt, records, "rlvr", policy=POLICY, output=output, round_no=4, steps_before=2, current_adapter=adapter, checkpoint_sha256="cp-sha", campaign_plan_sha256="plan-sha", device="cpu"), calls

def test_accepted_update_replay_and_export_reload_are_exact(monkeypatch, tmp_path):
    trainer, opt, records, adapter = _setup(monkeypatch, tmp_path)
    def accepted(n):
        with torch.no_grad(): trainer.p.add_(1.0)
        return _decision(True)
    result, calls = _run(monkeypatch, trainer, opt, records, adapter, tmp_path / "accepted", accepted)
    assert result["accepted"] is True and calls == [0, 1] and result["steps"] == 3
    assert (tmp_path / "accepted" / "candidate.adapter").exists() and float(trainer.p) == 1.0
    assert json.loads((tmp_path / "accepted" / "QUALIFIED.json").read_text())["export_reload_exact"]

def test_rejected_update_never_exports(monkeypatch, tmp_path):
    trainer, opt, records, adapter = _setup(monkeypatch, tmp_path)
    result, calls = _run(monkeypatch, trainer, opt, records, adapter, tmp_path / "rejected", lambda n: _decision(False))
    assert result["accepted"] is False and result["steps"] == 2 and calls == [0]
    assert not (tmp_path / "rejected" / "candidate.adapter").exists() and float(trainer.p) == 0.0

def test_altered_replay_is_rejected(monkeypatch, tmp_path):
    trainer, opt, records, adapter = _setup(monkeypatch, tmp_path)
    def altered(n):
        with torch.no_grad(): trainer.p.add_(1.0)
        return _decision(True, "first" if n == 1 else "altered")
    with pytest.raises(ValueError, match="tree scalar"):
        _run(monkeypatch, trainer, opt, records, adapter, tmp_path / "altered", altered)
    assert not (tmp_path / "altered" / "candidate.adapter").exists()

def test_source_buffer_mutation_is_rejected(monkeypatch, tmp_path):
    trainer, opt, records, adapter = _setup(monkeypatch, tmp_path)
    def mutate(n):
        with torch.no_grad(): trainer.p.add_(1.0); trainer.frozen.add_(1.0)
        return _decision(True)
    with pytest.raises(ValueError, match="frozen buffer mutation"):
        _run(monkeypatch, trainer, opt, records, adapter, tmp_path / "buffer", mutate)
    assert not (tmp_path / "buffer" / "candidate.adapter").exists()

def test_export_reload_mismatch_is_rejected(monkeypatch, tmp_path):
    trainer, opt, records, adapter = _setup(monkeypatch, tmp_path)
    def wrong_load(t, path, **kwargs):
        with torch.no_grad(): t.p.zero_()
    monkeypatch.setattr(tool_training.terminal, "load_adapter", wrong_load)
    with pytest.raises(ValueError, match="export/reload changed"):
        _run(monkeypatch, trainer, opt, records, adapter, tmp_path / "mismatch", lambda n: (trainer.p.detach().add_(1.0), _decision(True))[1])
    assert not (tmp_path / "mismatch" / "QUALIFIED.json").exists()

def test_current_adapter_must_be_real_parent_for_receipt(monkeypatch, tmp_path):
    trainer, opt, records, _ = _setup(monkeypatch, tmp_path)
    with pytest.raises((TypeError, AttributeError)):
        _run(monkeypatch, trainer, opt, records, None, tmp_path / "none", lambda n: _decision(False))


def test_optimizer_policy_learning_rate_mismatch_rejects_before_update(monkeypatch, tmp_path):
    trainer, opt, records, adapter = _setup(monkeypatch, tmp_path)
    for group in opt.param_groups:
        group["lr"] = 0.02
    calls = []
    monkeypatch.setattr(tool_training, "perform_update",
                        lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match="optimizer/policy learning rate mismatch"):
        tool_training.replay_qualified_update(
            trainer, opt, records, "rlvr", policy=POLICY,
            output=tmp_path / "lr-mismatch", round_no=4, steps_before=2,
            current_adapter=adapter, checkpoint_sha256="cp-sha",
            campaign_plan_sha256="plan-sha", device="cpu")
    assert calls == []
    assert not (tmp_path / "lr-mismatch" / "candidate.adapter").exists()


def test_rollout_parent_adapter_mismatch_rejects_before_update(monkeypatch, tmp_path):
    trainer, opt, records, adapter = _setup(monkeypatch, tmp_path)
    records[0]["policy_adapter_sha256"] = "wrong-parent-sha"
    calls = []
    monkeypatch.setattr(tool_training, "perform_update",
                        lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match="rollout/parent adapter binding"):
        tool_training.replay_qualified_update(
            trainer, opt, records, "rlvr", policy=POLICY,
            output=tmp_path / "adapter-mismatch", round_no=4, steps_before=2,
            current_adapter=adapter, checkpoint_sha256="cp-sha",
            campaign_plan_sha256="plan-sha", device="cpu")
    assert calls == []
    assert not (tmp_path / "adapter-mismatch" / "candidate.adapter").exists()
