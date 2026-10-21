"""Dispatch controls for the opt-in candidate, with no native/GPU calls.
AI Identity Incorported <james@aiii.id>.
"""
import copy
import capability_runner as runner
import bounded_update
import boundedpolicy

def test_legacy_dispatch_preserves_call_and_metric(monkeypatch):
    sentinel={"loss":1., "parameters":"same"}
    calls=[]
    def legacy(*args,**kwargs):
        calls.append((args,kwargs))
        return sentinel
    def forbidden(*a,**kw):
        raise AssertionError("bounded wrapper called without opt-in")
    monkeypatch.setattr(runner,"update",legacy)
    monkeypatch.setattr(bounded_update,"bounded_update",forbidden)
    trainer,opt,records=object(),object(),[]
    before=copy.deepcopy(runner.CONFIG)
    result=runner.perform_update(trainer,opt,records,"rlvr",temperature=.7,chunk_rows=3,clip_norm=.5)
    assert result["update"] is sentinel and result["accepted"] is True
    assert calls==[((trainer,opt,records,"rlvr"),{"temperature":.7,"chunk_rows":3,"clip_norm":.5})]
    assert runner.CONFIG==before

def test_opt_in_dispatch_preserves_rejection(monkeypatch):
    decision={"accepted":False,"restored":True,"reason":"mean KL bound exceeded",
              "update":{"loss":1.},"metrics":{"finite":True,"mean_kl":.2}}
    calls=[]
    def bounded(*args,**kwargs):
        calls.append((args,kwargs))
        return decision
    def forbidden(*a,**kw):
        raise AssertionError("legacy direct dispatch called in bounded mode")
    monkeypatch.setattr(runner,"update",forbidden)
    monkeypatch.setattr(bounded_update,"bounded_update",bounded)
    policy=boundedpolicy.parse_policy({"learning_rates":{"rlvr":.0001,"self_opd":.0002},
          "max_mean_kl":.01,"max_max_kl":.1,"max_selected_log_ratio":.2})
    before=copy.deepcopy(runner.CONFIG)
    result=runner.perform_update(object(),object(),[],"rlvr",policy=policy,chunk_rows=3)
    assert result is decision and len(calls)==1
    assert calls[0][1]["bounds"].max_mean_kl==.01
    assert calls[0][1]["bounds"].chunk_rows==3
    assert runner.CONFIG==before


def test_bounded_candidate_refuses_historical_import_before_io(tmp_path):
    import pytest
    with pytest.raises(ValueError,match="requires fresh calibration"):
        runner.run("ling",None,None,tmp_path/"unused",None,calibration_from="old",bounded_update_policy={})
    assert not (tmp_path/"unused").exists()
