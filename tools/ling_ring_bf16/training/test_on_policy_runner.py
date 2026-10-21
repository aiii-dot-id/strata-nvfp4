import json
from pathlib import Path
import numpy as np
import pytest
import torch

import on_policy_runner as runner


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "<chat>"+messages[0]["content"]+("<think>\n" if add_generation_prompt else "")
    def encode(self, text, add_special_tokens=False): return list(range(1, min(len(text),20)+1))


class LongTokenizer(FakeTokenizer):
    def encode(self, text, add_special_tokens=False): return list(range(1,201))


def test_prompt_contract_reference_and_ring_suffix(monkeypatch):
    tok=FakeTokenizer()
    ling,meta=runner.prompt_ids(tok,"ling","solve",None)
    assert ling and meta["original_template"] and not meta["privileged_reference"] and meta["controller_suffix"]==""
    ring,rm=runner.prompt_ids(tok,"ring","solve",None)
    assert ring and rm["controller_suffix"]=="</think>\n" and ring[-2:]==[len(ring)-1,len(ring)]
    privileged,pm=runner.prompt_ids(tok,"ling","solve","reference")
    assert len(privileged)>len(ling) and pm["privileged_reference"] and pm["rendered_sha256"]!=meta["rendered_sha256"]


def test_prompt_and_response_alignment_reject_bad_bounds():
    with pytest.raises(ValueError): runner.response_rows([1,2],[3],3)
    with pytest.raises(ValueError): runner.prompt_ids(LongTokenizer(),"ling","x",None)


class FakeTrainer(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.p=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.)) for _ in range(6)])
    def forward(self, features):
        base=torch.tensor([[1.,0.,-1.],[0.,1.,-1.]],device=self.p[0].device).repeat(features.shape[0]//2,1)
        shift=sum((i+1)*p for i,p in enumerate(self.p))*torch.ones_like(base)
        return base+shift


def _record(reward=1.0,group=0):
    logits=np.array([[1.,0.,-1.],[0.,1.,-1.]],dtype=np.float32)
    lp=torch.log_softmax(torch.from_numpy(logits),-1).gather(1,torch.tensor([[0],[1]])).flatten().numpy()
    return {"features":torch.zeros(2,3),"logits":logits,"ids":[0,1],"sampling_logprobs":lp,"teacher":np.zeros_like(logits),"grade":{"reward":reward},"group":group}


def test_update_cpu_fake_trainer_checks_gradients_for_both_objectives(monkeypatch):
    monkeypatch.setattr(torch.Tensor,"cuda",lambda self,*a,**k:self)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    original_tensor=torch.tensor
    def cpu_tensor(*args,**kwargs):
        if kwargs.get("device")=="cuda": kwargs["device"]="cpu"
        return original_tensor(*args,**kwargs)
    monkeypatch.setattr(torch,"tensor",cpu_tensor)
    for arm,records in (("self_opd",[_record()]),("rlvr",[_record(1.,4),_record(0.,4)])):
        t=FakeTrainer(); opt=torch.optim.Adam(t.parameters(),lr=.01,foreach=False)
        result=runner.update(t,opt,records,arm)
        assert result["objective"]==arm and result["gradient_norm"]>0 and len(result["gradient"])==6


def test_runner_source_orders_teacher_and_freeze_before_heldout():
    source=Path(runner.__file__).read_text()
    teacher=source.index("teacher_prompt,tm=prompt_ids")
    update=source.index("result=update(t,opt,records,arm)")
    assert teacher<update
    frozen=source.index("FROZEN-CANDIDATES.json")
    held=source.index("for i,task in enumerate(held+ret)")
    assert frozen<held
    assert source.index("release(t);t=None",frozen)<held


def test_runner_release_and_current_policy_contracts_are_explicit():
    source=Path(runner.__file__).read_text()
    assert "release(t);d=out/name" in source
    assert "if roundno>0 and any(r['policy_adapter_sha256']!=native.sha(current)" in source
    assert "fresh_prefixes_after_previous_update" in source
