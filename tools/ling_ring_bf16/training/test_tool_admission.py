import json, sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tool_admission as a

SUITE=Path('/work/AI/experiments/strata-capability-20261021-r2/tool-strategy-prep/tool-fixtures/fixture-set-v8')

class FakeTok:
    pass
class FakeTrainer:
    def __init__(self, logits): self.logits=logits
    def cuda(self): return self
    def __call__(self, x):
        import torch
        return torch.as_tensor(self.logits[:len(x)], dtype=torch.float32)

def patch_common(monkeypatch, tmp_path, parity=True, replay_equal=True):
    import torch
    monkeypatch.setattr(a.native, '_git_state', lambda root: {'commit':'clean-test','dirty':False,'status':[]})
    monkeypatch.setattr(a.native, '_source_closure', lambda root: {'tool_admission.py': {'size':1,'sha256':'source'}})
    monkeypatch.setattr(a.native, 'sha', lambda p: 'b'*64)
    monkeypatch.setattr(a.AutoTokenizer, 'from_pretrained', lambda *x, **k: FakeTok())
    monkeypatch.setattr(a.torch.cuda, 'manual_seed_all', lambda *x, **k: None)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *x, **k: self)
    monkeypatch.setattr(a.qualify, 'compare_logits', lambda expected, actual: {'passed':parity,'max_abs':0.0})
    monkeypatch.setattr(a, 'advantages', lambda records: [1.0])
    monkeypatch.setattr(a, 'release', lambda trainer=None: None)
    fake_logits=np.arange(8,dtype=np.float32).reshape(2,4)
    monkeypatch.setattr(a.terminal, 'build_terminal', lambda *x, **k: FakeTrainer(fake_logits))
    def save_adapter(trainer, path, checkpoint, checkpoint_sha256=None):
        Path(path).write_bytes(b'zero-adapter')
        Path(str(path)+'.json').write_text(json.dumps({'model':'ling','checkpoint_sha256':'b'*64,'hidden':1536}))
    monkeypatch.setattr(a.terminal, 'save_adapter', save_adapter)
    calls=[]
    def rec(success=True, calls_n=1, ids=None):
        import torch
        ids=[1,2] if ids is None else ids
        return {'ids':ids,'transcript_ids':[9]+ids,'features':torch.zeros((2,4608)),
                'logits':fake_logits.copy(),'sampling_logprobs':[-.1,-.2],
                'success':success,'complete':True,'tool_calls':calls_n,'tool_attempts':calls_n,'wall_seconds':0.01}
    def sample(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get('adapter') is not None and not replay_equal:
            return rec(ids=[3,4])
        return rec()
    monkeypatch.setattr(a, 'sample_tool_task', sample)
    return calls

def test_load_inputs_actual_64_controls_and_pins():
    public, oracle, manifests, pins=a.load_inputs(SUITE)
    assert len(public)==len(oracle)==len(manifests)==64
    assert all(t['id'] in oracle and t['id'] in manifests for t in public)
    assert set(pins)=={'SUITE.public.json','ORACLES.private.json','FIXTURE-MANIFESTS.public.json','ROOT-INDEPENDENT-REVIEW.json','CANONICAL-TOOL-CPU-CONTROLS.json'}

def test_no_actual_tools_rejected(monkeypatch, tmp_path):
    patch_common(monkeypatch,tmp_path)
    import torch
    def no_tools(*args, **kwargs):
        return {'ids':[],'transcript_ids':[],'features':torch.zeros((0,4608)),'logits':np.zeros((0,4),np.float32),'sampling_logprobs':[],'success':False,'complete':False,'tool_calls':0,'tool_attempts':0,'wall_seconds':0.01}
    monkeypatch.setattr(a,'sample_tool_task',no_tools)
    with pytest.raises(ValueError, match='no actual tool calls'):
        a.run('ling',SUITE,tmp_path/'out',tmp_path/'bridge','b'*64)

def test_zero_verified_success_rejected(monkeypatch,tmp_path):
    patch_common(monkeypatch,tmp_path)
    import torch
    def zero_success(*args,**kwargs):
        return {'ids':[1,2],'transcript_ids':[9,1,2],'features':torch.zeros((2,4608)),'logits':np.zeros((2,4),np.float32),'sampling_logprobs':[-.1,-.2],'success':False,'complete':True,'tool_calls':1,'tool_attempts':1,'wall_seconds':0.01}
    monkeypatch.setattr(a,'sample_tool_task',zero_success)
    with pytest.raises(ValueError, match='no verified tool completion'):
        a.run('ling',SUITE,tmp_path/'out',tmp_path/'bridge','b'*64)

def test_clean_source_zero_adapter_replay_and_parity(monkeypatch,tmp_path):
    calls=patch_common(monkeypatch,tmp_path,parity=True,replay_equal=True)
    result=a.run('ling',SUITE,tmp_path/'out',tmp_path/'bridge','b'*64)
    assert result['status']=='qualified_native_tool_path'
    assert result['zero_adapter_replay_exact'] is True
    assert result['training_updates']==0 and result['heldout_used'] is False
    assert len(calls)==5 and calls[-1].get('adapter') is not None

def test_terminal_parity_mismatch_rejected(monkeypatch,tmp_path):
    patch_common(monkeypatch,tmp_path,parity=False)
    with pytest.raises(ValueError, match='terminal parity'):
        a.run('ling',SUITE,tmp_path/'out',tmp_path/'bridge','b'*64)

def test_zero_adapter_replay_mismatch_rejected(monkeypatch,tmp_path):
    patch_common(monkeypatch,tmp_path,parity=True,replay_equal=False)
    with pytest.raises(ValueError, match='zero-adapter tool replay differs'):
        a.run('ling',SUITE,tmp_path/'out',tmp_path/'bridge','b'*64)


def test_dirty_source_rejected_before_native_or_tools(monkeypatch, tmp_path):
    called=[]
    monkeypatch.setattr(a.native, '_git_state', lambda root: {'commit':'dirty-test','dirty':True,'status':[' M source.py']})
    monkeypatch.setattr(a, 'sample_tool_task', lambda *args, **kwargs: called.append(True))
    monkeypatch.setattr(a.native, 'sha', lambda p: 'b'*64)
    monkeypatch.setattr(a.AutoTokenizer, 'from_pretrained', lambda *x, **k: FakeTok())
    with pytest.raises(ValueError, match='source must be frozen'):
        a.run('ling', SUITE, tmp_path/'out', tmp_path/'bridge', 'b'*64)
    assert called == []
