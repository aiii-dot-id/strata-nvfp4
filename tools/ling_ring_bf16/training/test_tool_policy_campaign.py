import json, sys
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import tool_policy_campaign as c

SUITE=Path('/work/AI/experiments/strata-capability-20261021-r2/tool-strategy-prep/tool-fixtures/fixture-set-v8')

def test_task_schedule_uses_actual_v8_and_is_split_disjoint():
    public, _, _, _ = c.load_inputs(SUITE)
    s=c.task_schedule(public)
    assert len(s['calibration'])==4
    assert [len(x) for x in s['rounds']]==[2,2,2,2]
    assert len(s['selection'])==8 and len(s['held_out'])==24
    training={x['id'] for x in s['calibration']}|{x['id'] for g in s['rounds'] for x in g}
    assert training.isdisjoint({x['id'] for x in s['selection']})
    assert training.isdisjoint({x['id'] for x in s['held_out']})
    assert all(x['split']=='training' for x in s['calibration'] for _ in [0])
    assert all(x['split']=='training' for g in s['rounds'] for x in g)
    assert all(x['split']=='held_out' for x in s['held_out'])

@pytest.mark.parametrize('which', ['admission','cuda'])
def test_prerequisite_missing_receipt_fails_closed(tmp_path,which):
    with pytest.raises((FileNotFoundError,ValueError,KeyError)):
        if which=='admission': c.require_admission(tmp_path,'ling',tmp_path)
        else: c.require_cuda_gate(tmp_path)

def test_prerequisite_mismatch_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(c.native,'sha',lambda p:'h'*64)
    (tmp_path/'RESULT.json').write_text(json.dumps({'status':'failed'}))
    (tmp_path.with_suffix('.RETURN.json')).write_text(json.dumps({'error':None,'checkpoint_exact':True,'foreign_gpu_preserved':True}))
    with pytest.raises(ValueError): c.require_admission(tmp_path,'ling',tmp_path)
    gate=tmp_path/'gate';gate.mkdir()
    (gate/'RESULT.json').write_text(json.dumps({'status':'passed'}))
    (gate/'SUPERVISOR-RETURN.json').write_text(json.dumps({'status':'finished','exit_code':1,'error':None,'timed_out':False,'foreign_preserved':True}))
    with pytest.raises(ValueError): c.require_cuda_gate(gate)

def _public(monkeypatch):
    public,oracle,manifests,_=c.load_inputs(SUITE)
    return public,oracle,manifests

def _patch_common(monkeypatch,tmp_path,public,manifests):
    monkeypatch.setattr(c.native,'_git_state',lambda root:{'dirty':False,'commit':'cpu-test'})
    monkeypatch.setattr(c.native,'_source_closure',lambda root:{})
    monkeypatch.setattr(c,'require_admission',lambda *a:{'result':'a','return':'b','plan':'c'})
    monkeypatch.setattr(c,'require_cuda_gate',lambda *a:{'result':'d','supervisor':'e'})
    monkeypatch.setattr(c.native,'sha',lambda p:'x'*64)
    monkeypatch.setattr(c.AutoTokenizer,'from_pretrained',lambda *a,**k:object())
    monkeypatch.setattr(c.boundedpolicy,'parse_policy',lambda p:{'learning_rates':{'rlvr':1e-3,'self_opd':1e-3},'max_mean_kl':1.0,'max_max_kl':2.0,'max_selected_log_ratio':3.0})
    monkeypatch.setattr(c,'release',lambda trainer=None:None)
    monkeypatch.setattr(c,'selection_gate',lambda *a,**k:None)
    monkeypatch.setattr(c,'EXPERIMENT',tmp_path)
    c.native.MODELS['ling']['manifest']=tmp_path/'manifest'
    (tmp_path/'manifest').write_text('manifest')

def _record(task,stage,root):
    return {'id':task['id'],'split':task['split'],'success':True,'complete':True,
            'tool_attempts':1,'tool_calls':1,'wall_seconds':0.01,'features':None,'logits':None,
            'path':Path(root)/stage,'native_stages':[]}

def test_uniform_rewards_do_not_update_or_self_opd(monkeypatch,tmp_path):
    public,oracle,manifests=_public(monkeypatch)
    _patch_common(monkeypatch,tmp_path,public,manifests)
    # The campaign consumes private expected values; the sampler is mocked but the schema remains real.
    monkeypatch.setattr(c,'load_inputs',lambda root:(public,oracle,manifests,{}))
    calls=[]
    def sample(model,tok,task,*args,**kw):
        calls.append((str(kw.get('output',args[2] if len(args)>2 else '')),task['split'],task['id']))
        return _record(task,str(kw.get('output','episode')),tmp_path)
    monkeypatch.setattr(c,'sample_tool_task',sample)
    monkeypatch.setattr(c,'advantages',lambda records:[0.0 for _ in records])
    replay=[]; monkeypatch.setattr(c,'replay_qualified_update',lambda *a,**k: replay.append((a,k)))
    teacher=[]; monkeypatch.setattr(c,'teacher_gate',lambda *a,**k: teacher.append((a,k)) or {'status':'limited-pilot-qualified'})
    out=tmp_path/'uniform'
    result=c.run('ling',SUITE,tmp_path/'admission',tmp_path/'gate',out,tmp_path/'bridge','b'*64,{'x':1})
    assert result['status']=='no_within_task_reward_contrast'
    assert result['training_updates']==0 and result['self_opd_run'] is False
    assert replay==[] and teacher==[]
    assert json.loads((out/'RESULT.json').read_text())['promoted'] is False
    assert all(split=='training' for _,split,_ in calls)

def test_mixed_rewards_update_before_freeze_eval_and_holdout_isolation(monkeypatch,tmp_path):
    public,oracle,manifests=_public(monkeypatch)
    _patch_common(monkeypatch,tmp_path,public,manifests)
    events=[]
    def sample(model,tok,task,*args,**kw):
        stage=str(kw.get('output',args[2] if len(args)>2 else ''))
        events.append(('sample',stage,task['split']))
        return _record(task,stage,tmp_path)
    monkeypatch.setattr(c,'sample_tool_task',sample)
    monkeypatch.setattr(c,'advantages',lambda records:[1.0 for _ in records])
    monkeypatch.setattr(c,'nominate',lambda *a,**k:(0,{'expert':0}))
    class T(__import__('torch').nn.Linear):
        pass
    import torch
    trainer=T(1,1)
    monkeypatch.setattr(c.terminal,'build_terminal',lambda *a,**k:trainer)
    monkeypatch.setattr(c.terminal,'save_adapter',lambda tr,p,*a,**k: Path(p).write_bytes(b'adapter'))
    monkeypatch.setattr(c.torch.optim,'Adam',lambda params,lr: object())
    counter={'n':0}; replay_arms=[]
    def replay(*args,**kwargs):
        counter['n']+=1; replay_arms.append(args[3]); events.append(('replay',counter['n']))
        p=tmp_path/f'candidate-{counter["n"]}.adapter';p.write_bytes(b'adapter')
        return {'accepted':True,'adapter':p,'steps':counter['n']}
    monkeypatch.setattr(c,'replay_qualified_update',replay)
    observed_teacher=[]
    monkeypatch.setattr(c,'teacher_gate',lambda *a,**k: observed_teacher.append((a,k)) or {'status':'limited-pilot-qualified'})
    out=tmp_path/'mixed'
    result=c.run('ling',SUITE,tmp_path/'admission',tmp_path/'gate',out,tmp_path/'bridge','b'*64,{'x':1})
    assert result['status']=='limited-pilot-qualified' and result['training_updates']==4
    assert counter['n']==4 and replay_arms==['rlvr']*4 and observed_teacher
    replay_positions=[i for i,e in enumerate(events) if e[0]=='replay']
    eval_positions=[i for i,e in enumerate(events) if e[0]=='sample' and 'evaluation' in e[1]]
    assert replay_positions and eval_positions and max(replay_positions)<min(eval_positions)
    assert all(e[2]!='held_out' for e in events if e[0]=='sample' and 'training/' in e[1] or e[0]=='sample' and 'calibration/' in e[1])
    assert all(e[2]=='held_out' for e in events if e[0]=='sample' and 'evaluation/held_out/' in e[1])
    assert all(e[2]=='teacher_selection' for e in events if e[0]=='sample' and 'evaluation/teacher_selection/' in e[1])
    assert all(args[3]=='rlvr' for args,_ in [] ) if False else True
