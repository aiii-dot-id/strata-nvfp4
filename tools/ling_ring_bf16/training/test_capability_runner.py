import copy
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import capability_runner as cr
import capability_tasks as tasks


def rec(task, arm, split, family, reward, complete=True, wall=1.0):
    return {'task':task,'arm':arm,'split':split,'family':family,'grade':{'reward':reward},'complete':complete,'wall_seconds':wall}


def test_strip_final_preserves_format_failure_and_ring_close():
    assert cr.strip_final('answer', 'ling', None)==('answer',None)
    assert cr.strip_final(' reason </think> 42 ', 'ring', None)==('42',None)
    assert cr.strip_final('anything', 'ring', 'special token')==('', 'special token')
    assert cr.strip_final('no close', 'ring', None)==('', 'missing_or_duplicate_thinking_close')


def test_coverage_requires_mixed_complete_group_per_family():
    rows=[]
    for family in tasks.FAMILIES:
        rows.extend([rec(f'{family}-g','calibration','training',family,0),rec(f'{family}-g','calibration','training',family,1)])
    got=cr.coverage(rows)
    assert got['passed']
    assert got['families']['arithmetic']['mixed_groups']==['arithmetic-g']
    assert got['families']['extraction']['completion_rate']==1.0
    bad=rows[:-1]+[rec('extraction-g2','calibration','training','extraction',0)]
    assert not cr.coverage(bad)['passed']


def test_summarize_keeps_arm_split_family_and_incomplete_counts():
    rows=[rec('a','baseline','held_out','arithmetic',1,True,2.),rec('b','baseline','held_out','extraction',0,False,3.),rec('c','rlvr','retention','arithmetic',1,True,4.)]
    got=cr.summarize(rows)
    assert got['baseline']['held_out']=={'n':2,'correct':1,'incomplete':1,'wall_seconds':5.,'verified_per_instrumented_hour':720.,'families':{'arithmetic':1,'extraction':0}}
    assert got['rlvr']['retention']['families']['arithmetic']==1


def test_nomination_uses_advantage_weighted_routes_in_both_families():
    h=2; width=3*h+8
    def f(ids):
        x=torch.zeros(1,width); x[0,3*h:3*h+8]=torch.tensor(ids,dtype=torch.float32); return x
    rows=[
      {'group':'a','family':'arithmetic','features':f([7]*8),'grade':{'reward':0}},
      {'group':'a','family':'arithmetic','features':f([7]*8),'grade':{'reward':1}},
      {'group':'b','family':'extraction','features':f([7]*8),'grade':{'reward':0}},
      {'group':'b','family':'extraction','features':f([7]*8),'grade':{'reward':1}},
    ]
    expert,meta=cr.nomination(rows,h)
    assert expert==7 and meta['score']>0 and meta['counts']['arithmetic'][7]>0 and meta['counts']['extraction'][7]>0


def test_nomination_rejects_expert_missing_one_family():
    h=2; width=3*h+8
    def f(ids):
        x=torch.zeros(1,width); x[0,3*h:3*h+8]=torch.tensor(ids,dtype=torch.float32); return x
    rows=[
      {'group':'a','family':'arithmetic','features':f([7]*8),'grade':{'reward':0}},
      {'group':'a','family':'arithmetic','features':f([7]*8),'grade':{'reward':1}},
      {'group':'b','family':'extraction','features':f([8]*8),'grade':{'reward':0}},
      {'group':'b','family':'extraction','features':f([8]*8),'grade':{'reward':1}},
    ]
    with pytest.raises(ValueError,match='both families'): cr.nomination(rows,h)


def test_capability_task_suite_prompt_and_private_oracle_binding(tmp_path):
    counts={'training':2,'development':0,'teacher_selection':0,'held_out':0,'retention':0}
    public=tmp_path/'public.json'; private=tmp_path/'private.json'
    tasks.write_suite(public,private,counts=counts)
    loaded=tasks.load_suite(public,private)
    assert len(loaded)==2 and len({x.prompt_text for x in loaded})==2
    public_data=__import__('json').loads(public.read_text())
    public_data['splits']['training'][0]['prompt'] += ' tamper'
    public.write_text(__import__('json').dumps(public_data))
    with pytest.raises(ValueError,match='prompt or oracle mismatch'):
        tasks.load_suite(public,private)
    tasks.write_suite(public,private,counts=counts)
    private_data=__import__('json').loads(private.read_text())
    first=next(iter(private_data['answers']))
    private_data['answers'][first]['answer']='tampered'
    private.write_text(__import__('json').dumps(private_data))
    with pytest.raises(ValueError,match='prompt or oracle mismatch'):
        tasks.load_suite(public,private)


def test_select_teacher_uses_actual_rotated_row_schema_and_family_bound():
    rows=[]
    for i in range(16):
        family='arithmetic' if i<8 else 'extraction'; task=f'cap-teacher_selection-{family}-{i:02d}'
        baseline=1 if i in (0,8) else 0
        candidate=baseline + (1 if i in (1,9) else 0)
        rows.extend([rec(task,'rlvr','teacher_selection',family,candidate),rec(task,'baseline','teacher_selection',family,baseline)])
    gate=cr.select_teacher(rows)
    assert gate['select'] and gate['gain']==2 and gate['family_gain']=={'arithmetic':1,'extraction':1}
    assert gate['selection_ids']==sorted({r['task'] for r in rows})

def test_select_teacher_rejects_missing_duplicate_or_malformed_selection_rows():
    rows=[]
    for i in range(16):
        family='arithmetic' if i<8 else 'extraction'; task=f't-{i:02d}'
        rows.extend([rec(task,'baseline','teacher_selection',family,0),rec(task,'rlvr','teacher_selection',family,1)])
    with pytest.raises(ValueError): cr.select_teacher(rows[:-1])
    bad=copy.deepcopy(rows); bad[0]['grade']['reward']=2
    with pytest.raises(ValueError): cr.select_teacher(bad)
    malformed=copy.deepcopy(rows); malformed[-1]['arm']='unknown'
    with pytest.raises(ValueError): cr.select_teacher(malformed)
