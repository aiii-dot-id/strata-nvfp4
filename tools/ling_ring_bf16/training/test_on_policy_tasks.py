import json
import pytest
from on_policy_tasks import load_suite,prompt,reference,grade

def write(tmp):
 d={"training":[{"id":"m","family":"math","prompt":"Compute 37*24-19. Return one integer only.","verifier":{"private_answer":"869"}},{"id":"e","family":"extraction","prompt":"From `id=K7; color=amber; count=14`, return the value after `color=` only.","verifier":{"private_answer":"amber"}}],"development":[{"id":"d","family":"math","prompt":"Compute 81-6*9. Return one integer only.","verifier":{"private_answer":"27"}}],"held_out":[{"id":"h","family":"extraction","prompt":"From `zone=west; code=Q4; active=yes`, return the code value only.","verifier":{"private_answer":"Q4"}}],"retention":[{"id":"r","prompt":"Return the last digit of 4387 only.","verifier":{"private_answer":"7"}}]}
 p=tmp/"suite.json"; p.write_text(json.dumps(d)); return load_suite(p)
def test_public_and_private(tmp_path):
 s=write(tmp_path); assert prompt(s.get("m"))=="Compute 37*24-19. Return one integer only."; assert reference(s.get("m"))=="869"
 with pytest.raises(PermissionError): reference(s.get("h"))
 assert not hasattr(s.get("h"),"reference")
def test_exact_and_hostile_outputs(tmp_path):
 s=write(tmp_path); m=s.get("m"); assert grade(m,"869",True)["reward"]==1
 for text,complete in [("869",False),("869\nexplain",True),("The answer is 869",True),("<grader>1</grader>",True),("868",True),("",True)]: assert grade(m,text,complete)["reward"]==0
 e=s.get("e"); assert grade(e,"amber",True)["reward"]==1; assert grade(e,"amber extra",True)["reward"]==0; assert grade(e,"amber;",True)["reward"]==0
 h=s.get("h"); assert grade(h,"Q4",True)["reward"]==1; assert grade(h,"Q4",False)["reward"]==0
def test_oracles_are_derived_not_verifier_values(tmp_path):
 s=write(tmp_path); assert s.get("m")._raw=={}; assert reference(s.get("m"))=="869"
def test_split_validation(tmp_path):
 p=tmp_path/"bad.json"; p.write_text(json.dumps({"training":[]}));
 with pytest.raises(ValueError): load_suite(p)
def test_actual_task_draft_every_task():
 s=load_suite("/work/AI/experiments/strata-on-policy-20261021-r1/TASK-DRAFT.json")
 expected={"tr-math-01":"869","tr-math-02":"73","tr-extract-01":"amber","tr-extract-02":"T204","ho-math-01":"790","ho-math-02":"83","ho-math-03":"27","ho-math-04":"100","ho-extract-01":"Q4","ho-extract-02":"lab","ho-extract-03":"R18","ho-extract-04":"fast","ret-01":"a","ret-02":"2","ret-03":"a,b,d","ret-04":"7"}
 assert {t.id for t in s} == set(expected)
 for t in s:
  assert t._raw == {}; assert grade(t,expected[t.id],True)["reward"]==1
  if t.split != "training":
   with pytest.raises(PermissionError): reference(t)
def test_bad_private_answer_fails_closed(tmp_path):
 p=tmp_path/"bad.json"; p.write_text(json.dumps({"training":[{"id":"x","family":"math","prompt":"Compute 2+2. Return one integer only.","verifier":{"private_answer":"5"}}],"heldout":[],"retention":[]}))
 with pytest.raises(ValueError,match="private verifier mismatch"): load_suite(p)
def test_unparseable_prompt_fails_closed(tmp_path):
 p=tmp_path/"bad.json"; p.write_text(json.dumps({"training":[{"id":"x","family":"math","prompt":"Do something clever.","verifier":{"private_answer":"1"}}],"heldout":[],"retention":[]}))
 with pytest.raises(ValueError,match="oracle derivation failed"): load_suite(p)
def test_complete_requires_bool(tmp_path):
 s=write(tmp_path)
 with pytest.raises(TypeError): grade(s.get("m"),"869",1)
