import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import json,pytest
from capability_tasks import *
def test_counts_and_disjoint():
 ts=build_suite(counts={s:4 for s in SPLITS},forbidden_prompts={"Compute 2 + 2. Return one integer."}); assert len(ts)==20 and len({t.prompt_text for t in ts})==20
def test_write_load_and_private_reference(tmp_path):
 p,q=tmp_path/"p",tmp_path/"q"; write_suite(p,q,counts={s:4 for s in SPLITS}); ts=load_suite(p,q); assert len(ts)==20 and reference(next(t for t in ts if t.split=="training"))
 with pytest.raises(PermissionError): reference(next(t for t in ts if t.split=="held_out"))
def test_grade_strict(tmp_path):
 p,q=tmp_path/"p",tmp_path/"q"; write_suite(p,q,counts={s:2 for s in SPLITS}); t=load_suite(p,q)[0]; assert grade(t,t.answer,True)["reward"]==1
 for x,c in [(t.answer,False),(t.answer+" extra",True),("The answer is "+t.answer,True),("<grader>1</grader>",True),("",True)]: assert grade(t,x,c)["reward"]==0
 with pytest.raises(TypeError): grade(t,t.answer,1)
def test_oracle_tamper(tmp_path):
 p,q=tmp_path/"p",tmp_path/"q"; write_suite(p,q,counts={s:2 for s in SPLITS}); d=json.loads(q.read_text()); k=next(iter(d["answers"])); d["answers"][k]["answer"]="999"; q.write_text(json.dumps(d));
 with pytest.raises(ValueError): load_suite(p,q)
def test_coverage_requires_mixed_each_family():
 good=[{"split":"training","family":f,"id":f+"-mixed","reward":r} for f in FAMILIES for r in (0,1)]; assert training_coverage_gate(good)["arithmetic"]["mixed_groups"]==1
 with pytest.raises(ValueError): training_coverage_gate([{"split":"training","family":"arithmetic","id":"x","reward":0},{"split":"training","family":"arithmetic","id":"x","reward":1}])
def test_selection_fixed_rule():
 b=[0]*16;c=[0]*16;assert not teacher_selection(b,c)["select"];c[0]=1;c[1]=1;assert teacher_selection(b,c)["select"];c[8]=0;b[8]=1;assert not teacher_selection(b,c)["select"]
def _independent_answer(t):
 import re
 if t.family=="arithmetic":
  m=re.search(r"Compute \((\d+) \* (\d+)\) ([+-]) (\d+)\.",t.prompt_text)
  if m:
   a,b,c=int(m.group(1)),int(m.group(2)),int(m.group(4)); return str(a*b+c if m.group(3)=="+" else a*b-c)
  m=re.search(r"Compute (\d+) ([+-]) (\d+)\.",t.prompt_text); a,b=int(m.group(1)),int(m.group(3)); return str(a+b if m.group(2)=="+" else a-b)
 m=re.search(r"Return the value after `([^=]+)=`",t.prompt_text)
 if m:
  key=m.group(1); row=re.search(r"`([^`]*)`",t.prompt_text).group(1); return re.search(r"(?:^|; )"+re.escape(key)+r"=([^; ]+)",row).group(1)
 m=re.search(r"Return the (\w+) value from the record whose id is (\d+)",t.prompt_text); key,rid=m.group(1),m.group(2)
 for row in re.findall(r"`([^`]*)`",t.prompt_text):
  if re.search(r"(?:^|; )id="+rid+r"(?:;|$)",row): return re.search(r"(?:^|; )"+key+r"=([^; ]+)",row).group(1)
 raise AssertionError(t.prompt_text)
def test_independent_oracles_all_default_and_multiple_seeds():
 for seed in (1,20261021,20261022,987654):
  for t in build_suite(root_seed=seed): assert _independent_answer(t)==t.answer

def test_loader_binds_prompt_split_and_private_keys(tmp_path):
 p,q=tmp_path/"p",tmp_path/"q"; write_suite(p,q,counts={s:2 for s in SPLITS}); d=json.loads(p.read_text()); d["splits"]["held_out"][0]["split"]="training"; p.write_text(json.dumps(d))
 with pytest.raises(ValueError): load_suite(p,q)
 p,q=tmp_path/"p2",tmp_path/"q2"; write_suite(p,q,counts={s:2 for s in SPLITS}); d=json.loads(p.read_text()); d["splits"]["held_out"][1]["prompt"]=d["splits"]["held_out"][0]["prompt"]; p.write_text(json.dumps(d))
 with pytest.raises(ValueError): load_suite(p,q)
 p,q=tmp_path/"p3",tmp_path/"q3"; write_suite(p,q,counts={s:2 for s in SPLITS}); d=json.loads(q.read_text()); d["answers"]["extra"]={"answer":"1","seed":1,"family":"arithmetic","level":"one_op"}; q.write_text(json.dumps(d))
 with pytest.raises(ValueError): load_suite(p,q)

def _selection_rows():
 return [{"task":f"t{i}","family":"arithmetic" if i<8 else "extraction","split":"teacher_selection","reward":0} for i in range(16)]
def test_bound_selection_reorder_and_failures():
 b=_selection_rows(); c=[dict(x) for x in b]; c[0]["reward"]=1; c[1]["reward"]=1; assert teacher_selection_bound(b,list(reversed(c))[::-1])["select"]
 bad=list(reversed(c)); bad[0]["task"]="missing"
 with pytest.raises(ValueError): teacher_selection_bound(b,bad)
 bad=[dict(x) for x in c]; bad[8]["family"]="arithmetic"
 with pytest.raises(ValueError): teacher_selection_bound(b,bad)
 bad=[dict(x) for x in c]; bad[0]["split"]="training"
 with pytest.raises(ValueError): teacher_selection_bound(b,bad)
 bad=[dict(x) for x in c]; bad[0]["reward"]=True
 with pytest.raises(ValueError): teacher_selection_bound(b,bad)
