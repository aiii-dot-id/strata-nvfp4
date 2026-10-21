"""Fresh deterministic arithmetic/extraction coverage tasks."""
import hashlib,json,random,re
from dataclasses import dataclass
from pathlib import Path
AUTHOR="AI Identity Incorported <james@aiii.id>"
SPLITS=("training","development","teacher_selection","held_out","retention")
FAMILIES=("arithmetic","extraction")
DEFAULT_COUNTS={"training":24,"development":16,"teacher_selection":16,"held_out":24,"retention":12}
@dataclass(frozen=True)
class Task:
 id:str; split:str; family:str; level:str; prompt_text:str; answer:str; seed:int
 def public(self): return {"id":self.id,"split":self.split,"family":self.family,"level":self.level,"prompt":self.prompt_text}
def _seed(root,s,f,i): return root+SPLITS.index(s)*100000+FAMILIES.index(f)*10000+i
def _math(r,l):
 if l=="one_op":
  a,b=r.randint(12,90),r.randint(3,40); op=r.choice(("+","-")); a,b=(b,a) if op=="-" and b>a else (a,b); return f"Compute {a} {op} {b}. Return one integer.",str(a+b if op=="+" else a-b)
 a,b,c=r.randint(3,24),r.randint(3,18),r.randint(1,30); op=r.choice(("+","-")); c=min(c,a*b-1) if op=="-" else c; return f"Compute ({a} * {b}) {op} {c}. Return one integer.",str(a*b+c if op=="+" else a*b-c)
def _extract(r,l):
 vals=["amber","blue","copper","fast","green","lab","north","Q4","R18","T204","west"]; target=r.choice(vals); key=r.choice(("color","status","code","ticket")); other=[v for v in vals if v!=target]
 def row(rid,chosen):
  fields={"color":r.choice(other),"status":r.choice(other),"code":r.choice(other),"ticket":r.choice(other)}; fields[key]=chosen
  return f"id={rid}; " + "; ".join(f"{k}={v}" for k,v in fields.items())
 ids=r.sample(range(10,99),3)
 if l=="one_row":
  d=row(ids[0],target); return f"Read the record `{d}`. Return the value after `{key}=`.",target
 rows=[row(ids[0],target),row(ids[1],r.choice(other)),row(ids[2],r.choice(other))]; r.shuffle(rows); queried=rows[0]; rid=queried.split("=",1)[1].split(";",1)[0]; answer=re.search(r"(?:^|; )"+re.escape(key)+r"=([^; ]+)",queried).group(1)
 return "Read these records:\n"+"\n".join(f"{i+1}. `{x}`" for i,x in enumerate(rows))+f"\nReturn the {key} value from the record whose id is {rid}.",answer
def _level(f,i): return ("one_op","two_op")[i%2] if f=="arithmetic" else ("one_row","multi_row")[i%2]
def build_suite(*,root_seed=20261021,counts=None,forbidden_prompts=()):
 counts=dict(DEFAULT_COUNTS if counts is None else counts); seen=set(); out=[]
 for s in SPLITS:
  n=counts.get(s,0)
  if n<0 or n%2: raise ValueError(f"{s} count must be nonnegative and even")
  for f in FAMILIES:
   for j in range(n//2):
    seed=_seed(root_seed,s,f,j); r=random.Random(seed); l=_level(f,j); text,ans=_math(r,l) if f=="arithmetic" else _extract(r,l)
    if text in seen or text in set(forbidden_prompts): raise ValueError("exact prompt collision")
    out.append(Task(f"cap-{s}-{f}-{j:02d}",s,f,l,text,ans,seed)); seen.add(text)
 return out
def write_suite(public_path,private_path,**kw):
 ts=build_suite(**kw); pub={s:[] for s in SPLITS}; priv={}
 for t in ts: pub[t.split].append(t.public()); priv[t.id]={"answer":t.answer,"seed":t.seed,"family":t.family,"level":t.level}
 Path(public_path).write_text(json.dumps({"author":AUTHOR,"schema":"capability-task-public-v1","splits":pub},sort_keys=True,indent=2)+"\n"); Path(private_path).write_text(json.dumps({"author":AUTHOR,"schema":"capability-task-private-v1","answers":priv},sort_keys=True,indent=2)+"\n"); return ts
def load_suite(public_path,private_path,forbidden_prompts=()):
 p=json.loads(Path(public_path).read_text()); q=json.loads(Path(private_path).read_text()).get("answers");
 if p.get("schema")!="capability-task-public-v1" or not isinstance(q,dict): raise ValueError("bad suite schema")
 out=[]; seen=set(); prompts=set(); forbidden=set(forbidden_prompts)
 for s in SPLITS:
  for x in p.get("splits",{}).get(s,[]):
   if not isinstance(x,dict) or x.get("split")!=s: raise ValueError("split binding")
   tid,text,f,l=x.get("id"),x.get("prompt"),x.get("family"),x.get("level")
   if not isinstance(tid,str) or tid in seen or not isinstance(text,str) or text in forbidden or text in prompts or f not in FAMILIES: raise ValueError("bad task binding or duplicate prompt")
   if l not in (("one_op","two_op") if f=="arithmetic" else ("one_row","multi_row")): raise ValueError("bad level")
   v=q.get(tid)
   if not isinstance(v,dict) or set(v)!={"answer","seed","family","level"} or v.get("family")!=f or v.get("level")!=l: raise ValueError("private binding")
   t=Task(tid,s,f,l,text,str(v.get("answer")),int(v.get("seed"))); r=random.Random(t.seed); generated=_math(r,l) if f=="arithmetic" else _extract(r,l)
   if generated[0]!=text or generated[1]!=t.answer: raise ValueError("prompt or oracle mismatch")
   out.append(t); seen.add(tid); prompts.add(text)
 if set(q)!=seen: raise ValueError("private task key mismatch")
 return out
def prompt(task): return task.prompt_text
def reference(task):
 if task.split!="training": raise PermissionError("reference restricted to training")
 return task.answer
def grade(task,text,complete):
 if type(complete) is not bool: raise TypeError("complete must be bool")
 if not complete:return {"reward":0,"reason":"truncated_or_missing_eos"}
 if not isinstance(text,str):return {"reward":0,"reason":"non_string"}
 out=text.strip()
 if not out:return {"reward":0,"reason":"empty"}
 if "\n" in out or "\r" in out or "```" in out or re.search(r"<\/?(?:think|tool|grader)",out,re.I):return {"reward":0,"reason":"format_or_injection"}
 if task.family=="arithmetic" and not re.fullmatch(r"-?\d+",out):return {"reward":0,"reason":"not_bare_integer"}
 if task.family=="extraction" and not re.fullmatch(r"[A-Za-z0-9_-]+",out):return {"reward":0,"reason":"not_bare_literal"}
 return {"reward":int(out==task.answer),"reason":"exact_match" if out==task.answer else "wrong_answer"}
def training_coverage_gate(records):
 by={f:{} for f in FAMILIES}
 for x in records:
  if x.get("split")!="training" or x.get("family") not in FAMILIES: raise ValueError("training records required")
  by[x["family"]].setdefault(x["id"],[]).append(int(x["reward"]))
 out={}
 for f,g in by.items():
  if not g or not any(set(v)=={0,1} for v in g.values()): raise ValueError(f"no mixed sampled training prompt group for {f}")
  out[f]={"groups":len(g),"mixed_groups":sum(set(v)=={0,1} for v in g.values()),"successes":sum(v.count(1) for v in g.values()),"failures":sum(v.count(0) for v in g.values())}
 return out
def teacher_selection_bound(baseline_records,candidate_records):
 """Gate paired teacher selection using task IDs and verified family labels."""
 if not isinstance(baseline_records,list) or not isinstance(candidate_records,list) or len(baseline_records)!=16 or len(candidate_records)!=16: raise ValueError("teacher selection requires 16 records per arm")
 def norm(rows):
  out={}
  for r in rows:
   if not isinstance(r,dict) or r.get("split")!="teacher_selection" or r.get("family") not in FAMILIES: raise ValueError("selection record binding")
   tid=r.get("task"); reward=r.get("reward")
   if not isinstance(tid,str) or tid in out or type(reward) is not int or reward not in (0,1): raise ValueError("selection id/reward binding")
   out[tid]=(r["family"],reward)
  if {f for f,_ in out.values()} != set(FAMILIES) or sum(f=="arithmetic" for f,_ in out.values())!=8 or sum(f=="extraction" for f,_ in out.values())!=8: raise ValueError("selection family counts")
  return out
 b=norm(baseline_records); c=norm(candidate_records)
 if set(b)!=set(c): raise ValueError("selection task IDs differ")
 for tid in b:
  if b[tid][0]!=c[tid][0]: raise ValueError("selection family differs")
 gain=sum(c[tid][1]-b[tid][1] for tid in sorted(b))
 famgain={f:sum(c[tid][1]-b[tid][1] for tid in b if b[tid][0]==f) for f in FAMILIES}
 selected=gain>=2 and all(v>=0 for v in famgain.values())
 return {"select":selected,"status":"candidate_better_on_fixed_selection" if selected else "inconclusive_or_not_better","gain":gain,"family_gain":famgain,"selection_ids":sorted(b)}

def teacher_selection(baseline,candidate):
 if len(baseline)!=16 or len(candidate)!=16: raise ValueError("teacher selection requires 16 paired tasks")
 gain=sum(int(c)-int(b) for b,c in zip(baseline,candidate)); fam=(range(8),range(8,16))
 if gain<2 or any(sum(candidate[i] for i in z)<sum(baseline[i] for i in z) for z in fam): return {"select":False,"status":"inconclusive_or_not_better","gain":gain}
 return {"select":True,"status":"candidate_better_on_fixed_selection","gain":gain,"exploratory":True}
