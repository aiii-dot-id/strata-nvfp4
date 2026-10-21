"""CPU-only calibration admission classifier; no model or native calls."""
from __future__ import annotations
from collections import defaultdict
EXPECTED=tuple(f"cap-training-{f}-{i:02d}" for f in ("arithmetic","extraction") for i in range(4))
FAMILIES=("arithmetic","extraction")
def classify_calibration(records, expected_ids=EXPECTED):
 if not isinstance(records,list) or len(records)!=32: raise ValueError("calibration must contain exactly 32 records")
 expected=set(expected_ids)
 if expected != set(EXPECTED) or len(expected)!=8: raise ValueError("unexpected expected task IDs")
 groups=defaultdict(list); seen=set(); expected_family={tid:f for f in FAMILIES for tid in (f"cap-training-{f}-{i:02d}" for i in range(4))}
 for r in records:
  if not isinstance(r,dict) or r.get("split")!="training": raise ValueError("record split binding")
  tid=r.get("task",r.get("id")); fam=r.get("family"); reward=r.get("reward",r.get("grade",{}).get("reward")); complete=r.get("complete")
  if tid not in expected or fam not in FAMILIES or expected_family.get(tid)!=fam or type(reward) is not int or reward not in (0,1) or complete is not True: raise ValueError("invalid or incomplete calibration record")
  seed=r.get("seed")
  idx=int(tid.rsplit("-",1)[1]) + (0 if fam=="arithmetic" else 4)
  if type(seed) is not int or seed not in {721021+idx*10+j for j in range(4)}: raise ValueError("calibration seed binding")
  key=(tid,seed)
  if key in seen: raise ValueError("duplicate calibration sample")
  seen.add(key); groups[tid].append(int(reward))
 if set(groups)!=expected or any(len(v)!=4 for v in groups.values()): raise ValueError("exact task coverage or four samples violated")
 family={}
 for fam in FAMILIES:
  tids=tuple(t for t in sorted(expected) if t.startswith(f"cap-training-{fam}-")); vals=[x for t in tids for x in groups[t]]
  mixed_groups=[t for t in tids if set(groups[t])=={0,1}]
  if mixed_groups: family[fam]={"status":"active_mixed","active":True,"retention_control":False,"rewards":vals,"mixed_groups":mixed_groups}
  elif set(vals)=={1}: family[fam]={"status":"saturated_perfect_retention_control","active":False,"retention_control":True,"rewards":vals,"mixed_groups":[]}
  elif set(vals)=={0}: raise ValueError(f"all-zero family is noninformative: {fam}")
  else: raise ValueError(f"invalid family reward set: {fam}")
 active=[f for f in FAMILIES if family[f]["active"]]
 if not active: raise ValueError("no active mixed family")
 return {"status":"admit_scoped_training","active_families":active,"retention_control_families":[f for f in FAMILIES if family[f]["retention_control"]],"family":family,"task_ids":sorted(expected),"samples_per_task":4,"selection_is_calibration_derived":True,"unbiased_estimator_claim":False}
