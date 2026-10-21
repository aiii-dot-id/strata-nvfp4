import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pathlib,sys,pytest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from admission_scope import classify_calibration

def rows(arith=(0,1,0,1),extract=(1,1,1,1),complete=True):
 out=[]
 for fam,rews in (("arithmetic",arith),("extraction",extract)):
  for i in range(4):
   for j in range(4): out.append({"task":f"cap-training-{fam}-{i:02d}","family":fam,"split":"training","seed":721021+((i if fam=="arithmetic" else i+4)*10)+j,"reward":rews[i],"complete":complete})
 return out
def test_mixed_and_perfect_control():
 x=rows(arith=(0,0,0,0))
 for j in range(4): x[j]["reward"]=j%2
 x=classify_calibration(x); assert x["active_families"]==["arithmetic"]; assert x["retention_control_families"]==["extraction"]
def test_partial_wrong_count_and_duplicate_rejected():
 with pytest.raises(ValueError): classify_calibration(rows()[:-1])
 x=rows(); x[-1]["seed"]=x[-2]["seed"]
 with pytest.raises(ValueError): classify_calibration(x)
 x=rows(); x[0]["task"]="bad"
 with pytest.raises(ValueError): classify_calibration(x)
def test_all_zero_incomplete_and_no_active_rejected():
 with pytest.raises(ValueError): classify_calibration(rows(arith=(0,0,0,0),extract=(1,1,1,1)))
 with pytest.raises(ValueError): classify_calibration(rows(arith=(1,1,1,1),extract=(1,1,1,1)))
 with pytest.raises(ValueError): classify_calibration(rows(complete=False))
def test_binary_reward_strict():
 x=rows(); x[0]["reward"]=True
 with pytest.raises(ValueError): classify_calibration(x)
def test_cross_group_only_variation_rejected():
 x=rows(arith=(0,0,0,0),extract=(1,1,1,1))
 for i in range(4):
  for j in range(4): x[i*4+j]["reward"]=0 if i<2 else 1
 with pytest.raises(ValueError): classify_calibration(x)
def test_family_and_seed_binding_rejected():
 x=rows(); x[0]["family"]="extraction"
 with pytest.raises(ValueError): classify_calibration(x)
 x=rows(); x[0]["seed"]=999
 with pytest.raises(ValueError): classify_calibration(x)
