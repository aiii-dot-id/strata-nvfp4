"""CPU-only integration controls against immutable retained native calibration."""
import os,sys,copy,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pytest
import import_calibration as mod
import capability_tasks as tasks
from transformers import AutoTokenizer
SOURCE=Path(os.environ.get('STRATA_CALIBRATION_TEST_SOURCE','/nonexistent'))
pytestmark=pytest.mark.skipif(not (SOURCE/'PLAN.json').is_file(),reason='requires retained calibration evidence')
@pytest.fixture
def setup(tmp_path):
 old=mod.read(SOURCE/'PLAN.json');new=copy.deepcopy(old)
 # Positive replay controls use the original admitted source identity.
 # A bounded candidate is intentionally not admitted by this historical importer.
 new['allow_saturated_controls']=True;new['calibration_from']=str(SOURCE)
 plan=tmp_path/'PLAN.json';plan.write_text(json.dumps(new))
 suite=tasks.load_suite(SOURCE.parent/'SUITE.public.json',SOURCE.parent/'SUITE.private.json')
 calibration=[t for f in tasks.FAMILIES for t in [x for x in suite if x.split=='training' and x.family==f][:4]]
 tok=AutoTokenizer.from_pretrained(mod.native.MODELS['ling']['checkpoint'],local_files_only=True)
 return plan,tmp_path/'IMPORT.json',calibration,tok
def invoke(setup):return mod.import_calibration(SOURCE,*setup)
def test_actual_calibration_import(setup):
 rows=invoke(setup)
 assert len(rows)==32 and rows[0]['ids']==[19,20,156895]
 assert sum(r['grade']['reward'] for r in rows)==29
 assert rows[0]['features'].shape==(3,25104)
 assert rows[0]['logits'].shape==(3,157184)
 assert json.loads(setup[1].read_text())['training_data_reused'] is False
 with pytest.raises(ValueError,match='already exists'):invoke(setup)
@pytest.mark.parametrize('mutation',['missing_return_flag','dirty_return','updated','adapter','lp','prompt','native_source','seed','positions'])
def test_mutations_fail(setup,monkeypatch,mutation):
 original=mod.read
 def changed(path):
  x=original(path);p=Path(path)
  if mutation=='missing_return_flag' and p==SOURCE.with_suffix('.RETURN.json'):x.pop('foreign_gpu_preserved')
  if mutation=='dirty_return' and p==SOURCE.with_suffix('.RETURN.json'):x['error']='interrupted'
  if mutation=='updated' and p==SOURCE/'RESULT.json':x['updates']=1
  if p.name=='ROLLOUT.json':
   if mutation=='adapter':x['policy_adapter_sha256']='bad'
   if mutation=='lp':x['sampling_logprobs'][0]+=1
   if mutation=='prompt':x['prompt_ids'][0]+=1
   if mutation=='seed':x['seed']+=10000000
   if mutation=='positions':x['teacher_rows'][0]+=1
  if mutation=='native_source' and p.name=='RUN.json':x['source_dirty']=True
  return x
 monkeypatch.setattr(mod,'read',changed)
 with pytest.raises((ValueError,KeyError)):invoke(setup)
def test_retained_payload_hash_mutation(setup,monkeypatch):
 original=mod.native.sha
 monkeypatch.setattr(mod.native,'sha',lambda p:'0'*64 if Path(p).name=='selected-logits.f32' else original(p))
 with pytest.raises(ValueError,match='hash mismatch'):invoke(setup)


def test_current_candidate_source_is_not_silently_readmitted(setup):
 plan=setup[0];payload=mod.read(plan)
 payload['pins']['source']=mod.native._source_closure(mod.native.ROOT)
 plan.write_text(json.dumps(payload))
 with pytest.raises(ValueError,match='native or objective source drift'):
  invoke(setup)
