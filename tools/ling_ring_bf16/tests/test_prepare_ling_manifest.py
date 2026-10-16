import importlib.util, tempfile, unittest, json, struct
from pathlib import Path
spec=importlib.util.spec_from_file_location('ling','/work/AI/src/strata-ling-ring-bf16/tools/ling_ring_bf16/prepare_ling_manifest.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def config():
 c=dict(m.EXPECTED_CONFIG); c.update({'architectures':['BailingMoeV3ForCausalLM']}); return c
class TestLingAdmission(unittest.TestCase):
 def test_exact_config_and_schema_rejects_missing(self):
  m.validate_config(config()); exp=m.expected_records(config()); self.assertGreater(len(exp),9000)
  with self.assertRaises(m.AdmissionError): m.validate_records(dict(list(exp.items())[:-1]),config())
 def test_shape_dtype_reject(self):
  exp=m.expected_records(config()); rows={k:{'dtype':v[0],'shape':list(v[1])} for k,v in exp.items()}
  rows['model.layers.1.mlp.gate.expert_bias']['dtype']='BF16'
  with self.assertRaises(m.AdmissionError): m.validate_records(rows,config())
 def test_mla_cannot_use_kda_schema(self):
  exp=m.expected_records(config()); rows={k:{"dtype":v[0],"shape":list(v[1])} for k,v in exp.items()}
  del rows['model.layers.3.attention.dense.weight']
  rows['model.layers.3.attention.A_log']={"dtype":"F32","shape":[16]}
  with self.assertRaises(m.AdmissionError): m.validate_records(rows,config())
 def test_wrong_headwise_gate_shape(self):
  exp=m.expected_records(config()); rows={k:{"dtype":v[0],"shape":list(v[1])} for k,v in exp.items()}
  rows['model.layers.3.attention.g_proj.weight']['shape']=[2048,1536]
  with self.assertRaises(m.AdmissionError): m.validate_records(rows,config())
 def test_overlapping_offsets_reject(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); header={"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[1,3]}}
   raw=json.dumps(header,separators=(',',':')).encode(); (root/'x.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+b'xxx')
   with self.assertRaises(m.AdmissionError): m._records(root,{'a':'x.safetensors','b':'x.safetensors'})
 def test_index_extra_reject(self):
  with self.assertRaises(m.AdmissionError): m.validate_index({'a':'x.safetensors'}, {'a':{'shard':'x.safetensors'},'extra':{'shard':'x.safetensors'}})
 def test_unsafe_shard_reject(self):
  for value in ('../x.safetensors','/tmp/x.safetensors','a/b.safetensors','x.txt'):
   with self.assertRaises(m.AdmissionError): m._safe_shard(value)
 def test_empty_weight_map_and_existing_output(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); (root/'config.json').write_text('{}'); (root/'model.safetensors.index.json').write_text('{"weight_map":{}}')
   out=root/'out'; out.mkdir()
   with self.assertRaises(m.AdmissionError): m.prepare(root,out)
if __name__=='__main__': unittest.main(verbosity=2)
