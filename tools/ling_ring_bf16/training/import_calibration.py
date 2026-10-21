"""Import completed Ling calibration for admission only, never optimizer updates.
AI Identity Incorported <james@aiii.id>.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from admission_scope import classify_calibration
import capability_tasks as tasks
import native

PREFIX='tools/ling_ring_bf16/training/'
ALLOWED_CHANGES={PREFIX+n for n in ('capability_runner.py','admission_scope.py','import_calibration.py',
 'test_admission_scope.py','test_import_calibration.py','test_nomination_scope.py')}
def read(p):return json.loads(Path(p).read_text())
def require(ok,reason):
 if not ok:raise ValueError(reason)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def import_calibration(source,new_plan,destination,calibration,tok):
 source=Path(source).resolve();destination=Path(destination)
 require(not destination.exists(),'import destination already exists')
 previous=read(source/'PLAN.json');current=read(new_plan)
 require(current.get('allow_saturated_controls') is True and current.get('calibration_from')==str(source),'explicit scoped reuse absent')
 require(previous.get('model')==current.get('model')=='ling','import qualified for Ling only')
 result=read(source/'RESULT.json');returned=read(source.with_suffix('.RETURN.json'))
 require(result.get('status')=='coverage_gate_failed' and result.get('updates')==0 and result.get('self_opd_run') is False and result.get('native_calls')==32,'source not clean completed calibration')
 require('error' in returned and returned.get('error') is None and all(returned.get(k) is True for k in ('checkpoint_exact','foreign_gpu_preserved')),'source restoration failed')
 require(returned.get('outcome')=='coverage_gate_failed' and returned.get('services_touched') is False and returned.get('mac_touched') is False,'source return outcome invalid')
 require(not list(source.rglob('*.adapter')) and not (source/'FROZEN-CANDIDATES.json').exists(),'source already trained')
 for key in ('suite','private','executable','manifest','native_profile','parent_checkpoint_digest'):
  require(previous['pins'].get(key) is not None and previous['pins'][key]==current['pins'].get(key),'pin drift: '+key)
 require(previous['completion']==current['completion'],'completion changed')
 require(previous['seed']==current['seed'] and previous['temperature']==current['temperature'] and previous['samples_per_prompt']==current['samples_per_prompt']==4,'sampling changed')
 oldsrc=previous['pins']['source'];newsrc=current['pins']['source']
 changed={k for k in set(oldsrc)|set(newsrc) if oldsrc.get(k)!=newsrc.get(k)}
 require(changed<=ALLOWED_CHANGES,'native or objective source drift: '+str(sorted(changed-ALLOWED_CHANGES)))
 before=read(source.with_suffix('.binding-before.json'));after=read(source.with_suffix('.binding-after.json'))
 require(before==after and digest(before)==previous['pins']['parent_checkpoint_digest'],'checkpoint binding mismatch')
 expected={t.id:t for t in calibration}
 require(len(expected)==8,'calibration task count')
 found=sorted(source.rglob('ROLLOUT.json'));require(len(found)==32,'unexpected rollout count')
 loaded=[];pins={}
 def bind(path,sha=None):
  path=Path(path);actual=native.sha(path)
  require(sha is None or actual==sha,'artifact hash mismatch: '+str(path));pins[str(path)]=actual
 for path in [source/'PLAN.json',source/'RESULT.json',source.with_suffix('.RETURN.json'),source.with_suffix('.binding-before.json'),source.with_suffix('.binding-after.json')]:bind(path)
 for rp in found:
  r=read(rp);bind(rp)
  require(r.get('task') in expected and r.get('arm')=='calibration' and r.get('split')=='training' and r.get('greedy') is False and r.get('policy_adapter_sha256') is None,'not baseline calibration')
  task=expected[r['task']]
  require(r.get('family')==task.family and r.get('level')==task.level,'task metadata mismatch')
  idx=list(expected).index(task.id);sample=r['seed']-previous['seed']-idx*10
  require(type(sample)is int and 0<=sample<4,'sample seed mismatch')
  require(rp==source/f'calibration/round-0/{task.id}/sample-{sample}/ROLLOUT.json','rollout path mismatch')
  text=tasks.prompt(task)+'\n\nFinal answer: write only the requested literal on one line. No labels, markdown, or extra text.'
  rendered=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  prompt=tok.encode(rendered,add_special_tokens=False)
  require(r['prompt_ids']==prompt and r['prompt_sha256']==hashlib.sha256(rendered.encode()).hexdigest(),'prompt mismatch')
  require(len(r['native_stages'])==1 and not r['inserted_ids'] and r['forced_close'] is False,'Ling stage layout')
  sd=rp.parent/'completion';s=r['native_stages'][0]
  require(Path(s['path'])==sd,'stage owner mismatch');bind(sd/'RUN.json',s['run_sha256']);bind(sd/'RETENTION.json',s['retention_sha256'])
  run=read(sd/'RUN.json');ret=read(sd/'RETENTION.json');meta=read(sd/'terminal-meta.json');bind(sd/'terminal-meta.json')
  require(run.get('passed') is True and run.get('model')=='ling' and run.get('adapter') is None and run.get('capture') is True,'native run invalid')
  require(run.get('source_closure')==oldsrc and run.get('source_dirty') is False and run.get('source_status')==[],'native source mismatch')
  require(run['checkpoint_binding']==previous['pins']['manifest'],'native manifest mismatch')
  for f,h in run['input_pins'].items():bind(f,h)
  require(run['input_pins'].get(run['argv'][0])==previous['pins']['executable'],'native executable mismatch')
  ex=run['execution'];g=run['generation']
  require(ex['quantized'] is False and ex['activation_dtype']=='FP32' and ex['terminal_adapter'] is False,'precision/adapter mismatch')
  ids=g['generated_token_ids'];positions=list(range(len(prompt),len(prompt)+len(ids)));rows=[x-1 for x in positions]
  require(g['prompt_token_ids']==prompt and g['max_new_tokens']==previous['completion']['max_new_tokens'],'native prompt/budget mismatch')
  require(g['finish_reason']=='eos' and ids and ids[-1] in previous['completion']['eos_token_ids'],'incomplete calibration')
  require(r['complete'] is True and r['format_error'] is None and r['ids']==ids and r['action_positions']==positions and r['teacher_rows']==rows and r['transcript_ids']==prompt+ids,'action alignment mismatch')
  require(ex['token_ids']==(prompt+ids)[:-1],'native executed tokens mismatch')
  sampling=g['sampling'];require(sampling['seed']==r['seed']+1000000 and sampling['temperature']==previous['temperature'],'native sampling mismatch')
  require(r['sampling_logprobs']==sampling['sampled_logprobs'],'logprob receipt mismatch')
  raw=tok.decode(ids[:-1],skip_special_tokens=False,clean_up_tokenization_spaces=False)
  require(r['raw_text']==raw and r['text']==raw.strip() and r['grade']==tasks.grade(task,raw.strip(),True),'decode or grade mismatch')
  require(abs(r['wall_seconds']-run['wall_seconds'])<1e-9,'wall accounting mismatch')
  require(meta['model']=='ling' and meta['hidden']==1536 and meta['rowfloats']==25104 and meta['adapterpath'] is None,'feature geometry mismatch')
  arrays=[]
  for name,width in [('selected-terminal.f32',25104),('selected-logits.f32',157184)]:
   entry=ret['retained'][name];p=sd/name
   require(entry['positions']==rows and entry['bytes']==len(ids)*width*4 and p.stat().st_size==entry['bytes'],'retained shape mismatch')
   bind(p,entry['sha256']);a=np.fromfile(p,dtype='<f4').reshape(len(ids),width)
   require(np.isfinite(a).all(),'nonfinite retained values');arrays.append(a)
  logits=arrays[1].astype(np.float64)/previous['temperature'];maximum=logits.max(axis=1)
  lps=logits[np.arange(len(ids)),ids]-maximum-np.log(np.exp(logits-maximum[:,None]).sum(axis=1))
  require(np.max(np.abs(lps-np.asarray(r['sampling_logprobs'])))<1e-9,'retained logprob mismatch')
  loaded.append(dict(r,path=rp.parent,group=task.id,features=torch.from_numpy(arrays[0]),logits=arrays[1]))
 scope=classify_calibration(loaded)
 # Stable ordering preserves the original zero-adapter and rollback reference.
 loaded.sort(key=lambda r:(list(expected).index(r['task']),r['seed']))
 native.dump(destination,{'schema':'strata-calibration-import-v2','source':str(source),'changed_campaign_files':sorted(changed),
  'source_result_status':result['status'],'source_files_sha256':pins,'scope':scope,'record_count':32,
  'new_plan_sha256':native.sha(new_plan),'training_data_reused':False,'use':'admission, expert nomination, and zero/rollback reference only'})
 return loaded
