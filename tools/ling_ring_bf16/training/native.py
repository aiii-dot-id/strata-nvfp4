"""Bound native Strata runs for terminal adapter experiments.
AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations
import hashlib, json, math, os, signal, subprocess, time, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path('/work/AI/experiments/strata-terminal-training-20261021-r1')
MODELS = {
 'ling': {'checkpoint':'/work/AI/checkpoints/Ling-3.0-tiny','manifest':'/work/AI/experiments/ling-strata-20261016/admission/manifest.tsv','hidden':1536,'layers':24,'capacity':1024,'profile':['--matmul','fp32-cublas','--rms-mode','torch-cuda-order','--pointwise','torch-cuda']},
 'ring': {'checkpoint':'/work/AI/checkpoints/Ring-mini-2.0','manifest':'/work/AI/experiments/ring-strata-20261016/admission/manifest.tsv','hidden':2048,'layers':20,'capacity':4096,'profile':['--matmul','fp32-cublas','--router','torch-cuda','--rms-mode','torch-cuda','--rope-frequency','original','--attention-backend','torch-cuda','--pointwise','torch-cuda']}}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def _source_closure(root):
    """Hash tracked and non-ignored untracked source inputs, including dirty files."""
    cmd=['git','-C',str(root),'ls-files','--cached','--others','--exclude-standard','--',
         'tools/ling_ring_bf16','include','src','CMakeLists.txt','cmake']
    names=subprocess.check_output(cmd,text=True).splitlines()
    out={}
    for name in sorted(set(names)):
        p=root/name
        if p.is_file(): out[name]={'size':p.stat().st_size,'sha256':sha(p)}
    return out

def _git_state(root):
    status=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=all'],text=True)
    return {'commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
            'dirty':bool(status),'status':status.splitlines()}

def dump(p,v):
 p=Path(p)
 with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

def finite_file(path, floats):
 p=Path(path)
 if p.stat().st_size != floats*4:raise ValueError('wrong float payload size '+str(p))
 if floats:
  a=np.memmap(p,dtype='<f4',mode='r')
  for offset in range(0,len(a),1<<20):
   if not np.isfinite(a[offset:offset+(1<<20)]).all():raise ValueError('nonfinite payload '+str(p))

def _validate_sampling_receipt(generation, tokens, generated, temperature, seed, logits_path):
 s=generation.get('sampling')
 if not isinstance(s,dict) or s.get('method')!='categorical_full_vocab' or s.get('algorithm')!='mt19937_64_double_max_shift':
  raise RuntimeError('generation sampling receipt method mismatch')
 receipt_temp=s.get('temperature'); receipt_seed=s.get('seed')
 if (isinstance(receipt_temp,bool) or not isinstance(receipt_temp,(int,float)) or
     not math.isfinite(float(receipt_temp)) or float(receipt_temp)<=0 or
     not isinstance(receipt_seed,int) or isinstance(receipt_seed,bool) or
     not 0<=receipt_seed<1<<64 or receipt_seed!=seed or float(receipt_temp)!=float(temperature)):
  raise RuntimeError('generation sampling receipt binding mismatch')
 uniforms=s.get('uniform_draws'); logprobs=s.get('sampled_logprobs')
 if not isinstance(uniforms,list) or not isinstance(logprobs,list) or len(uniforms)!=len(generated) or len(logprobs)!=len(generated):
  raise RuntimeError('generation sampling audit length mismatch')
 row_count=len(tokens)+len(generated)-1
 if Path(logits_path).stat().st_size != row_count*157184*4:
  raise RuntimeError('sampling logits payload shape mismatch')
 rows=np.memmap(logits_path,dtype='<f4',mode='r',shape=(row_count,157184))
 for i,(token,u,recorded) in enumerate(zip(generated,uniforms,logprobs)):
  if type(token) is not int or not 0<=token<157184:
   raise RuntimeError('generation sampling token invalid')
  if type(u) not in (int,float) or not math.isfinite(float(u)) or not 0 <= float(u) < 1:
   raise RuntimeError('generation sampling uniform invalid')
  if type(recorded) not in (int,float) or not math.isfinite(float(recorded)):
   raise RuntimeError('generation sampled logprob invalid')
  row=rows[len(tokens)-1+i]; maximum=float(np.max(row)); weights=[]; total=0.0
  for value in row:
   scaled=(float(value)-maximum)/temperature
   weight=math.exp(scaled) if math.isfinite(scaled) and scaled > -745.0 else 0.0
   weights.append(weight); total += weight
  if not total>0 or not math.isfinite(total):
   raise RuntimeError('generation sampling probability mass invalid')
  target=float(u)*total; cumulative=0.0; chosen=len(weights)-1
  for j,weight in enumerate(weights):
   cumulative += weight
   if target < cumulative:
    chosen=j; break
  expected=(float(row[token])-maximum)/temperature-math.log(total)
  if chosen!=token or abs(expected-float(recorded))>1e-10:
   raise RuntimeError('generation sampling audit mismatch')

def _validate_generation(generation, tokens, generate, eos, actual, sampling_temperature=None, sampling_seed=None, logits_path=None):
 if generation.get('schema') not in ('strata-ling-generation-v1','strata-ring-generation-v1'): raise RuntimeError('generation schema mismatch')
 if generation.get('prompt_token_ids')!=tokens or generation.get('eos_token_ids')!=list(eos) or generation.get('max_new_tokens')!=generate: raise RuntimeError('generation binding mismatch')
 ids=generation.get('generated_token_ids'); reason=generation.get('finish_reason')
 if not isinstance(ids,list) or not ids or len(ids)>generate or any(type(x)is not int or not 0<=x<157184 for x in ids): raise RuntimeError('generated IDs invalid')
 if any(x in eos for x in ids[:-1]) or reason not in ('eos','length'): raise RuntimeError('generation terminal semantics invalid')
 if reason=='eos' and ids[-1] not in eos: raise RuntimeError('generation EOS mismatch')
 if reason=='length' and (len(ids)!=generate or ids[-1] in eos): raise RuntimeError('generation length mismatch')
 if actual!=tokens+ids[:-1]:raise RuntimeError('generated/committed mismatch')
 if sampling_temperature is None:
  if 'sampling' in generation: raise RuntimeError('unexpected sampling receipt')
 elif logits_path is None:
  raise RuntimeError('sampling logits are required for audit')
 else:
  _validate_sampling_receipt(generation,tokens,ids,sampling_temperature,sampling_seed,logits_path)

def _validate_adapter_meta(adapter, model, manifest_sha):
 p=Path(str(adapter)+'.json')
 try: meta=json.loads(p.read_text())
 except Exception as ex: raise RuntimeError('adapter metadata missing or invalid') from ex
 if meta.get('model')!=model or meta.get('checkpoint_sha256')!=manifest_sha:
  raise RuntimeError('adapter metadata model/checkpoint binding mismatch')
 if type(meta.get('hidden')) is not int or meta['hidden']!=MODELS[model]['hidden']:
  raise RuntimeError('adapter metadata hidden size mismatch')
 return meta

def run(model, tokens, output, *, executable=None, adapter=None, capture=False,
        generate=0, eos=(), timeout=300, cache_mib=1024, checkpoint_binding=None,
        sampling_temperature=None, sampling_seed=None):
 if model not in MODELS:raise ValueError('model')
 c=MODELS[model]; tokens=list(tokens);out=Path(output)
 if not tokens or any(type(t)is not int or not 0<=t<157184 for t in tokens):raise ValueError('tokens')
 if type(generate)is not int or generate<0 or len(tokens)+generate>c['capacity']:raise ValueError('capacity')
 if generate and (not eos or any(type(t)is not int or not 0<=t<157184 for t in eos)):raise ValueError('eos')
 if (sampling_temperature is None) != (sampling_seed is None):raise ValueError('sampling temperature and seed must be supplied together')
 if sampling_temperature is not None:
  if isinstance(sampling_temperature,bool) or not isinstance(sampling_temperature,(int,float)) or not math.isfinite(float(sampling_temperature)) or float(sampling_temperature)<=0:raise ValueError('sampling_temperature must be positive finite nonbool')
  if not isinstance(sampling_seed,int) or isinstance(sampling_seed,bool) or not 0<=sampling_seed<1<<64:raise ValueError('sampling_seed must be uint64')
  if not generate:raise ValueError('sampling requires generate')
 if out.exists():raise FileExistsError(out)
 if shutil.disk_usage(out.parent if out.parent.exists() else EXPERIMENT).free<10*(1<<30):raise RuntimeError('less than 10 GiB free; retain evidence and stop')
 out.parent.mkdir(parents=True,exist_ok=True)
 exe=Path(executable or EXPERIMENT/f'build/{model}-torch/strata_{model}_bf16')
 manifest=Path(c['manifest']); manifest_sha=sha(manifest)
 if checkpoint_binding is not None:
  if not isinstance(checkpoint_binding,str) or len(checkpoint_binding)!=64 or any(x not in '0123456789abcdef' for x in checkpoint_binding): raise ValueError('checkpoint_binding must be lowercase SHA-256')
  if checkpoint_binding!=manifest_sha: raise ValueError('checkpoint binding does not match manifest')
 source_before=_source_closure(ROOT); state_before=_git_state(ROOT)
 pins={str(exe):sha(exe),str(manifest):manifest_sha,str(Path(__file__)):sha(__file__)}
 if adapter:
  _validate_adapter_meta(adapter,model,manifest_sha)
  pins[str(Path(adapter))]=sha(adapter);pins[str(Path(str(adapter)+'.json'))]=sha(str(adapter)+'.json')
 argv=[str(exe),c['checkpoint'],str(manifest),','.join(map(str,tokens)),str(out),str(cache_mib),sha(manifest),*c['profile'],'--lm-head-residency','cuda']
 if capture:argv+=['--terminal-capture','1']
 if adapter:argv+=['--terminal-adapter',str(adapter)]
 if generate:argv+=['--generate',str(generate),'--eos',','.join(map(str,eos))]
 if sampling_temperature is not None:argv+=['--sampling-temperature',repr(float(sampling_temperature)),'--sampling-seed',str(sampling_seed)]
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 stdout=out.with_name(out.name+'.stdout.log');stderr=out.with_name(out.name+'.stderr.log')
 start=time.monotonic();rss=0;proc=None;failure=None;interrupted=[]
 previous={}
 def cancel(signum, _frame): interrupted.append(signum)
 try:
  for signum in (signal.SIGINT,signal.SIGTERM):
   previous[signum]=signal.getsignal(signum);signal.signal(signum,cancel)
  with stdout.open('xb') as so,stderr.open('xb') as se:
   proc=subprocess.Popen(argv,stdout=so,stderr=se,start_new_session=True,env=env)
   while proc.poll() is None:
    if interrupted: raise RuntimeError(f'cancelled by {signal.Signals(interrupted[0]).name}')
    if time.monotonic()-start>timeout:raise TimeoutError(f'native timeout {timeout}')
    try:
     status=Path(f'/proc/{proc.pid}/status').read_text()
     for line in status.splitlines():
      if line.startswith('VmHWM:'):rss=max(rss,int(line.split()[1])*1024)
    except (FileNotFoundError,ProcessLookupError):pass
    time.sleep(.1)
  if interrupted: raise RuntimeError(f'cancelled by {signal.Signals(interrupted[0]).name}')
  if proc.returncode:raise RuntimeError(f'native return {proc.returncode}: '+stderr.read_text()[-2000:])
  if any(sha(p)!=h for p,h in pins.items()):raise RuntimeError('input changed during run')
  source_after=_source_closure(ROOT); state_after=_git_state(ROOT)
  if source_before!=source_after: raise RuntimeError('source closure changed during run')
  e=json.loads((out/'execution.json').read_text())
  if e.get('status')!='executed' or e.get('quantized') is not False or e.get('activation_dtype')!='FP32':raise RuntimeError('native precision/status')
  if e.get('checkpoint_id')!=sha(manifest):raise RuntimeError('checkpoint identity')
  actual=e.get('token_ids',[]);n=len(actual)
  if actual[:len(tokens)]!=tokens or e.get('tokens')!=n or e.get('logits_tokens')!=n:raise RuntimeError('token/row accounting')
  generation=json.loads((out/'generation.json').read_text()) if generate else None
  if generate:
   _validate_generation(generation,tokens,generate,eos,actual,sampling_temperature,sampling_seed,out/'logits.f32')
  elif actual!=tokens:raise RuntimeError('teacher forced mismatch')
  for name,width in [('logits.f32',157184),('final-hidden.f32',c['hidden'])]+[(f'layer-{i}.f32',c['hidden']) for i in range(c['layers'])]:finite_file(out/name,n*width)
  if capture:finite_file(out/'terminal.f32',n*(11*c['hidden']+8208))
  payloads={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}
  if e.get('terminal_adapter') is not bool(adapter): raise RuntimeError('adapter receipt binding mismatch')
  rec={'schema':'strata-terminal-native-run-v1','passed':True,'model':model,'argv':argv,'input_pins':pins,'execution':e,'generation':generation,'capture':capture,'adapter':str(adapter) if adapter else None,'checkpoint_binding':manifest_sha,'native_payloads':payloads,'wall_seconds':time.monotonic()-start,'native_process_hwm_bytes':rss,'source_state_before':state_before,'source_commit':state_after['commit'],'source_dirty':state_after['dirty'],'source_status':state_after['status'],'source_closure':source_after}
  dump(out/'RUN.json',rec);return rec
 except BaseException as ex:
  failure=str(ex)
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(5)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(5)
  failed=out.with_name(out.name+'.FAILED.json')
  if not failed.exists():dump(failed,{'passed':False,'error':failure,'argv':argv,'wall_seconds':time.monotonic()-start,'native_process_hwm_bytes':rss})
  raise
 finally:
  for signum,handler in previous.items(): signal.signal(signum,handler)

def exact_native(a,b):
 a,b=Path(a),Path(b)
 names=[p.name for p in a.glob('*.f32') if p.name!='terminal.f32']+['routes.tsv']
 diffs=[name for name in names if sha(a/name)!=sha(b/name)]
 if diffs:raise ValueError('native payload mismatch: '+','.join(diffs))
 return {'exact_files':len(names)}
