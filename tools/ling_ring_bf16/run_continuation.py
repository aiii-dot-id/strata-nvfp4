#!/usr/bin/env python3
"""AI Identity Incorported <james@aiii.id>.
Verified complete Ring/Ling continuation. GPU/process ownership remains with the caller.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, struct, tempfile, time, zlib
from pathlib import Path
import run_ling_verified as ling
import run_ring_verified as ring

REPO=Path(__file__).resolve().parents[2]
VOCAB=157184
LIMIT=512*1024*1024
class StateError(RuntimeError): pass

def require(ok,message):
    if not ok: raise StateError(message)
def sha(path): return ring._sha(Path(path))
def canonical(value): return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def read_json(path): return ling._read_json(Path(path),"continuation metadata")
def save_json(path,value): Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def state_info(path):
    import numpy as np
    path=Path(path);require(not path.is_symlink() and path.is_file(),"state payload must be a regular file")
    require(32<=path.stat().st_size<=LIMIT,"state size outside bound")
    data=path.read_bytes();end=len(data)-8
    expected_crc=struct.unpack_from('<Q',data,end)[0]
    require(zlib.crc32(memoryview(data)[:end])==expected_crc,'native state CRC32 checksum mismatch')
    cursor=0
    def number():
        nonlocal cursor
        require(cursor+8<=end,"truncated state number");x=struct.unpack_from('<Q',data,cursor)[0];cursor+=8;return x
    def text(maximum):
        nonlocal cursor
        n=number();require(n<=maximum and cursor+n<=end,"invalid state string")
        try: s=data[cursor:cursor+n].decode('ascii')
        except UnicodeError as ex: raise StateError('non-ASCII state metadata') from ex
        cursor+=n;return s
    def vector():
        nonlocal cursor
        n=number();require(n<=(end-cursor)//4,"invalid state array length")
        values=np.frombuffer(data,dtype='<f4',count=n,offset=cursor);require(bool(np.isfinite(values).all()),"nonfinite state array");cursor+=n*4;return n
    require(text(16)=='STRATA-CONT-1','unsupported state schema');binding=text(64)
    require(len(binding)==64 and all(c in '0123456789abcdef' for c in binding),'invalid state binding')
    position=number();count=number();require(count==position and count<=4096,'invalid state position')
    prefix=[number() for _ in range(count)];require(all(0<=x<VOCAB for x in prefix),'invalid state token')
    logits=vector();require(logits==(VOCAB if position else 0),'state logits shape')
    arrays={};n=number();require(n<=128,'too many state arrays')
    for _ in range(n):
        name=text(128);require(name and all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in name),'invalid state array name');require(name not in arrays,'duplicate state array');arrays[name]=vector()
    require(cursor==end,'trailing state bytes')
    return {'binding_id':binding,'position':position,'prefix':prefix,'last_logits_count':logits,'arrays':arrays,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def load_snapshot(directory):
    directory=Path(directory)
    require(not directory.is_symlink() and directory.is_dir(),'snapshot directory is missing or linked')
    require({p.name for p in directory.iterdir()}=={'state.bin','manifest.json'},'snapshot components missing or extra')
    require(not (directory/'manifest.json').is_symlink(),'snapshot manifest is linked')
    meta=read_json(directory/'manifest.json');info=state_info(directory/'state.bin')
    require(meta.get('schema')=='strata-continuation-snapshot-v1','snapshot schema')
    require(meta.get('binding_id')==digest(meta.get('binding'))==info['binding_id'],'snapshot binding mismatch')
    require(canonical(meta.get('state'))==canonical(info),'snapshot payload metadata mismatch')
    return meta

def fork_snapshot(source,destination):
    source=Path(source);destination=Path(destination);before=load_snapshot(source)
    require(not destination.exists() and not destination.is_symlink(),'fork destination exists')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.mkdir()  # Atomic ownership claim; an existing path is never replaced.
    try:
        shutil.copyfile(source/'state.bin',destination/'state.bin')
        shutil.copyfile(source/'manifest.json',destination/'manifest.json')
        require(load_snapshot(destination)==before and load_snapshot(source)==before,'snapshot drift during fork')
    except BaseException:
        shutil.rmtree(destination)
        raise
    require((source/'state.bin').stat().st_ino!=(destination/'state.bin').stat().st_ino,'fork must own independent payload bytes')
    return before

def frontend_module(model):
    return __import__("generate_"+model)

def profile(model,cache_mib):
    common={'matmul_backend':'fp32-cublas','lm_head_residency':'cuda','cache_budget_bytes':cache_mib*1024*1024}
    if model=='ring':return dict(common,rope_frequency='original',router_backend='torch-cuda',rms_backend='torch-cuda',attention_backend='torch-cuda',pointwise_backend='torch-cuda',attention_value_order='native',attention_score_scale='native')
    return dict(common,rope_frequency='native',rms_mode='torch-cuda-order',pointwise_mode='torch-cuda')

def input_bindings(model,checkpoint,manifest,exe,frontend):
    if model=='ring':x=ring._all_inputs(checkpoint,manifest,exe,REPO,frontend,'torch-cuda','torch-cuda','torch-cuda','torch-cuda',4)
    else:x=ling._all_inputs(checkpoint,manifest,exe,REPO,frontend,'fp32-cublas','torch-cuda',4)
    x['continuation_source']={str(p.relative_to(REPO)): {'sha256':sha(p),'size':p.stat().st_size} for p in [Path(__file__),REPO/'include/strata/ring/continuation.hpp']}
    return x

def finite_file(path,size):
    import numpy as np
    require(path.is_file() and not path.is_symlink() and path.stat().st_size==size,'output size '+path.name)
    with path.open('rb') as f:
        while True:
            data=f.read(4*1024*1024)
            if not data:break
            require(len(data)%4==0 and bool(np.isfinite(np.frombuffer(data,dtype='<f4')).all()),'nonfinite output '+path.name)

def validate_outputs(output,model,start,end):
    layers,hidden,experts=(20,2048,256) if model=='ring' else (24,1536,128)
    result={};count=end-start
    for name,width in [(f'layer-{i}.f32',hidden) for i in range(layers)]+[('final-hidden.f32',hidden),('logits.f32',VOCAB)]:
        p=output/name
        finite_file(p,count*width*4);result[name]={'sha256':sha(p),'size':p.stat().st_size}
    p=output/'routes.tsv';rows=p.read_text().splitlines();require(len(rows)==count*(layers-1),'route row count')
    for n,row in enumerate(rows):
        fields=row.split('\t');pos=start+n//(layers-1);layer=1+n%(layers-1)
        require(len(fields)==10 and fields[:2]==[str(pos),str(layer)],'route ordering/position')
        ids=[]
        for item in fields[2:]:
            pair=item.split(':');require(len(pair)==2,'route weight syntax');i=int(pair[0]);v=float(pair[1]);require(0<=i<experts and math.isfinite(v) and v>=0,'invalid route');ids.append(i)
        require(len(set(ids))==8,'duplicate selected expert')
    result[p.name]={'sha256':sha(p),'size':p.stat().st_size}
    for name in ['execution.json','continuation.json']:
        p=output/name;result[name]={'sha256':sha(p),'size':p.stat().st_size}
    return result

def run(model,checkpoint,manifest,executable,output,*,tokens=None,prompt=None,system=None,thinking='on',max_new_tokens=0,eos_ids=None,state_in=None,stop_after=None,cache_mib=1024,timeout=600):
    started=time.monotonic();require(model in {'ring','ling'},'unknown model')
    cp,manifest,exe,out=[Path(p).resolve() for p in [checkpoint,manifest,executable,output]]
    require(not out.exists(),'output already exists');require(type(cache_mib) is int and 6<=cache_mib<=2048,'cache size');require(type(timeout) in (int,float) and math.isfinite(timeout) and 0<timeout<=3600,'timeout')
    require((tokens is None)!=(prompt is None),'choose tokens or prompt');require(type(max_new_tokens) is int and max_new_tokens>=0,'generation budget')
    require(stop_after is None or type(stop_after) is int and stop_after>=0,'invalid stop position')
    module=ring if model=='ring' else ling
    front=frontend_module(model);bindings=front._binding_paths(cp);frontend_before={str(p):sha(p) for p in bindings};tokenizer=None
    if prompt is not None:
        tokenizer=front._tokenizer(cp);messages=([{'role':'system','content':system}] if system is not None else [])+[{'role':'user','content':prompt}];kw={'tokenize':True,'add_generation_prompt':True}
        if model=='ling':kw['enable_thinking']=thinking=='on'
        token_ids=front._ids(tokenizer.apply_chat_template(messages,**kw))
    else:token_ids=module._tokens(tokens)
    capacity=(4096 if model=='ring' else 1024)
    require(token_ids and len(token_ids)+max_new_tokens<=capacity,'context bound')
    explicit=([int(x) for x in eos_ids.split(',')] if isinstance(eos_ids,str) else eos_ids)
    if not max_new_tokens:eos=[]
    elif model=='ring':eos=explicit if explicit is not None else front._eos_ids(cp)
    else:eos=front._eos_ids(cp,explicit)[0]
    require(not max_new_tokens or isinstance(eos,list) and eos and len(set(eos))==len(eos) and all(type(x) is int and 0<=x<VOCAB for x in eos),'invalid EOS contract')
    require(frontend_before=={str(p):sha(p) for p in bindings},'frontend changed during rendering')
    incoming=load_snapshot(state_in) if state_in else None
    if incoming:
        prefix=incoming['state']['prefix'];require(prefix[:min(len(prefix),len(token_ids))]==token_ids[:min(len(prefix),len(token_ids))],'snapshot prefix mismatch')
        require(len(prefix)<=len(token_ids)+max_new_tokens,'snapshot exceeds capacity')
    with tempfile.TemporaryDirectory(prefix=model+'-continuation-manifest-') as tmp:
        module.prepare(cp,Path(tmp)/'admission');require((Path(tmp)/'admission/manifest.tsv').read_bytes()==manifest.read_bytes(),'fresh manifest mismatch')
    before=input_bindings(model,cp,manifest,exe,bindings)
    prof=profile(model,cache_mib);binding={'schema':'strata-continuation-binding-v1','model':model,'inputs':before,'profile':prof,'activation_dtype':'FP32','quantized':False,'eos_token_ids':eos,'adapter':'none'};bid=digest(binding)
    if incoming:require(incoming['binding']==binding and incoming['binding_id']==bid,'snapshot model/executable/source/tokenizer/profile mismatch')
    start=incoming['state']['position'] if incoming else 0
    require(stop_after is None or start<=stop_after<=len(token_ids)+max_new_tokens,'stop position outside current request')
    argv=[str(exe),str(cp),str(manifest),','.join(map(str,token_ids)),str(out),str(cache_mib),sha(manifest),'--matmul','fp32-cublas','--lm-head-residency','cuda','--state-binding',bid]
    if model=='ring':argv+=['--router','torch-cuda','--rms-mode','torch-cuda','--rope-frequency','original','--attention-backend','torch-cuda','--pointwise','torch-cuda']
    else:argv+=['--rms-mode','torch-cuda-order','--pointwise','torch-cuda']
    if max_new_tokens:argv+=['--generate',str(max_new_tokens),'--eos',','.join(map(str,eos))]
    if incoming:argv+=['--state-in',str(Path(state_in).resolve()/'state.bin')]
    argv+=['--state-out',str(out/'native-state.bin')]
    if stop_after is not None:argv+=['--stop-after',str(stop_after)]
    rc,stdout,stderr=module._run_owned(argv,timeout)
    require(rc==0,'native continuation failed: '+stderr[-1500:])
    after=input_bindings(model,cp,manifest,exe,bindings);require(before==after,'checkpoint/executable/source drift')
    if incoming:require(load_snapshot(state_in)==incoming,'input snapshot mutated')
    c=read_json(out/'continuation.json');e=read_json(out/'execution.json')
    end=c.get('processed_position');require(type(end) is int and start<=end<=len(token_ids)+max_new_tokens,'invalid processed position');require(c.get('start_position')==start and c.get('processed_tokens')==end-start and type(c.get('paused')) is bool,'continuation accounting')
    require(e.get('status')=='executed' and e.get('model')==('Ring-mini-2.0' if model=='ring' else 'Ling-3.0-tiny'),'native identity');require(e.get('checkpoint_id')==sha(manifest) and e.get('quantized') is False and e.get('activation_dtype')=='FP32','native checkpoint/precision')
    require(e.get('tokens')==end and e.get('logits_tokens')==end-start and len(e.get('token_ids',[]))==end,'native token accounting')
    for k,v in prof.items():require(e.get(k)==v,'native profile '+k)
    require(e.get('lm_head_resident_uploads')==1 and e.get('lm_head_resident_apply_calls')==end-start,'native head calls')
    prefix=e['token_ids'];require(prefix[:min(end,len(token_ids))]==token_ids[:min(end,len(token_ids))],'native prompt trace')
    if incoming:require(prefix[:start]==incoming['state']['prefix'],'native snapshot prefix drift')
    native=validate_outputs(out,model,start,end)
    info=state_info(out/'native-state.bin');require(info['prefix']==prefix and info['position']==end and info['binding_id']==bid,'output state binding')
    bundle=out/'snapshot';bundle.mkdir();shutil.copyfile(out/'native-state.bin',bundle/'state.bin');meta={'schema':'strata-continuation-snapshot-v1','binding':binding,'binding_id':bid,'state':info};save_json(bundle/'manifest.json',meta);require(load_snapshot(bundle)==meta,'snapshot readback failed')
    generation=None;decoded=None
    if max_new_tokens and not c['paused']:
        generation=module._require_generation(out/'generation.json',token_ids,max_new_tokens,eos);require(prefix==token_ids+generation['generated_token_ids'][:-1],'generation/committed trace mismatch');p=out/'generation.json';native[p.name]={'sha256':sha(p),'size':p.stat().st_size}
        if tokenizer is None:tokenizer=front._tokenizer(cp)
        decoded=tokenizer.decode(generation['generated_token_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False);(out/'generated.txt').write_text(decoded)
    if not max_new_tokens and not c['paused']:require(prefix==token_ids,'teacher-forced trace mismatch')
    receipt={'schema':'strata-verified-continuation-run-v1','status':'verified','model':model,'inputs':before,'binding_id':bid,'binding':binding,'execution':e,'continuation':c,'native_outputs':native,'generation':generation,'decoded':decoded,'snapshot':str(out/'snapshot'),'input_snapshot':str(state_in) if incoming else None,'input_snapshot_unchanged':True if incoming else None,'verification_hash_workers':4,'charged_wrapper_wall_seconds':time.monotonic()-started,'argv':argv,'scope':'complete live greedy continuation; no expert/head warm-cache persistence, training, serving or quality claim'}
    save_json(out/'provenance.json',receipt);return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--fork-source',type=Path);p.add_argument('--fork-destination',type=Path)
    p.add_argument('--model',choices=['ring','ling']);p.add_argument('--checkpoint',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--executable',type=Path);p.add_argument('--output',type=Path);p.add_argument('--tokens');p.add_argument('--prompt');p.add_argument('--system');p.add_argument('--thinking',choices=['on','off'],default='on');p.add_argument('--max-new-tokens',type=int,default=0);p.add_argument('--eos-ids');p.add_argument('--state-in',type=Path);p.add_argument('--stop-after',type=int);p.add_argument('--cache-mib',type=int,default=1024);p.add_argument('--timeout',type=float,default=600)
    a=p.parse_args()
    try:
        if a.fork_source or a.fork_destination:
            require(a.fork_source and a.fork_destination,'fork needs source and destination');fork_snapshot(a.fork_source,a.fork_destination);print(json.dumps({'forked':True}));return
        require(all([a.model,a.checkpoint,a.manifest,a.executable,a.output]),'model/checkpoint/manifest/executable/output required')
        r=run(a.model,a.checkpoint,a.manifest,a.executable,a.output,tokens=a.tokens,prompt=a.prompt,system=a.system,thinking=a.thinking,max_new_tokens=a.max_new_tokens,eos_ids=a.eos_ids,state_in=a.state_in,stop_after=a.stop_after,cache_mib=a.cache_mib,timeout=a.timeout);print(json.dumps({'status':r['status'],'continuation':r['continuation'],'decoded':r['decoded']}))
    except (OSError,ValueError,StateError,ring.VerificationError,ling.VerificationError) as ex:p.error(str(ex))
if __name__=='__main__':main()
