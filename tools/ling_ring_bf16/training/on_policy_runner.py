"""Native Strata self-OPD and verifier-reward policy-gradient qualification.
AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations
import argparse, copy, fcntl, gc, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native, terminal, qualify, on_policy_losses as losses, on_policy_tasks as tasks
from transformers import AutoTokenizer

EXP=Path('/work/AI/experiments/strata-on-policy-20261021-r1')
ARMS=('rlvr','self_opd')
CONFIG={'schema':'strata-on-policy-qualification-v1','rounds':2,'samples_per_prompt':4,
        'temperature':1.,'max_new_tokens':32,'lr':0.001,'rank':8,'alpha':16,
        'init_seed':61021,'rollout_seed':41021,'eval_seed':41022,
        'max_rlvr_batches_per_round':3,'native_timeout_seconds':300,
        'max_job_seconds':14400,'trainable_scope':'one final-layer expert gate/up/down LoRA',
        'self_opd':'mean full-vocabulary reverse KL at fresh student-sampled prefixes; frozen RLVR-trained endpoint of same base as teacher; identical original prompt; fixed-prefix semi-gradient',
        'rlvr':'negative mean leave-one-out advantage times sequence-summed sampled log probability; one update per freshly collected round',
        'restore_audit':'replay second batch only to compare uninterrupted and restored state; not another on-policy update',
        'teacher':'RLVR endpoint frozen before new zero-adapter Self-OPD student; no reference-answer augmentation',
        'evaluation':'paired seeds and equal answer-only token allowance; 8 held-out and 4 retention tasks, after both candidate freezes',
        'non_claims':['not full-model training','not a broad capability benchmark','not a matched training-cost comparison if RLVR needs extra groups','no promotion'],
        'quantization':False}

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def log(out,v):
    with (out/'events.jsonl').open('a') as f:
        f.write(json.dumps(v,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(v,sort_keys=True),flush=True)
def plain(v):
    if torch.is_tensor(v):return v.detach().cpu().tolist()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v

def prompt_ids(tok,model,text,reference=None):
    if reference is not None:
        text=text+'\nVerified reference answer: '+reference+'\nUse this reference to guide the answer.'
    rendered=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    suffix=''
    if model=='ring':
        if not rendered.endswith('<think>\n'):raise ValueError('Ring original-template opening changed')
        suffix='</think>\n';rendered+=suffix
    ids=tok.encode(rendered,add_special_tokens=False)
    if not ids or len(ids)>=160:raise ValueError('bounded prompt contract')
    return ids,{'original_template':True,'controller_suffix':suffix,'privileged_reference':reference is not None,'rendered_sha256':hashlib.sha256(rendered.encode()).hexdigest()}

def response_rows(prompt,generated,n):
    rows=list(range(len(prompt)-1,len(prompt)-1+len(generated)))
    if not generated or rows[-1]!=n-1:raise ValueError('response-token alignment')
    return rows


def teacher_input(tok,model,record,adapter,expected_sha):
    if adapter is None or native.sha(adapter)!=expected_sha:raise ValueError('frozen teacher changed')
    teacher_prompt,meta=prompt_ids(tok,model,tasks.prompt(record['task_object']))
    if teacher_prompt!=record['prompt_ids'] or meta['privileged_reference']:raise ValueError('teacher/student prompt mismatch')
    generated=record['ids']
    if not generated or any(type(x)is not int or not 0<=x<157184 for x in generated):raise ValueError('invalid sampled prefix')
    full=teacher_prompt+generated[:-1]
    rows=response_rows(teacher_prompt,generated,len(full))
    return full,rows,teacher_prompt,meta

def release(t=None):
    if t is not None:t.cpu()
    gc.collect();torch.cuda.empty_cache()

def compact(directory,positions):
    """Retain all scored rows after native validation; keep every original hash."""
    d=Path(directory);r=json.loads((d/'RUN.json').read_text());n=r['execution']['tokens'];h=native.MODELS[r['model']]['hidden'];kept={}
    for name,width in [('logits.f32',157184),('terminal.f32',11*h+8208)]:
        p=d/name
        if p.exists():
            a=np.memmap(p,dtype='<f4',mode='r').reshape(n,width)
            b=np.asarray(a[positions],dtype='<f4');q=d/('selected-'+name)
            with q.open('xb') as f:f.write(b.tobytes())
            kept[q.name]={'sha256':native.sha(q),'positions':positions,'bytes':q.stat().st_size};del a
    removed={}
    known={'logits.f32','terminal.f32','final-hidden.f32'}|{f'layer-{i}.f32' for i in range(native.MODELS[r['model']]['layers'])}
    if any(name.endswith('.f32') and name not in known for name in r['native_payloads']):raise ValueError('unknown trace retention requirement')
    for name,desc in r['native_payloads'].items():
        if name.endswith('.f32'):
            p=d/name
            if native.sha(p)!=desc['sha256']:raise ValueError('trace changed before retention')
            removed[name]=desc;p.unlink()
    native.dump(d/'RETENTION.json',{'retained':kept,'removed_validated_transient_floats':removed})

def update(t,opt,records,arm):
    """Exactly one update; all records were produced by the current adapter."""
    opt.zero_grad(set_to_none=True);start=time.monotonic()
    features=torch.cat([r['features'] for r in records]).cuda()
    logits=t(features)
    native_logits=np.concatenate([r['logits'] for r in records])
    parity=qualify.compare_logits(logits.detach().cpu().numpy(),native_logits)
    if not parity['passed']:raise ValueError('current-policy trainer/native logit mismatch')
    ids=torch.tensor([x for r in records for x in r['ids']],device='cuda',dtype=torch.long)
    observed=(logits.detach().double()/CONFIG['temperature']).log_softmax(-1).gather(1,ids[:,None]).flatten().cpu().numpy()
    recorded=np.asarray([x for r in records for x in r['sampling_logprobs']])
    error=float(np.abs(observed-recorded).max())
    if error>0.002:raise ValueError('behavior-policy log probability mismatch')
    if arm=='self_opd':
        teacher=torch.from_numpy(np.concatenate([r['teacher'] for r in records])).to(logits)
        loss,metrics=losses.self_opd(logits,teacher,CONFIG['temperature']);loss=loss/len(logits)
    elif arm=='rlvr':
        loss,metrics=losses.rlvr(logits,ids,[len(r['ids']) for r in records],[r['grade']['reward'] for r in records],[r['group'] for r in records],CONFIG['temperature'])
        if not bool((metrics['advantages']!=0).any()):raise ValueError('RLVR has no informative verifier-reward group')
    else:raise ValueError('unknown objective')
    if not torch.isfinite(loss):raise ValueError('nonfinite objective')
    loss.backward();grad=qualify.tensors_digest((n,p.grad) for n,p in t.named_parameters() if p.grad is not None)
    if len(grad)!=6 or any(p.grad is None or not torch.isfinite(p.grad).all() for p in t.parameters()):raise ValueError('gradient contract')
    norm=float(sum(p.grad.double().square().sum() for p in t.parameters()).sqrt())
    if norm==0:raise ValueError('zero update is not learning qualification')
    before=qualify.tensors_digest(t.named_parameters());opt.step();after=qualify.tensors_digest(t.named_parameters())
    if before==after:raise ValueError('adapter did not change')
    with torch.no_grad():
        post=t(features)
        if arm=='self_opd':post_loss,_=losses.self_opd(post,teacher,CONFIG['temperature']);post_loss=post_loss/len(post)
        else:post_loss,_=losses.rlvr(post,ids,[len(r['ids']) for r in records],[r['grade']['reward'] for r in records],[r['group'] for r in records],CONFIG['temperature'])
        if not torch.isfinite(post_loss):raise ValueError('nonfinite post-update diagnostic')
    torch.cuda.synchronize()
    return {'post_update_fixed_batch_loss':float(post_loss),'objective':arm,'loss':float(loss.detach()),'gradient_norm':norm,'gradient':grad,'parameters':after,'metrics':plain(metrics),'current_policy_logit_parity':parity,'sampled_logprob_max_abs':error,'response_tokens':len(logits),'update_seconds':time.monotonic()-start}

def run(model,suite_path,out,*,parent_checkpoint_digest):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);suite=tasks.load_suite(suite_path)
    train=[t for t in suite if t.split=='training'];held=[t for t in suite if t.split=='held_out'];ret=[t for t in suite if t.split=='retention']
    if (len(train),len(held),len(ret))!=(4,8,4):raise ValueError('suite split contract')
    c=native.MODELS[model];cp=Path(c['checkpoint']);manifest=native.sha(c['manifest'])
    exe=EXP/f'build/{model}-torch/strata_{model}_bf16';tok=AutoTokenizer.from_pretrained(cp,local_files_only=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    if model=='ling':
        from generate_ling import _eos_ids
        eos,eos_source=_eos_ids(cp)
    else:
        from generate_ring import _eos_ids
        eos=_eos_ids(cp);eos_source='generation_config.json'
    pins={'suite':native.sha(suite_path),'runner':native.sha(__file__),'executable':native.sha(exe),'manifest':manifest,'source_closure':native._source_closure(native.ROOT),'parent_checkpoint_digest':parent_checkpoint_digest,'native_profile':c['profile']}
    plan=dict(CONFIG,model=model,pins=pins,eos=eos,eos_source=eos_source,train_ids=[t.id for t in train],heldout_ids=[t.id for t in held],retention_ids=[t.id for t in ret])
    native.dump(out/'PLAN.json',plan);start=time.monotonic();t=None;cost={};counter=0
    def native_stage(name,tokens,**kw):
        nonlocal counter
        if time.monotonic()-start>CONFIG['max_job_seconds']:raise TimeoutError('qualification job time cap')
        if native._source_closure(native.ROOT)!=pins['source_closure']:raise ValueError('source changed during qualification')
        if native.sha(exe)!=pins['executable'] or c['profile']!=pins['native_profile']:raise ValueError('native executable/profile changed')
        release(t);d=out/name
        r=native.run(model,tokens,d,executable=exe,checkpoint_binding=manifest,timeout=CONFIG['native_timeout_seconds'],**kw)
        counter+=1;cost[name.split('/')[0]]=cost.get(name.split('/')[0],0)+r['wall_seconds']
        return d,r
    def sample(task,arm,roundno,j,adapter,batch=0,capture=True,eval_index=None):
        p,pmeta=prompt_ids(tok,model,tasks.prompt(task))
        seed=(CONFIG['rollout_seed']+roundno*10000+batch*1000+train.index(task)*10+j if eval_index is None else CONFIG['eval_seed']+eval_index)
        name=(f'{arm}/round-{roundno}/batch-{batch}/{task.id}-{j}' if eval_index is None else f'evaluation/{task.id}/{arm}')
        d,r=native_stage(name,p,adapter=adapter,capture=capture,generate=CONFIG['max_new_tokens'],eos=eos,sampling_temperature=CONFIG['temperature'],sampling_seed=seed)
        ids=r['generation']['generated_token_ids'];complete=r['generation']['finish_reason']=='eos'
        text=tok.decode(ids[:-1] if complete else ids,skip_special_tokens=False,clean_up_tokenization_spaces=False).strip()
        grade=tasks.grade(task,text,complete);rows=response_rows(p,ids,r['execution']['tokens'])
        features=terminal.load_features(d).rows[rows] if capture else None
        logits=np.array(np.memmap(d/'logits.f32',dtype='<f4',mode='r').reshape(-1,157184)[rows],copy=True)
        entry={'stage':'rollout','arm':arm,'round':roundno,'batch':batch,'task':task.id,'split':task.split,'seed':seed,'prompt_ids':p,'prompt_contract':pmeta,'ids':ids,'rows':rows,'complete':complete,'text':text,'grade':grade,'policy_adapter_sha256':native.sha(adapter) if adapter else None,'wall_seconds':r['wall_seconds'],'sampling_logprobs':r['generation']['sampling']['sampled_logprobs'],'receipt':str(d/'RUN.json')}
        native.dump(d/'ROLLOUT.json',entry);log(out,entry);compact(d,rows)
        return dict(entry,features=features,logits=logits,group=train.index(task) if eval_index is None else eval_index,path=d,task_object=task)
    def collect(arm,roundno,adapter,batch=0):
        return [sample(task,arm,roundno,j,adapter,batch) for task in train for j in range(CONFIG['samples_per_prompt'])]
    prior=native.EXPERIMENT/({'ling':'ling-q1','ring':'ring-q2'}[model])
    control=json.loads((prior/'INPUT.json').read_text())
    default_dir,_=native_stage('default-control',control['tokens'],capture=True)
    default_exact=native.exact_native(prior/'native-original',default_dir)
    native.dump(default_dir/'DEFAULT-PARITY.json',default_exact)
    initial_records=collect('initial',0,None)
    expert,exposure=qualify.nomination(torch.cat([r['features'] for r in initial_records]),c['hidden']);native.dump(out/'NOMINATION.json',exposure)
    torch.manual_seed(CONFIG['init_seed']);torch.cuda.manual_seed_all(CONFIG['init_seed'])
    t=terminal.build_terminal(cp,model,expert,rank=CONFIG['rank'],alpha=CONFIG['alpha'],device='cpu')
    initial={k:v.detach().clone() for k,v in t.named_parameters()};frozen=qualify.tensors_digest(t.named_buffers())
    zero=out/'zero.adapter';terminal.save_adapter(t,zero,cp,checkpoint_sha256=manifest)
    # Reproduce a complete initial trajectory with a zero adapter and identical RNG.
    z=sample(train[0],'zero-audit',0,0,zero)
    if z['ids']!=initial_records[0]['ids'] or not np.array_equal(z['logits'],initial_records[0]['logits']):raise ValueError('zero-adapter sampling parity')
    adapters={'baseline':None};results={};datahash=digest({'suite':pins['suite'],'training_ids':[x.id for x in train]})
    try:
        for arm in ARMS:
            with torch.no_grad():
                for name,p in t.named_parameters():p.copy_(initial[name])
            t.cuda();opt=torch.optim.Adam(t.parameters(),lr=CONFIG['lr'],foreach=False)
            conf=dict(CONFIG,objective=arm,model=model,expert=expert,manifest_sha256=manifest,suite_sha256=pins['suite'],source_sha256=digest(pins['source_closure']),executable_sha256=pins['executable'],native_profile=c['profile'],parent_checkpoint_digest=parent_checkpoint_digest,initial_adapter_sha256=native.sha(zero))
            steps=[];current=zero;records=None
            teacher_adapter=adapters.get('rlvr') if arm=='self_opd' else None
            if arm=='self_opd':
                if teacher_adapter is None:raise ValueError('Self-OPD requires qualified RLVR endpoint')
                conf['teacher_adapter_sha256']=native.sha(teacher_adapter)
                if conf['teacher_adapter_sha256']!=results['rlvr']['adapter_sha256']:raise ValueError('teacher endpoint drift')
                native.dump(out/'FROZEN-TEACHER.json',{'method':'Self-OPD(RLVR)','adapter':str(teacher_adapter),'adapter_sha256':native.sha(teacher_adapter),'metadata_sha256':native.sha(str(teacher_adapter)+'.json'),'same_base_manifest':manifest,'same_original_prompt':True,'reference_augmentation':False,'before_student_updates':True})
            for roundno in range(CONFIG['rounds']):
                records=[dict(r) for r in initial_records] if roundno==0 else collect(arm,roundno,current)
                # Round zero is on both policies because zero-adapter equality was proved.
                def informative(rs):
                    return any(len({r['grade']['reward'] for r in rs if r['group']==g})>1 for g in {r['group'] for r in rs})
                if arm=='rlvr':
                    for b in range(1,CONFIG['max_rlvr_batches_per_round']):
                        if informative(records):break
                        records+=collect(arm,roundno,current,batch=b)
                    if not informative(records):raise RuntimeError('RLVR non-informative after declared fresh rollout budget; no update or qualification claimed')
                if arm=='self_opd':
                    for i,r in enumerate(records):
                        full,rows,teacher_prompt,tm=teacher_input(tok,model,r,teacher_adapter,conf['teacher_adapter_sha256'])
                        d,rec=native_stage(f'teacher/round-{roundno}/{i}',full,capture=False,adapter=teacher_adapter)
                        r['teacher']=np.array(np.memmap(d/'logits.f32',dtype='<f4',mode='r').reshape(-1,157184)[rows],copy=True)
                        native.dump(d/'ALIGNMENT.json',{'student_rollout':r['receipt'],'student_ids':r['ids'],'teacher_rows':rows,'teacher_prompt_ids':teacher_prompt,'teacher_prompt_contract':tm,'teacher_adapter':str(teacher_adapter),'teacher_adapter_sha256':native.sha(teacher_adapter),'frozen_teacher':True,'teacher_method':'RLVR','reference_augmentation':False})
                        compact(d,rows)
                batch_binding=digest([{'receipt':native.sha(r['path']/'ROLLOUT.json'),'policy':r['policy_adapter_sha256'],'seed':r['seed'],'ids':r['ids'],'retained_payloads':native.sha(r['path']/'RETENTION.json'),'teacher_logits_sha256':hashlib.sha256(r['teacher'].tobytes()).hexdigest() if 'teacher' in r else None} for r in records])
                native.dump(out/f'{arm}-round-{roundno}-BATCH.json',{'current_adapter':native.sha(current),'fresh_prefixes_after_previous_update':roundno>0,'zero_policy_equivalence_audit':roundno==0,'batch_sha256':batch_binding,'records':[str(r['path']/'ROLLOUT.json') for r in records]})
                if roundno>0 and any(r['policy_adapter_sha256']!=native.sha(current) for r in records):raise ValueError('stale/off-policy rollout')
                t.cuda();result=update(t,opt,records,arm);steps.append(result);log(out,dict(stage='update',arm=arm,round=roundno,loss=result['loss'],gradient_norm=result['gradient_norm'],response_tokens=result['response_tokens'],seconds=result['update_seconds']))
                current=out/f'{arm}-step-{roundno+1}.adapter';terminal.save_adapter(t,current,cp,checkpoint_sha256=manifest)
                if roundno==0:
                    snapshot_datahash=digest({'dataset':datahash,'round0_batch':batch_binding})
                    conf['snapshot_adapter_sha256']=native.sha(current)
                    terminal.save_training_state(t,opt,out/f'{arm}-snapshot-step1',step=1,iteration_cursor=1,data_hash=snapshot_datahash,config_binding=conf)
            final_params={k:v.detach().cpu().clone() for k,v in t.named_parameters()};endopt=copy.deepcopy(opt.state_dict());endrng=torch.get_rng_state().clone();endcuda=[x.clone() for x in torch.cuda.get_rng_state_all()]
            if native.sha(out/f'{arm}-step-1.adapter')!=conf['snapshot_adapter_sha256']:raise ValueError('snapshot adapter changed')
            terminal.restore_training_state(t,opt,out/f'{arm}-snapshot-step1',data_hash=snapshot_datahash,config_binding=conf)
            audit=update(t,opt,records,arm)
            qualify.tree_exact(dict(t.named_parameters()),final_params);qualify.tree_exact(opt.state_dict(),endopt);qualify.tree_exact(torch.get_rng_state(),endrng);qualify.tree_exact(torch.cuda.get_rng_state_all(),endcuda)
            for k in ('loss','gradient','parameters','metrics'):qualify.tree_exact(audit[k],steps[-1][k])
            resumed=out/f'{arm}-resumed.adapter';terminal.save_adapter(t,resumed,cp,checkpoint_sha256=manifest)
            if native.sha(current)!=native.sha(resumed):raise ValueError('restored adapter export differs')
            with torch.no_grad():
                for p in t.parameters():p.zero_()
            terminal.load_adapter(t,current,checkpoint_sha256=manifest);qualify.tree_exact(dict(t.named_parameters()),final_params)
            # Native adapted forward on a training trajectory with maximal expert exposure.
            r=max(records,key=lambda r:int((r['features'][:,3*c['hidden']:3*c['hidden']+8]==expert).sum()))
            with torch.no_grad():expected=t(r['features'].cuda()).cpu().numpy()
            d,rec=native_stage(f'{arm}-export-reload',r['prompt_ids']+r['ids'][:-1],adapter=current,capture=True)
            rows=r['rows'];observed=np.asarray(np.memmap(d/'logits.f32',dtype='<f4',mode='r').reshape(-1,157184)[rows]);parity=qualify.compare_logits(expected,observed)
            native.dump(d/'FORWARD-PARITY.json',parity);compact(d,rows)
            if not parity['passed']:raise ValueError('trained adapter native parity gate')
            if frozen!=qualify.tensors_digest(t.named_buffers()):raise ValueError('frozen trainer weights changed')
            results[arm]={'steps':steps,'export_reload_exact':True,'native_forward':parity,'training_state_restore_exact':True,'optimizer_rng_exact':True,'adapter_sha256':native.sha(current),'restore_audit_not_an_on_policy_update':True}
            adapters[arm]=current;del opt,endopt,final_params,records;release(t)
            native.dump(out/f'{arm}-QUALIFIED.json',results[arm])
        frozen_adapters={a:native.sha(p) for a,p in adapters.items() if p};native.dump(out/'FROZEN-CANDIDATES.json',{'adapters':frozen_adapters,'before_heldout':True,'source_pins':pins})
        release(t);t=None;evals=[]
        for i,task in enumerate(held+ret):
            order=list(adapters);order=order[i%3:]+order[:i%3]
            for arm in order:
                r=sample(task,arm,-1,0,adapters[arm],capture=False,eval_index=i)
                evals.append({k:r[k] for k in ('arm','task','split','grade','complete','wall_seconds','ids')})
        sums={}
        for arm in adapters:
            sums[arm]={}
            for split in ('held_out','retention'):
                rs=[r for r in evals if r['arm']==arm and r['split']==split];secs=sum(r['wall_seconds'] for r in rs);n=sum(r['grade']['reward'] for r in rs)
                sums[arm][split]={'n':len(rs),'correct':n,'incomplete':sum(not r['complete'] for r in rs),'wall_seconds':secs,'verified_per_inference_hour':3600*n/secs}
        # Same original prompt, RNG and no adapter: rollback is an observed native fact.
        rollback=sample(train[0],'rollback',0,0,None)
        if rollback['ids']!=initial_records[0]['ids'] or not np.array_equal(rollback['logits'],initial_records[0]['logits']):raise ValueError('base rollback mismatch')
        rollback_dir,_=native_stage('default-rollback',control['tokens'],capture=True)
        rollback_exact=native.exact_native(default_dir,rollback_dir)
        if any(native.sha(adapters[a])!=h for a,h in frozen_adapters.items()):raise ValueError('frozen candidate drift')
        if native.sha(suite_path)!=pins['suite'] or native.sha(exe)!=pins['executable'] or native._source_closure(native.ROOT)!=pins['source_closure']:raise ValueError('input drift')
        result={'mechanics_qualified':True,'model':model,'expert':expert,'trainable_parameters':sum(x.numel() for x in initial.values()),'training':results,'evaluation':sums,'cost_seconds':cost,'native_calls':counter,'base_rollback_exact':rollback_exact,'default_unchanged_vs_prior_qualified_build':default_exact,'no_quantization':True,'promoted':False,'scope':CONFIG['trainable_scope'],'quality_claim':'paired small qualification set only; independent larger validation required for promotion'}
        native.dump(out/'RESULT.json',result);return result
    finally:release(t)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=native.MODELS);p.add_argument('--suite',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with open('/work/AI/experiments/strata-dev8-gpu.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        before=qualify.bind_checkpoint(a.model);foreign=qualify.gpu_processes();native.dump(a.output.with_suffix('.binding-before.json'),before)
        error=None;start=time.monotonic()
        try:run(a.model,a.suite,a.output,parent_checkpoint_digest=digest(before))
        except BaseException as e:error=repr(e);raise
        finally:
            after=qualify.bind_checkpoint(a.model);native.dump(a.output.with_suffix('.binding-after.json'),after)
            try:qualify.check_foreign(foreign);foreign_ok=True
            except RuntimeError:foreign_ok=False
            native.dump(a.output.with_suffix('.RETURN.json'),{'error':error,'checkpoint_exact':before==after,'foreign_gpu_preserved':foreign_ok,'before_gpu':foreign,'after_gpu':qualify.gpu_processes(),'wall_seconds':time.monotonic()-start})
            if error is None and before!=after:raise RuntimeError('checkpoint changed')
            if error is None and not foreign_ok:raise RuntimeError('foreign GPU process changed')
if __name__=='__main__':main()
