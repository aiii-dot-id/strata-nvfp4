"""Completion-calibrated native RLVR / Self-OPD capability campaign.
AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations
import argparse, copy, dataclasses, fcntl, gc, hashlib, json, math, os, signal, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import native, terminal, qualify, on_policy_runner as old, capability_tasks as tasks
from completion import CompletionConfig, run_completion
from capability_update import update, advantages
from transformers import AutoTokenizer
PREVIOUS=Path('/work/AI/experiments/strata-on-policy-20261021-r1')
CONFIG={'schema':'strata-capability-v2','author':terminal.AUTHOR,'rlvr_rounds':6,'opd_rounds':4,
        'samples_per_prompt':4,'lr':0.01,'opd_lr':0.003,'rank':8,'alpha':16,'temperature':1.,
        'chunk_rows':8,'clip_norm':1.,'calibration_tasks_per_family':4,'max_training_groups_per_family':2,
        'ling_answer_tokens':96,'ring_thinking_tokens':128,'ring_answer_tokens':96,
        'max_job_seconds':43200,'native_timeout':420,'min_completion_rate':0.80,
        'teacher_min_selection_gain':2,'no_quantization':True,'promoted':False,
        'training_scope':'one selected final-layer routed expert',
        'selection_claim':'exploratory fixed development gate; independent held-out follows',
        'objectives':'RLOO verifier-reward policy gradient then full-vocabulary reverse-KL fixed-prefix Self-OPD',
        'stopping':'fixed rollout counts; no additional sampling conditional on mixed reward',
        'development_split':'unused; completion and reward calibration uses training tasks only',
        'comparison':'paired original/RLVR/Self-OPD with identical inference budgets',
        'seed':721021}
def log(out,event):
    with (out/'events.jsonl').open('a') as f:
        f.write(json.dumps(event,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(event,sort_keys=True,allow_nan=False),flush=True)
def digest(x):return old.digest(x)
def release(t=None):
    if t is not None:t.cpu()
    gc.collect();torch.cuda.empty_cache()
def strip_final(text,model,format_error):
    if format_error:return '',format_error
    if model=='ring':
        if text.count('</think>')!=1:return '','missing_or_duplicate_thinking_close'
        text=text.split('</think>',1)[1]
    return text.strip(),None
def coverage(records):
    compact=[{'split':'training','family':r['family'],'id':r['task'],'reward':r['grade']['reward']} for r in records]
    by={}
    for family in tasks.FAMILIES:
        rows=[r for r in records if r['family']==family]
        groups={}
        for r in rows:groups.setdefault(r['task'],[]).append(r)
        mixed=[k for k,v in groups.items() if {x['grade']['reward'] for x in v}=={0,1}]
        by[family]={'n':len(rows),'complete':sum(r['complete'] for r in rows),'correct':sum(r['grade']['reward'] for r in rows),
                    'completion_rate':sum(r['complete'] for r in rows)/len(rows) if rows else 0.,'mixed_groups':mixed,
                    'all_groups':[{'task':k,'rewards':[r['grade']['reward'] for r in v]} for k,v in groups.items()]}
    passed=all(v['mixed_groups'] and v['completion_rate']>=CONFIG['min_completion_rate'] for v in by.values())
    return {'passed':passed,'families':by,'requirement':'at least one mixed sampled prompt group per target family; failures retained'}
def nomination(records,h):
    adv=advantages(records);counts={f:torch.zeros(256,dtype=torch.float64) for f in tasks.FAMILIES}
    for r,a in zip(records,adv):
        if a==0:continue
        ids=r['features'][:,3*h:3*h+8].long().flatten()
        counts[r['family']]+=torch.bincount(ids,minlength=256).double()*abs(a)
    normalized=[v/max(float(v.sum()),1.) for v in counts.values()]
    score=torch.stack(normalized).min(0).values
    expert=int(score.argmax())
    if not float(score[expert])>0:raise ValueError('no expert exposed to informative groups in both families')
    return expert,{'expert':expert,'rule':'maximum minimum family share of advantage-weighted sampled-token routes','counts':{k:v.tolist() for k,v in counts.items()},'score':float(score[expert])}
def summarize(rows):
    out={}
    for arm in sorted({r['arm'] for r in rows}):
        out[arm]={}
        for split in sorted({r['split'] for r in rows}):
            rs=[r for r in rows if r['arm']==arm and r['split']==split]
            secs=sum(r['wall_seconds'] for r in rs);correct=sum(r['grade']['reward'] for r in rs)
            out[arm][split]={'n':len(rs),'correct':correct,'incomplete':sum(not r['complete'] for r in rs),
                            'wall_seconds':secs,'verified_per_instrumented_hour':correct*3600/secs if secs else None,
                            'families':{f:sum(r['grade']['reward'] for r in rs if r['family']==f) for f in tasks.FAMILIES}}
    return out
def select_teacher(rows):
    def arm(name):
        return [dict(task=r['task'],split=r['split'],family=r['family'],reward=r['grade']['reward']) for r in rows if r['arm']==name]
    return tasks.teacher_selection_bound(arm('baseline'),arm('rlvr'))

def run(model,suite,private,out,checkpoint_digest):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    all_tasks=tasks.load_suite(suite,private)
    if len([t for t in all_tasks if t.split=='teacher_selection'])!=16:raise ValueError('selection count')
    cfg=native.MODELS[model];cp=Path(cfg['checkpoint']);manifest=native.sha(cfg['manifest'])
    exe=PREVIOUS/f'build/{model}-torch/strata_{model}_bf16'
    tok=AutoTokenizer.from_pretrained(cp,local_files_only=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    if model=='ling':
        from generate_ling import _eos_ids
        eos,_=_eos_ids(cp)
    else:
        from generate_ring import _eos_ids
        eos=_eos_ids(cp)
    pins={'suite':native.sha(suite),'private':native.sha(private),'executable':native.sha(exe),
          'manifest':manifest,'source':native._source_closure(native.ROOT),'parent_checkpoint_digest':checkpoint_digest,
          'native_profile':cfg['profile']}
    controller=CompletionConfig(model=model,max_new_tokens=CONFIG['ling_answer_tokens'],
                 thinking_tokens=CONFIG['ring_thinking_tokens'],answer_tokens=CONFIG['ring_answer_tokens'],
                 context_capacity=cfg['capacity'],eos_token_ids=tuple(eos))
    native.dump(out/'PLAN.json',dict(CONFIG,model=model,pins=pins,completion=dataclasses.asdict(controller),
                 heldout_sealed_until_freeze=True,old_qualification_source='4ac6b3a14964b19529e77802ec14f7745004472f',
                 suite_schema='fresh task prompts and evaluator-only private oracles'))
    start=time.monotonic();trainer=None;native_count=0
    def stage(path,ids,**kw):
        nonlocal native_count
        if time.monotonic()-start>CONFIG['max_job_seconds']:raise TimeoutError('campaign total time cap')
        if native._source_closure(native.ROOT)!=pins['source']:raise ValueError('source drift')
        if native.sha(suite)!=pins['suite'] or native.sha(private)!=pins['private']:raise ValueError('suite drift')
        release(trainer)
        rec=native.run(model,ids,path,executable=exe,checkpoint_binding=manifest,timeout=CONFIG['native_timeout'],**kw)
        native_count+=1
        return rec
    def sample(task,arm,label,seed,adapter,capture=True,greedy=False):
        text=tasks.prompt(task)+'\n\nFinal answer: write only the requested literal on one line. No labels, markdown, or extra text.'
        rendered=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        if model=='ring' and not rendered.endswith('<think>\n'):raise ValueError('Ring opening template changed')
        p=tok.encode(rendered,add_special_tokens=False);d=out/label;d.mkdir(parents=True,exist_ok=False)
        pieces=[];seq=0
        def callback(ids,*,max_new_tokens,adapter,stage: str):
            nonlocal seq
            sd=d/stage;seq+=1;kw={} if greedy else {'sampling_temperature':CONFIG['temperature'],'sampling_seed':seed+seq*1000000}
            rec=stage_call(sd,list(ids),adapter=adapter,capture=capture,generate=max_new_tokens,eos=eos,**kw)
            g=rec['generation'];generated=g['generated_token_ids'];done=g['finish_reason']=='eos'
            rows=list(range(len(ids)-1,len(ids)-1+len(generated)))
            raw=np.memmap(sd/'logits.f32',dtype='<f4',mode='r').reshape(-1,157184)
            logits=np.array(raw[rows],copy=True);del raw
            lps=g.get('sampling',{}).get('sampled_logprobs')
            if greedy:
                lps=[]
                for row,target in zip(logits,generated):
                    x=row.astype('float64');m=x.max();lps.append(float(x[target]-m-np.log(np.exp(x-m).sum())))
            f=terminal.load_features(sd).rows[rows] if capture else None
            if capture:old.compact(sd,rows)
            else:
                removed={}
                for name,desc in rec['native_payloads'].items():
                    if name.endswith('.f32'):
                        fp=sd/name
                        if native.sha(fp)!=desc['sha256']:raise ValueError('evaluation payload drift')
                        removed[name]=desc;fp.unlink()
                native.dump(sd/'RETENTION.json',{'removed_validated_transient_floats':removed,'generation_receipts_retained':True})
            pieces.append({'path':sd,'features':f,'logits':logits if capture else None,'rows':rows,'seconds':rec['wall_seconds']})
            visible=generated[:-1] if done else generated
            return {'ids':generated,'sampled_logprobs':lps,'text':tok.decode(visible,skip_special_tokens=False,clean_up_tokenization_spaces=False),
                    'eos':done,'finish_reason':g['finish_reason']}
        stage_call=stage
        completed=run_completion(p,adapter,callback,lambda s:tok.encode(s,add_special_tokens=False),
                    lambda ids:tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),controller)
        final,error=strip_final(completed.text,model,completed.format_error)
        grade=tasks.grade(task,final,completed.complete)
        if error:grade={'reward':0,'reason':error}
        actionrows=[x-1 for x in completed.action_positions]
        if [completed.transcript_ids[x] for x in completed.action_positions]!=list(completed.sampled_ids):raise ValueError('action positions mismatch')
        entry={'schema':'capability-rollout-v1','arm':arm,'task':task.id,'split':task.split,'family':task.family,'level':task.level,
            'seed':seed,'greedy':greedy,'prompt_ids':p,'prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest(),
            'transcript_ids':list(completed.transcript_ids),'ids':list(completed.sampled_ids),'action_positions':list(completed.action_positions),
            'teacher_rows':actionrows,'sampling_logprobs':list(completed.sampled_logprobs),'inserted_ids':list(completed.inserted_ids),
            'forced_close':completed.forced_close,'complete':completed.complete,'format_error':error,'text':final,'raw_text':completed.text,
            'grade':grade,'policy_adapter_sha256':native.sha(adapter) if adapter else None,'wall_seconds':sum(x['seconds'] for x in pieces),
            'native_stages':[{'path':str(x['path']),'run_sha256':native.sha(x['path']/'RUN.json'),
                              'retention_sha256':native.sha(x['path']/'RETENTION.json')} for x in pieces]}
        native.dump(d/'ROLLOUT.json',entry);log(out,dict(stage='rollout',**{k:entry[k] for k in ['arm','task','split','family','complete','grade','wall_seconds']},path=str(d/'ROLLOUT.json')))
        return dict(entry,path=d,group=task.id,features=torch.cat([x['features'] for x in pieces]) if capture else None,
                    logits=np.concatenate([x['logits'] for x in pieces]) if capture else None)
    def collect(selected,arm,roundno,adapter):
        return [sample(task,arm,f'{arm}/round-{roundno}/{task.id}/sample-{j}',
                    CONFIG['seed']+roundno*10000+i*10+j,adapter)
                for i,task in enumerate(selected) for j in range(CONFIG['samples_per_prompt'])]
    def evaluate(selected,adapters,label):
        rows=[];names=list(adapters)
        for i,task in enumerate(selected):
            order=names[i%len(names):]+names[:i%len(names)]
            for arm in order:
                r=sample(task,arm,f'{label}/{task.id}/{arm}',CONFIG['seed']+500000+i,adapters[arm],capture=False,greedy=True)
                rows.append({k:v for k,v in r.items() if k not in ('features','logits','path')})
        native.dump(out/(label+'-RESULT.json'),{'rows':rows,'summary':summarize(rows)});return rows
    try:
        train=[x for x in all_tasks if x.split=='training']
        calibration=[x for f in tasks.FAMILIES for x in [y for y in train if y.family==f][:CONFIG['calibration_tasks_per_family']]]
        initial=collect(calibration,'calibration',0,None);cov=coverage(initial);native.dump(out/'COVERAGE.json',cov)
        if not cov['passed']:
            result={'status':'coverage_gate_failed','model':model,'coverage':cov,'updates':0,'self_opd_run':False,'no_quantization':True,
                    'next':'repair development completion or task difficulty before training; no heldout examined','native_calls':native_count}
            native.dump(out/'RESULT.json',result);return result
        activeids=[tid for f in tasks.FAMILIES for tid in cov['families'][f]['mixed_groups'][:CONFIG['max_training_groups_per_family']]]
        active=[x for x in train if x.id in activeids]
        expert,nom=nomination(initial,cfg['hidden']);native.dump(out/'NOMINATION.json',dict(nom,active_training_ids=activeids))
        torch.manual_seed(CONFIG['seed']);torch.cuda.manual_seed_all(CONFIG['seed'])
        trainer=terminal.build_terminal(cp,model,expert,rank=CONFIG['rank'],alpha=CONFIG['alpha'],device='cpu')
        initial_params={k:v.detach().clone() for k,v in trainer.named_parameters()};frozen=qualify.tensors_digest(trainer.named_buffers())
        zero=out/'zero.adapter';terminal.save_adapter(trainer,zero,cp,checkpoint_sha256=manifest)
        z=sample(calibration[0],'zero-audit','zero-audit',CONFIG['seed'],zero)
        if z['ids']!=initial[0]['ids'] or not np.array_equal(z['logits'],initial[0]['logits']):raise ValueError('zero-adapter native sampling differs')
        native.dump(out/'ZERO-PARITY.json',{'exact':True,'baseline':str(initial[0]['path']),'zero':str(z['path'])})
        # Calibration decides a fixed prompt pool; no candidate-dependent rejection sampling.
        native.dump(out/'TRAINING-ADMISSION.json',{'scope':'one terminal expert','expert':expert,'active_ids':activeids,
                    'coverage':cov,'fixed_rounds':CONFIG['rlvr_rounds'],'lr':CONFIG['lr'],'heldout_used':False})
        results={};adapters={'baseline':None};frozen_teacher=None
        selection=[x for x in all_tasks if x.split=='teacher_selection']
        for arm in ('rlvr','self_opd'):
            if arm=='self_opd' and frozen_teacher is None:break
            with torch.no_grad():
                for n,p in trainer.named_parameters():p.copy_(initial_params[n])
            trainer.cuda();lr=CONFIG['lr'] if arm=='rlvr' else CONFIG['opd_lr']
            opt=torch.optim.Adam(trainer.parameters(),lr=lr,foreach=False)
            current=zero;steps=[]
            rounds=CONFIG['rlvr_rounds'] if arm=='rlvr' else CONFIG['opd_rounds']
            for roundno in range(rounds):
                records=collect(active,arm,roundno+1,current)
                rcov=coverage(records);native.dump(out/f'{arm}-round-{roundno+1}-COVERAGE.json',rcov)
                if arm=='rlvr' and not any(a!=0 for a in advantages(records)):
                    log(out,{'stage':'uniform_reward_round','arm':arm,'round':roundno+1,'skipped_update':True})
                    continue
                if arm=='self_opd':
                    for i,r in enumerate(records):
                        d=out/f'teacher/round-{roundno+1}/{i}';full=r['transcript_ids'][:-1]
                        if not full or max(r['teacher_rows'])>=len(full):raise ValueError('teacher action-prefix alignment')
                        rec=stage(d,full,adapter=frozen_teacher,capture=False)
                        a=np.memmap(d/'logits.f32',dtype='<f4',mode='r').reshape(-1,157184)
                        r['teacher']=np.array(a[r['teacher_rows']],copy=True);del a
                        native.dump(d/'ALIGNMENT.json',{'student':str(r['path']/'ROLLOUT.json'),'student_sha256':native.sha(r['path']/'ROLLOUT.json'),
                                    'full_prefix':full,'rows':r['teacher_rows'],'inserted_not_actions':True,
                                    'teacher_sha256':native.sha(frozen_teacher),'frozen':True,'reference_augmentation':False})
                        old.compact(d,r['teacher_rows'])
                batchhash=digest([{'rollout':native.sha(r['path']/'ROLLOUT.json'),'stages':r['native_stages']} for r in records])
                conf={'campaign_plan':native.sha(out/'PLAN.json'),'arm':arm,'expert':expert,'lr':lr,'parent_checkpoint':checkpoint_digest}
                trainer.cuda()
                # Every update is surrounded by a restorable state; exact replay audits final update.
                if steps:
                    beforestep=out/f'{arm}-snapshot-before-{roundno+1}'
                    terminal.save_training_state(trainer,opt,beforestep,step=len(steps),iteration_cursor=roundno,data_hash=batchhash,config_binding=conf)
                metric=update(trainer,opt,records,arm,temperature=CONFIG['temperature'],chunk_rows=CONFIG['chunk_rows'],clip_norm=CONFIG['clip_norm'])
                current=out/f'{arm}-step-{len(steps)+1}.adapter';terminal.save_adapter(trainer,current,cp,checkpoint_sha256=manifest)
                steps.append(metric);native.dump(out/f'{arm}-UPDATE-{len(steps)}.json',dict(metric,round=roundno+1,adapter_sha256=native.sha(current),batch_sha256=batchhash))
                log(out,{'stage':'update','arm':arm,'step':len(steps),'round':roundno+1,
                         'loss':metric['loss'],'gradient_norm':metric['gradient_norm'],'family_gradient_norm':metric['family_gradient_norm']})
                if len(steps)>1:
                    end_params={k:v.detach().clone() for k,v in trainer.named_parameters()};endopt=copy.deepcopy(opt.state_dict())
                    endrng=torch.get_rng_state().clone();endcuda=[x.clone() for x in torch.cuda.get_rng_state_all()]
                    terminal.restore_training_state(trainer,opt,beforestep,data_hash=batchhash,config_binding=conf)
                    repeated=update(trainer,opt,records,arm,temperature=CONFIG['temperature'],chunk_rows=CONFIG['chunk_rows'],clip_norm=CONFIG['clip_norm'])
                    qualify.tree_exact(dict(trainer.named_parameters()),end_params);qualify.tree_exact(opt.state_dict(),endopt)
                    qualify.tree_exact(torch.get_rng_state(),endrng);qualify.tree_exact(torch.cuda.get_rng_state_all(),endcuda)
                    for key in ('loss','gradient','parameters'):qualify.tree_exact(metric[key],repeated[key])
                    resumed=out/f'{arm}-resumed-step-{len(steps)}.adapter';terminal.save_adapter(trainer,resumed,cp,checkpoint_sha256=manifest)
                    if native.sha(current)!=native.sha(resumed):raise ValueError('resumed adapter byte mismatch')
                    native.dump(out/f'{arm}-RESTORE-{len(steps)}.json',{'parameters_exact':True,'optimizer_rng_exact':True,'resumed_export_exact':True})
                # Native replay checks the new adapter at actual action states.
                r=max(records,key=lambda r:int((r['features'][:,3*cfg['hidden']:3*cfg['hidden']+8]==expert).sum()))
                with torch.no_grad():expected=np.concatenate([trainer(r['features'][s:s+8].cuda()).cpu().numpy() for s in range(0,len(r['ids']),8)])
                vd=out/f'{arm}-native-check-{len(steps)}'
                stage(vd,r['transcript_ids'][:-1],adapter=current,capture=False)
                a=np.memmap(vd/'logits.f32',dtype='<f4',mode='r').reshape(-1,157184);actual=np.array(a[r['teacher_rows']],copy=True);del a
                parity=qualify.compare_logits(expected,actual);old.compact(vd,r['teacher_rows'])
                native.dump(vd/'PARITY.json',parity)
                if not parity['passed']:raise ValueError('updated adapter native parity')
                if frozen!=qualify.tensors_digest(trainer.named_buffers()):raise ValueError('frozen weights mutated')
                del records;release(trainer)
            if len(steps)<2:raise ValueError('fewer than two genuine updates; continuation unqualified')
            if arm=='rlvr' and any(not any(s['family_gradient_norm'].get(f,0)>0 for s in steps) for f in tasks.FAMILIES):
                raise ValueError('training lacks actual gradients for a target family')
            adapters[arm]=current;results[arm]={'steps':len(steps),'adapter_sha256':native.sha(current),'last_gradient_norm':steps[-1]['gradient_norm']}
            # Explicit export/reload after clobbering only the research parameters.
            pbefore=qualify.tensors_digest(trainer.named_parameters())
            with torch.no_grad():
                for p in trainer.parameters():p.zero_()
            terminal.load_adapter(trainer,current,checkpoint_sha256=manifest)
            if qualify.tensors_digest(trainer.named_parameters())!=pbefore:raise ValueError('export reload differs')
            results[arm]['export_reload_exact']=True
            native.dump(out/f'{arm}-QUALIFIED.json',results[arm]);release(trainer)
            if arm=='rlvr':
                frozen_sha=native.sha(current);rows=evaluate(selection,adapters,'teacher-selection')
                base={r['task']:r['grade']['reward'] for r in rows if r['arm']=='baseline'}
                cand={r['task']:r['grade']['reward'] for r in rows if r['arm']=='rlvr'}
                gate=select_teacher(rows)
                gate['candidate_sha256']=frozen_sha;gate['selection_ids']=[t.id for t in selection]
                native.dump(out/'TEACHER-SELECTION.json',gate)
                if native.sha(current)!=frozen_sha:raise ValueError('teacher changed during selection')
                if gate['select']:
                    frozen_teacher=current
                    native.dump(out/'FROZEN-TEACHER.json',{'adapter':str(current),'sha256':frozen_sha,'scope':'selection-set improvement only until independent heldout'})
                else:
                    log(out,{'stage':'teacher_gate_failed','gate':gate,'self_opd_run':False});break
        finalpins={a:native.sha(p) for a,p in adapters.items() if p}
        native.dump(out/'FROZEN-CANDIDATES.json',{'adapters':finalpins,'before_heldout':True,'source':pins['source']})
        evaluated=evaluate([t for t in all_tasks if t.split in ('held_out','retention')],adapters,'fresh-heldout')
        if any(native.sha(adapters[a])!=h for a,h in finalpins.items()):raise ValueError('candidate drift')
        rollback=sample(calibration[0],'rollback','rollback',CONFIG['seed'],None)
        if rollback['ids']!=initial[0]['ids'] or not np.array_equal(rollback['logits'],initial[0]['logits']):raise ValueError('baseline rollback differs')
        result={'status':'completed' if 'self_opd' in adapters else 'teacher_not_better_self_opd_not_admitted',
                'model':model,'training':results,'evaluation':summarize(evaluated),'base_rollback_exact':True,
                'teacher_selected':frozen_teacher is not None,'self_opd_run':'self_opd' in adapters,'native_calls':native_count,
                'no_quantization':True,'promoted':False,'wall_seconds':time.monotonic()-start,
                'scope':CONFIG['training_scope'],'quality_claim':'fresh held-out pilot only; no automatic capability or production promotion'}
        native.dump(out/'RESULT.json',result);return result
    finally:release(trainer)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=native.MODELS)
    ap.add_argument('--suite',type=Path,required=True);ap.add_argument('--private',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    with open('/work/AI/experiments/strata-dev8-gpu.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        before=qualify.bind_checkpoint(a.model);foreign=qualify.gpu_processes()
        native.dump(a.output.with_suffix('.binding-before.json'),before);error=None;outcome=None;started=time.monotonic()
        try:outcome=run(a.model,a.suite,a.private,a.output,digest(before))
        except BaseException as ex:error=repr(ex);raise
        finally:
            release();after=qualify.bind_checkpoint(a.model);native.dump(a.output.with_suffix('.binding-after.json'),after)
            try:qualify.check_foreign(foreign);preserved=True
            except RuntimeError:preserved=False
            native.dump(a.output.with_suffix('.RETURN.json'),{'error':error,'checkpoint_exact':before==after,
                'foreign_gpu_preserved':preserved,'before_gpu':foreign,'after_gpu':qualify.gpu_processes(),
                'wall_seconds':time.monotonic()-started,'services_touched':False,'mac_touched':False,'outcome':outcome['status'] if outcome else None})
            if error is None and (before!=after or not preserved):raise RuntimeError('restoration identity failed')
    return 0 if outcome and outcome.get('status')=='completed' else 2
if __name__=='__main__':sys.exit(main())
