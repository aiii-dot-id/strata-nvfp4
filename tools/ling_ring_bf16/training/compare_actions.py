"""Paired corrected-action pilot using real Strata captures and adapters.
AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations
import argparse,copy,fcntl,gc,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
SOURCE=Path('/work/AI/src/strata-ling-ring-training-20261021/tools/ling_ring_bf16/training')
sys.path.insert(0,str(SOURCE))
import native,terminal,qualify
from transformers import AutoTokenizer


def arm_order(i):
    v=['U','S','C'];return v[i%3:]+v[:i%3]


def weights_for(records,failures,arm):
    weights=torch.cat([torch.full((len(r['target_ids']),),2. if failures[r['case_id']] else 1.) for r in records])
    if arm=='S':weights.fill_(1)
    elif arm!='C':raise ValueError('unknown arm')
    return weights*(len(weights)/weights.sum())


def rows_and_targets(record):
    p=record['prompt_ids'];t=record['target_ids'];rows=record['target_row_indices']
    if rows!=list(range(len(p)-1,len(p)-1+len(t))) or not record['target_boundary_verified']:raise ValueError('invalid target mask')
    return p+t[:-1],rows,t


def compact(directory,positions):
    """Retain scored rows; full transient traces were validated and hashed by native.run."""
    directory=Path(directory);rec=json.loads((directory/'RUN.json').read_text());n=rec['execution']['tokens'];c=native.MODELS[rec['model']]
    kept={}
    for name,width in [('logits.f32',157184),('terminal.f32',11*c['hidden']+8208)]:
        path=directory/name
        if path.exists():
            a=np.memmap(path,dtype='<f4',mode='r').reshape(n,width)
            selected=np.asarray(a[positions],dtype='<f4');out=directory/('selected-'+name)
            with out.open('xb') as f:f.write(selected.tobytes())
            kept[out.name]={'positions':positions,'sha256':native.sha(out),'bytes':out.stat().st_size};del a
    removed={}
    for name,desc in rec['native_payloads'].items():
        if name.endswith('.f32'):
            path=directory/name
            if native.sha(path)!=desc['sha256']:raise ValueError('transient trace changed before retention')
            removed[name]=desc;path.unlink()
    native.dump(directory/'RETENTION.json',{'policy':'Validated full transient float traces are replaced with selected scored rows; native RUN contains original hashes. Original checkpoints and input artifacts are untouched.','retained':kept,'removed_transient_payloads':removed})


def jsonl(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(value,sort_keys=True),flush=True)


def decode_result(tok,receipt,eos):
    ids=receipt['generation']['generated_token_ids'];complete=receipt['generation']['finish_reason']=='eos'
    if complete:
        if ids[-1] not in eos:raise ValueError('EOS decode mismatch')
        ids=ids[:-1]
    return tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False).strip(),complete


def run(model,dataset,out):
    out=Path(out);out.mkdir(exist_ok=False,parents=True);dataset=Path(dataset)
    qroot=native.EXPERIMENT
    for m in ['ling','ring']:
        q=json.loads((qroot/({'ling':'ling-q1','ring':'ring-q2'}[m])/'RESULT.json').read_text());ret=json.loads((qroot/({'ling':'ling-q1','ring':'ring-q2'}[m]+'.RETURN.json')).read_text())
        if q.get('passed')is not True or ret.get('error') is not None or ret.get('checkpoint_exact') is not True:raise ValueError('both model qualifications must pass')
    spec=importlib.util.spec_from_file_location('pilot_grader',dataset/'grader.py');grader=importlib.util.module_from_spec(spec);spec.loader.exec_module(grader)
    allcases=json.loads((dataset/'cases.json').read_text())['cases'];cases={c['id']:c for c in allcases}
    prepared=json.loads((dataset/'evidence/prepared_actions.json').read_text())
    records=[r['record'] for r in prepared['records'] if r['model']==model]
    train=[r for r in records if r['split']=='train'];held=[r for r in records if r['split']=='heldout']
    retention=[r for r in json.loads((dataset/'evidence/retention_probes.json').read_text())['records'] if r['model']==model]
    if len(train)!=16 or len(held)!=16 or len(retention)!=4:raise ValueError('split count')
    c=native.MODELS[model];cp=Path(c['checkpoint']);manifest=native.sha(c['manifest'])
    tok=AutoTokenizer.from_pretrained(str(cp),local_files_only=True)
    sys.path.insert(0,str(SOURCE.parent))
    if model=='ling':
        from generate_ling import _eos_ids
        eos,eos_source=_eos_ids(cp)
    else:
        from generate_ring import _eos_ids
        eos=_eos_ids(cp);eos_source='generation_config.json'
    pins={str(p):native.sha(p) for p in [dataset/'cases.json',dataset/'PLAN.json',dataset/'grader.py',dataset/'evidence/prepared_actions.json',dataset/'evidence/retention_probes.json',Path(__file__)]}
    native.dump(out/'PLAN.json',{'source_pins':pins,'model':model,'manifest_sha256':manifest,'train':16,'heldout':16,'retention':4,'arms':['U','S','C'],'steps':20,'lr':.001,'rank':8,'alpha':16,'max_new_tokens':48,'timeout_seconds':600,'C_weights':'2 for baseline failed,1 for baseline passed; normalized to same total supervised token mass asS','selection':'max expert exposure over all corrected action training positions only','retention_policy':'compact each validated full trace to scored logits/capture rows plus original hashes, all JSON and routes retained','eos':eos,'eos_source':eos_source,'claim_limit':'small next-action simulator; not general tool capability or ToolOPD; no promotion'})
    ledger=out/'events.jsonl';cost={'baseline_collection':0.,'capture_collection':0.,'training_S':0.,'training_C':0.,'export_checks':0.}
    failures={};features=[];targets=[];row_groups=[]
    for i,r in enumerate(train):
        d=out/'train-baseline'/r['case_id'];receipt=native.run(model,r['prompt_ids'],d,generate=48,eos=eos,timeout=600)
        text,complete=decode_result(tok,receipt,eos);grade=grader.grade_case(cases[r['case_id']],text)
        if not complete:grade.update(correct=False,error='incomplete',valid=False)
        failures[r['case_id']]=not grade['correct'];cost['baseline_collection']+=receipt['wall_seconds']
        jsonl(ledger,{'stage':'train-baseline','index':i,'case_id':r['case_id'],'text':text,'complete':complete,'grade':grade,'wall_seconds':receipt['wall_seconds']})
        compact(d,[len(r['prompt_ids'])-1])
        full,positions,target=rows_and_targets(r)
        if grader.grade_case(cases[r['case_id']],r['corrected_action'])['correct'] is not True:raise ValueError('positive target not verified')
        d=out/'train-capture'/r['case_id'];receipt=native.run(model,full,d,capture=True,timeout=600)
        capture=terminal.load_features(d);f=capture.rows[positions];features.append(f);targets+=target;row_groups.append(len(f));cost['capture_collection']+=receipt['wall_seconds']
        compact(d,positions)
        jsonl(ledger,{'stage':'train-capture','index':i,'case_id':r['case_id'],'rows':len(f),'wall_seconds':receipt['wall_seconds']})
    features=torch.cat(features).cuda();target=torch.tensor(targets,dtype=torch.long,device='cuda');expert,exposure=qualify.nomination(features,c['hidden'])
    native.dump(out/'NOMINATION.json',exposure)
    torch.manual_seed(61021);torch.cuda.manual_seed_all(61021)
    t=terminal.build_terminal(cp,model,expert,device='cuda');initial={k:v.detach().clone() for k,v in t.named_parameters()};frozen=qualify.tensors_digest(t.named_buffers())
    train_receipts={};adapters={'U':None};candidate_digests={}
    # Both arms freeze before any held-out generation or scoring.
    for arm in ['S','C']:
        with torch.no_grad():
            for name,p in t.named_parameters():p.copy_(initial[name])
        torch.manual_seed(61021);torch.cuda.manual_seed_all(61021)
        opt=torch.optim.Adam(t.parameters(),lr=.001,foreach=False)
        weights=weights_for(train,failures,arm).cuda();start=time.monotonic();losses=[]
        for k in range(20):
            result=qualify.step(t,opt,features,target,weights);losses.append(result['loss'])
            jsonl(ledger,{'stage':'training','arm':arm,'step':k+1,'loss':result['loss']})
        torch.cuda.synchronize();cost['training_'+arm]=time.monotonic()-start
        final_params={k:v.detach().clone() for k,v in t.named_parameters()}
        path=out/(arm+'.adapter');terminal.save_adapter(t,path,cp,checkpoint_sha256=manifest)
        with torch.no_grad():
            for p in t.parameters():p.zero_()
        terminal.load_adapter(t,path,checkpoint_sha256=manifest);qualify.tree_exact(dict(t.named_parameters()),final_params)
        if qualify.tensors_digest(t.named_buffers())!=frozen:raise ValueError('frozen buffers changed')
        # Choose the training case with the greatest nominated expert exposure, never held-out.
        offset=0;exposures=[]
        for n in row_groups:
            ids=features[offset:offset+n,3*c['hidden']:3*c['hidden']+8];exposures.append(int((ids==expert).sum()));offset+=n
        j=int(np.argmax(exposures));r=train[j];offset=sum(row_groups[:j]);n=row_groups[j]
        with torch.no_grad():expected=t(features[offset:offset+n]).cpu().numpy()
        full,positions,_=rows_and_targets(r);d=out/('native-'+arm+'-forward')
        t.cpu();features=features.cpu();target=target.cpu();weights=weights.cpu();gc.collect();torch.cuda.empty_cache()
        receipt=native.run(model,full,d,adapter=path,capture=True,timeout=600);cost['export_checks']+=receipt['wall_seconds']
        t.cuda();features=features.cuda();target=target.cuda()
        observed=np.memmap(d/'logits.f32',dtype='<f4',mode='r').reshape(-1,157184)[positions]
        parity=qualify.compare_logits(expected,observed);native.dump(out/(arm+'-FORWARD.json'),parity)
        compact(d,positions)
        if not parity['passed']:raise ValueError(arm+' trained native forward gate failed')
        train_receipts[arm]={'losses':losses,'wall_seconds':cost['training_'+arm],'weight_sum':float(weights.sum()),'tokens':len(weights),'export_reload_exact':True,'native_forward':parity}
        adapters[arm]=path;candidate_digests[arm]=native.sha(path)
        del opt,weights,final_params;gc.collect();torch.cuda.empty_cache()
    native.dump(out/'FROZEN-ADAPTERS.json',{'adapters':candidate_digests,'before_heldout':True,'source_pins':pins})
    del t,features,target,initial;gc.collect();torch.cuda.empty_cache()
    evaluations=[]
    for i,r in enumerate(held+retention):
        is_retention=i>=len(held);caseid='retention-'+str(r['probe_index']) if is_retention else r['case_id']
        for arm in arm_order(i):
            d=out/'eval'/caseid/arm
            receipt=native.run(model,r['prompt_ids'],d,adapter=adapters[arm],generate=48,eos=eos,timeout=600)
            text,complete=decode_result(tok,receipt,eos)
            if is_retention:grade={'correct':bool(complete and text==r['target_answer']),'valid':complete,'expected':r['target_answer']}
            else:
                grade=grader.grade_case(cases[caseid],text)
                if not complete:grade.update(correct=False,valid=False,error='incomplete')
            entry={'stage':'evaluation','case_id':caseid,'arm':arm,'retention':is_retention,'text':text,'complete':complete,'grade':grade,'wall_seconds':receipt['wall_seconds'],'generated_tokens':len(receipt['generation']['generated_token_ids']),'finish_reason':receipt['generation']['finish_reason']}
            evaluations.append(entry);jsonl(ledger,entry);compact(d,list(range(len(r['prompt_ids'])-1,receipt['execution']['tokens'])))
    if any(native.sha(p)!=v for p,v in pins.items()):raise ValueError('frozen pilot inputs changed')
    for a,p in adapters.items():
        if p and native.sha(p)!=candidate_digests[a]:raise ValueError('candidate changed duringheldout')
    sums={}
    for arm in ['U','S','C']:
        rows=[r for r in evaluations if r['arm']==arm and not r['retention']];ret=[r for r in evaluations if r['arm']==arm and r['retention']]
        seconds=sum(r['wall_seconds'] for r in rows);correct=sum(r['grade']['correct'] for r in rows)
        sums[arm]={'n':len(rows),'correct':correct,'invalid':sum(not r['grade']['valid'] for r in rows),'incomplete':sum(not r['complete'] for r in rows),'inference_seconds':seconds,'correct_per_inference_hour':3600*correct/seconds,'retention_correct':sum(r['grade']['correct'] for r in ret),'retention_n':len(ret)}
    result={'completed':True,'model':model,'arms':sums,'training':train_receipts,'cost_seconds':cost,'training_baseline_failures':sum(failures.values()),'expert':expert,'credit_contrast_identifiable':0<sum(failures.values())<len(train),'scope':'single next-action pilot, not general capability or ToolOPD','promoted':False,'no_quantization':True}
    native.dump(out/'RESULT.json',result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=native.MODELS);ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    with open('/work/AI/experiments/strata-dev8-gpu.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
        before=qualify.bind_checkpoint(a.model);gpu=qualify.gpu_processes();native.dump(a.output.with_suffix('.binding-before.json'),before)
        start=time.monotonic();error=None
        try:run(a.model,a.dataset,a.output)
        except BaseException as e:error=repr(e);raise
        finally:
            after=qualify.bind_checkpoint(a.model);native.dump(a.output.with_suffix('.binding-after.json'),after)
            try:qualify.check_foreign(gpu);foreign=True
            except RuntimeError:foreign=False
            native.dump(a.output.with_suffix('.RETURN.json'),{'error':error,'checkpoint_exact':before==after,'foreign_gpu_preserved':foreign,'wall_seconds':time.monotonic()-start})
            if before!=after:raise RuntimeError('basecheckpointchanged')

if __name__=='__main__':main()
