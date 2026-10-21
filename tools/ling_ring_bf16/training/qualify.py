"""Real Strata terminal adapter qualification. AI Identity Incorported <james@aiii.id>."""
from __future__ import annotations
import argparse, concurrent.futures, copy, fcntl, gc, hashlib, json, os, signal, subprocess, time
from pathlib import Path
import numpy as np
import torch
import native
import terminal

ROOT=native.ROOT
EXP=native.EXPERIMENT


def tensors_digest(named):
    out={}
    for k,t in named:
        a=t.detach().cpu().contiguous()
        out[k]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.numpy().tobytes()).hexdigest()}
    return out


def tree_exact(a,b,path='root'):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a.cpu(),b.cpu()): raise ValueError('tree mismatch '+path)
    elif isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b): raise ValueError('tree keys '+path)
        for k in a:tree_exact(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b): raise ValueError('tree list '+path)
        for i,(x,y) in enumerate(zip(a,b)):tree_exact(x,y,path+'.'+str(i))
    elif a!=b:raise ValueError('tree scalar '+path)


def bind_checkpoint(model):
    cp=Path(native.MODELS[model]['checkpoint'])
    paths=sorted(p for p in cp.rglob('*') if p.is_file() and p.suffix in ('.safetensors','.json','.py','.jinja'))
    def one(p):return str(p.relative_to(cp)),{'size':p.stat().st_size,'sha256':native.sha(p)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:binding=dict(pool.map(one,paths))
    return binding


def gpu_processes():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    result=[]
    for line in text.strip().splitlines():
        pid,name,mem=[s.strip() for s in line.split(',',2)]
        try:start=Path('/proc/'+pid+'/stat').read_text().split()[21]
        except FileNotFoundError:continue
        result.append({'pid':int(pid),'name':name,'mib':int(mem),'start':start})
    return result


def check_foreign(before):
    for p in before:
        try:now=Path('/proc/'+str(p['pid'])+'/stat').read_text().split()[21]
        except FileNotFoundError:raise RuntimeError('foreign GPU process disappeared '+str(p))
        if now!=p['start']:raise RuntimeError('foreign GPU process changed '+str(p))


def compare_logits(actual,expected):
    a=np.asarray(actual,dtype=np.float32);b=np.asarray(expected,dtype=np.float32)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('logit shape/finiteness')
    d=np.abs(a-b)
    rec={'shape':list(a.shape),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'argmax_equal':bool(np.array_equal(a.argmax(-1),b.argmax(-1))),'tolerance_pass':bool(np.all(d<=.001+.0001*np.abs(b)))}
    rec['passed']=rec['argmax_equal'] and rec['tolerance_pass']
    return rec


def step(t,opt,features,targets,weights=None):
    opt.zero_grad(set_to_none=True)
    # Each row preserves the native GEMV semantics. Only adapters have gradients.
    logits=t(features)
    losses=torch.nn.functional.cross_entropy(logits,targets,reduction='none')
    loss=(losses*(torch.ones_like(losses) if weights is None else weights)).sum()/len(losses)
    if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
    loss.backward()
    for name,p in t.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('invalid gradient '+name)
    grad=tensors_digest((k,p.grad) for k,p in t.named_parameters())
    opt.step()
    return {'loss':float(loss.detach()),'gradient':grad,'adapter':tensors_digest(t.named_parameters())}


def nomination(features,h):
    ids=features[:,3*h:3*h+8]
    if not torch.equal(ids,ids.round()):raise ValueError('noninteger expert IDs')
    counts=torch.bincount(ids.to(torch.int64).flatten(),minlength=256)
    expert=int(counts.argmax());return expert,{'expert':expert,'response_positions':len(ids),'exposed_positions':int(counts[expert]),'counts':counts.tolist()}


def qualify(model,output,*,tokens,targets,positions,expected_expert=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if len(tokens)!=11 or positions!=[7,8,9,10] or targets!=tokens[8:]+[808]:raise ValueError('fixed qualification token/position contract')
    c=native.MODELS[model];binding=native.sha(c['manifest']);cp=c['checkpoint']
    native.dump(output/'INPUT.json',{'model':model,'tokens':tokens,'targets':targets,'positions':positions,'manifest_sha256':binding})
    baseline=output/'native-original'
    native.run(model,tokens,baseline,capture=True,timeout=600)
    capture=terminal.load_features(baseline)
    features=capture.rows[positions].cuda();target=torch.tensor(targets,dtype=torch.long,device='cuda')
    expert,exposure=nomination(features,c['hidden'])
    if expected_expert is not None and expert!=expected_expert:raise ValueError('nomination mismatch')
    native.dump(output/'NOMINATION.json',exposure)
    torch.manual_seed(61021);torch.cuda.manual_seed_all(61021)
    t=terminal.build_terminal(cp,model,expert,device='cuda')
    def native_stage(path,adapter=None):
        nonlocal features,target
        t.cpu();features=features.cpu();target=target.cpu();gc.collect();torch.cuda.empty_cache()
        result=native.run(model,tokens,path,capture=True,adapter=adapter,timeout=600)
        t.cuda();features=features.cuda();target=target.cuda()
        return result
    try:
        buffers=tensors_digest(t.named_buffers())
        def export(path):return terminal.save_adapter(t,path,cp,checkpoint_sha256=binding)
        zero=output/'zero.adapter';export(zero)
        native_stage(output/'native-zero',zero)
        native.exact_native(baseline,output/'native-zero')
        original_logits=np.memmap(baseline/'logits.f32',dtype='<f4',mode='r').reshape(-1,157184)
        with torch.no_grad():initial=t(features).cpu().numpy()
        forward=compare_logits(initial,original_logits[positions])
        native.dump(output/'FORWARD.json',forward)
        if not forward['passed']:raise ValueError('initial trainer-native forward gate failed')
        opt=torch.optim.Adam(t.parameters(),lr=.001,foreach=False)
        conf={'model':model,'expert':expert,'rank':8,'alpha':16,'optimizer':'Adam','lr':.001,'checkpoint_sha256':binding,'torch':torch.__version__}
        datahash=hashlib.sha256(json.dumps({'capture':capture.data_hash,'targets':targets,'positions':positions},sort_keys=True).encode()).hexdigest()
        steps=[]
        for i in range(3):
            steps.append(step(t,opt,features,target))
            if i==0:terminal.save_training_state(t,opt,output/'snapshot-step1',step=1,iteration_cursor=1,data_hash=datahash,config_binding=conf)
        with torch.no_grad():trained_logits=t(features).cpu().numpy();after_loss=float(torch.nn.functional.cross_entropy(t(features),target))
        if after_loss>=steps[0]['loss']:raise ValueError('training loss did not reduce')
        if tensors_digest(t.named_parameters())==steps[0]['adapter']:raise ValueError('no continued adapter update')
        if buffers!=tensors_digest(t.named_buffers()):raise ValueError('frozen tensor changed')
        trained=output/'trained.adapter';export(trained)
        end_params=copy.deepcopy(dict(t.named_parameters()));end_opt=copy.deepcopy(opt.state_dict())
        end_rng=torch.get_rng_state().clone();end_cuda=[x.clone() for x in torch.cuda.get_rng_state_all()]
        restored=terminal.restore_training_state(t,opt,output/'snapshot-step1',data_hash=datahash,config_binding=conf)
        resumed=[step(t,opt,features,target),step(t,opt,features,target)]
        tree_exact(dict(t.named_parameters()),end_params);tree_exact(opt.state_dict(),end_opt)
        tree_exact(torch.get_rng_state(),end_rng);tree_exact(torch.cuda.get_rng_state_all(),end_cuda)
        tree_exact(resumed,steps[1:])
        export(output/'resumed.adapter')
        if native.sha(trained)!=native.sha(output/'resumed.adapter'):raise ValueError('resumed binary differs')
        # Mutate research parameters, reload the export, then compare actual values/logits.
        with torch.no_grad():
            for p in t.parameters():p.zero_()
        terminal.load_adapter(t,trained,checkpoint_sha256=binding)
        tree_exact(dict(t.named_parameters()),end_params)
        with torch.no_grad():tree_exact(t(features).cpu(),torch.from_numpy(trained_logits))
        native_stage(output/'native-trained',trained)
        native_trained=np.memmap(output/'native-trained'/'logits.f32',dtype='<f4',mode='r').reshape(-1,157184)
        adapted=compare_logits(trained_logits,native_trained[positions])
        native.dump(output/'ADAPTED-FORWARD.json',adapted)
        if not adapted['passed']:raise ValueError('adapted trainer-native forward gate failed')
        native_stage(output/'native-rollback')
        exact=native.exact_native(baseline,output/'native-rollback')
        if buffers!=tensors_digest(t.named_buffers()):raise ValueError('frozen tensor changed')
        summary={'passed':True,'model':model,'scope':'one terminal expert adapter; no base weights updated','expert':expert,'trainable_parameters':sum(p.numel() for p in t.parameters()),'exposure':exposure,'forward':forward,'adapted_forward':adapted,'initial_loss':steps[0]['loss'],'final_loss':after_loss,'steps':steps,'resumed_steps':resumed,'export_reload_exact':True,'training_state_restore_exact':True,'optimizer_and_rng_exact':True,'rollback_exact':exact,'base_buffers_unchanged':True}
        native.dump(output/'RESULT.json',summary)
        del opt,end_params,end_opt;gc.collect();torch.cuda.empty_cache()
        return summary
    finally:
        t.cpu();features=features.cpu();target=target.cpu();gc.collect();torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=native.MODELS,required=True);ap.add_argument('--input',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    payload=json.loads(args.input.read_text())
    with open('/work/AI/experiments/strata-dev8-gpu.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        before_gpu=gpu_processes();before=bind_checkpoint(args.model)
        native.dump(args.output.with_suffix('.binding-before.json'),before)
        start=time.monotonic();error=None;recovery=None
        try:qualify(args.model,args.output,**payload)
        except BaseException as e:
            error=repr(e)
            try:
                base=args.output/'native-original'
                if (base/'RUN.json').exists():
                    native.run(args.model,payload['tokens'],args.output/'native-failure-rollback',capture=True,timeout=600)
                    recovery=native.exact_native(base,args.output/'native-failure-rollback')
            except BaseException as e2:recovery={'error':repr(e2)}
            raise
        finally:
            after=bind_checkpoint(args.model);native.dump(args.output.with_suffix('.binding-after.json'),after)
            same=before==after
            try:check_foreign(before_gpu);foreign_ok=True
            except RuntimeError:foreign_ok=False
            native.dump(args.output.with_suffix('.RETURN.json'),{'error':error,'checkpoint_exact':same,'foreign_gpu_preserved':foreign_ok,'before_gpu':before_gpu,'after_gpu':gpu_processes(),'recovery':recovery,'wall_seconds':time.monotonic()-start})
            if not same:raise RuntimeError('checkpoint bytes changed')
            fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__':main()
