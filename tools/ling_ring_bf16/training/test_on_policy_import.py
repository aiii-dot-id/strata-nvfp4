
import copy,json
from pathlib import Path
import pytest
import on_policy_runner as r

def write(p,v):p.write_text(json.dumps(v))
def fixture(root):
    root.mkdir()
    binding={'shard':{'sha256':'original','size':9}}
    pins={'suite':'suite','executable':'exe','manifest':'manifest','native_profile':['p'],'parent_checkpoint_digest':r.digest(binding),'source_closure':{'native.cpp':{'sha256':'unchanged','size':2}}}
    write(root/'PLAN.json',{'model':'ling','pins':pins})
    write(root.with_suffix('.binding-before.json'),binding);write(root.with_suffix('.binding-after.json'),binding)
    write(root.with_suffix('.RETURN.json'),{'checkpoint_exact':True,'foreign_gpu_preserved':True,'error':None})
    for name,raw in [('zero.adapter',b'zero'),('rlvr-step-1.adapter',b'step1'),('rlvr-step-2.adapter',b'step2'),('rlvr-resumed.adapter',b'step2')]:
        p=root/name;p.write_bytes(raw)
        write(Path(str(p)+'.json'),{'payload_sha256':r.native.sha(p),'checkpoint_sha256':'manifest','model':'ling'})
    q={'export_reload_exact':True,'training_state_restore_exact':True,'optimizer_rng_exact':True,'native_forward':{'passed':True},'steps':[{'objective':'rlvr','gradient_norm':1.},{'objective':'rlvr','gradient_norm':2.}],'adapter_sha256':r.native.sha(root/'rlvr-step-2.adapter')}
    write(root/'rlvr-QUALIFIED.json',q)
    st=root/'rlvr-snapshot-step1';st.mkdir();(st/'tensors.safetensors').write_bytes(b'state')
    write(st/'state.json',{'payload_sha256':r.native.sha(st/'tensors.safetensors'),'config_binding':{'snapshot_adapter_sha256':r.native.sha(root/'rlvr-step-1.adapter'),'parent_checkpoint_digest':pins['parent_checkpoint_digest']}})
    return pins

def test_import_completed_endpoint_and_binds_hashes(tmp_path):
    p=tmp_path/'prior';pins=fixture(p);q,origin=r.admit_rlvr_endpoint(p,'ling',pins)
    assert q['adapter_sha256']==origin['adapter_sha256']
    assert origin['qualification_sha256']==r.native.sha(p/'rlvr-QUALIFIED.json')

@pytest.mark.parametrize('mutation',['restore','payload','snapshot','source','parent','model','error','heldout'])
def test_import_rejects_drift(tmp_path,mutation):
    p=tmp_path/'prior';pins=fixture(p)
    if mutation=='restore':(p/'rlvr-resumed.adapter').write_bytes(b'wrong')
    elif mutation=='payload':(p/'rlvr-step-2.adapter').write_bytes(b'wrong')
    elif mutation=='snapshot':(p/'rlvr-snapshot-step1/tensors.safetensors').write_bytes(b'wrong')
    elif mutation=='source':pins=copy.deepcopy(pins);pins['source_closure']['native.cpp']['sha256']='changed'
    elif mutation=='heldout':(p/'evaluation').mkdir()
    elif mutation=='parent':pins['parent_checkpoint_digest']='wrong'
    elif mutation=='model':
        with pytest.raises(ValueError):r.admit_rlvr_endpoint(p,'ring',pins)
        return
    elif mutation=='error':write(p.with_suffix('.RETURN.json'),{'checkpoint_exact':True,'foreign_gpu_preserved':True,'error':'unrecorded failure'})
    with pytest.raises((ValueError,FileNotFoundError)):r.admit_rlvr_endpoint(p,'ling',pins)
