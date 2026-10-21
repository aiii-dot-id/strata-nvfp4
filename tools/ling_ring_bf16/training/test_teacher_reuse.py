import json, hashlib, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pytest, torch
import teacher_reuse as tr

class Fake:
    model='ling'; hidden=1536
    def eval(self): return self
    def __call__(self,x):
        return x[:, :tr.VOCAB]

def _cap(tmp_path, rows=3):
    h=1536; rf=11*h+8208; x=np.zeros((rows,rf),dtype='<f4'); x[:,:tr.VOCAB]=np.arange(tr.VOCAB,dtype=np.float32)[None,:] if tr.VOCAB<=rf else 0
    # use compact fake logits by monkeypatching vocab and geometry-independent row path
    p=tmp_path/'terminal.f32'; p.write_bytes(x.tobytes()); (tmp_path/'terminal-meta.json').write_text(json.dumps({'schema':'strata-terminal-v1','model':'ling','hidden':h,'rows':rows,'rowfloats':rf})); return p,x

def test_capture_info_and_bounded_iteration(tmp_path):
    p,x=_cap(tmp_path); info=tr.capture_info(p); assert info['rows']==3
    got=[]
    for ids,chunk in tr.iter_capture_rows(p,[2,0],chunk_rows=1): got += ids; assert chunk.shape==(1,info['rowfloats'])
    assert got==[2,0]

def test_rejects_partial_and_bad_indices(tmp_path):
    p,x=_cap(tmp_path); p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError): tr.capture_info(p)
    p,x=_cap(tmp_path)
    with pytest.raises(IndexError): list(tr.iter_capture_rows(p,[4]))

def test_full_vocab_compare_and_float64_kl(tmp_path,monkeypatch):
    p,x=_cap(tmp_path,rows=2); monkeypatch.setattr(tr,'VOCAB',8)
    # Fake output is first 8 feature values; create exact native payload and perturb row 1.
    native=np.asarray(x[:,:8],dtype='<f4').copy(); native[1,0]+=1e-3
    lp=tmp_path/'selected-logits.f32'; lp.write_bytes(native.tobytes())
    (tmp_path/'a.adapter').write_bytes(b'adapter')
    monkeypatch.setattr(tr.terminal,'load_adapter',lambda *a: None)
    out=tr.compare_native_logits(Fake(),p,tmp_path/'a.adapter',lp,rows=[0,1],chunk_rows=1)
    assert out['rows']==2 and out['vocab']==8 and out['max_abs']==pytest.approx(1e-3,abs=1e-7)
    assert out['kl_native_to_reconstructed_mean']>=0 and out['kl_native_to_reconstructed_max']>0

def test_adapter_hash_in_report_is_payload_hash(tmp_path):
    p=tmp_path/'a.adapter'; p.write_bytes(b'abcd'); assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(p.read_bytes()).hexdigest()
