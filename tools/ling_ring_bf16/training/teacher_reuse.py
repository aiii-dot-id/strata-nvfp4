"""Bounded CPU reuse of terminal captures for frozen-teacher full-vocabulary checks."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
from typing import Iterable
import numpy as np
import torch

try:
    from . import terminal
except ImportError:
    import terminal

VOCAB = 157184

def _paths(path):
    p=Path(path)
    if p.is_dir(): return p/'terminal.f32', p/'terminal-meta.json'
    if p.name.endswith('.f32'): return p, p.with_name('terminal-meta.json')
    raise ValueError('capture must be a directory or terminal.f32')

def capture_info(path):
    data,meta_path=_paths(path); meta=json.loads(meta_path.read_text())
    model=str(meta.get('model','')).lower(); hidden=terminal.HIDDEN.get(model)
    if hidden is None: raise ValueError('capture model must be ling or ring')
    rowfloats=11*hidden+8208
    if int(meta.get('rowfloats',rowfloats)) != rowfloats: raise ValueError('capture rowfloats mismatch')
    size=data.stat().st_size
    if size % (4*rowfloats): raise ValueError('capture has partial row')
    rows=size//(4*rowfloats)
    declared=meta.get('rows')
    if declared is not None and int(declared)!=rows and data.name not in ('selected-terminal.f32','terminal-selected.f32'):
        raise ValueError('capture rows mismatch')
    if meta.get('schema') not in (None,'strata-terminal-v1'): raise ValueError('unsupported capture schema')
    return {'path':str(data),'meta_path':str(meta_path),'meta':meta,'model':model,'hidden':hidden,'rowfloats':rowfloats,'rows':rows,'sha256':hashlib.sha256(data.read_bytes()).hexdigest()}

def _indices(rows, indices):
    if indices is None: return list(range(rows))
    out=[int(x) for x in indices]
    if any(x<0 or x>=rows for x in out): raise IndexError('capture row index out of range')
    return out

def iter_capture_rows(path, indices: Iterable[int] | None = None, chunk_rows: int = 8):
    if type(chunk_rows) is not int or chunk_rows<1: raise ValueError('chunk_rows must be positive int')
    info=capture_info(path); chosen=_indices(info['rows'],indices)
    arr=np.memmap(info['path'],dtype='<f4',mode='r',shape=(info['rows'],info['rowfloats']))
    for start in range(0,len(chosen),chunk_rows):
        take=chosen[start:start+chunk_rows]
        # The clone bounds the lifetime of each chunk and avoids retaining the mmap.
        x=torch.from_numpy(np.asarray(arr[take],dtype=np.float32)).clone()
        if not torch.isfinite(x).all(): raise ValueError('non-finite capture rows')
        yield take,x

def _load_logits(path, rows=None):
    p=Path(path); size=p.stat().st_size
    if size % (VOCAB*4): raise ValueError('native logits size mismatch')
    actual=size//(VOCAB*4)
    if rows is not None and actual < rows: raise ValueError('native logits row mapping out of range')
    return np.memmap(p,dtype='<f4',mode='r',shape=(actual,VOCAB))

def reconstruct_logits(trainer, capture_path, adapter_path, *, rows=None, chunk_rows=8):
    info=capture_info(capture_path); chosen=_indices(info['rows'],rows)
    if trainer.model != info['model'] or trainer.hidden != info['hidden']: raise ValueError('trainer/capture binding mismatch')
    terminal.load_adapter(trainer,adapter_path)
    trainer.eval(); outputs=[]
    with torch.no_grad():
        for _,features in iter_capture_rows(capture_path,chosen,chunk_rows):
            outputs.append(trainer(features))
    if not outputs: return np.empty((0,VOCAB),dtype=np.float32)
    return torch.cat(outputs,dim=0).detach().cpu().numpy()

def compare_native_logits(trainer, capture_path, adapter_path, native_logits_path, *, rows=None, native_rows=None, chunk_rows=8, temperature=1.0):
    if isinstance(temperature,bool) or not isinstance(temperature,(int,float)) or not math.isfinite(float(temperature)) or float(temperature)<=0: raise ValueError('temperature must be positive finite')
    info=capture_info(capture_path); chosen=_indices(info['rows'],rows)
    native_indices=chosen if native_rows is None else [int(x) for x in native_rows]
    if len(native_indices)!=len(chosen) or any(x<0 for x in native_indices): raise ValueError('native row mapping mismatch')
    native=_load_logits(native_logits_path,max(native_indices,default=-1)+1)
    got=reconstruct_logits(trainer,capture_path,adapter_path,rows=chosen,chunk_rows=chunk_rows)
    ref=np.asarray(native[native_indices],dtype=np.float32)
    if got.shape != ref.shape: raise ValueError('reconstructed/native shape mismatch')
    diff=np.abs(got.astype(np.float64)-ref.astype(np.float64)); flat=diff.ravel()
    # Compute KL in float64 to retain the tiny reverse-KL signal used by OPD.
    a=torch.from_numpy(ref.astype(np.float64))/float(temperature); b=torch.from_numpy(got.astype(np.float64))/float(temperature)
    lp=torch.log_softmax(a,dim=1); lq=torch.log_softmax(b,dim=1); kl=(lp.exp()*(lp-lq)).sum(dim=1)
    return {'rows':len(chosen),'vocab':VOCAB,'max_abs':float(flat.max()) if flat.size else 0.0,'mean_abs':float(flat.mean()) if flat.size else 0.0,'argmax_equal':bool(np.array_equal(np.argmax(ref,1),np.argmax(got,1))),'kl_native_to_reconstructed_mean':float(kl.mean()) if len(kl) else 0.0,'kl_native_to_reconstructed_max':float(kl.max()) if len(kl) else 0.0,'temperature':float(temperature),'capture_sha256':info['sha256'],'native_logits_sha256':hashlib.sha256(Path(native_logits_path).read_bytes()).hexdigest(),'adapter_sha256':hashlib.sha256(Path(adapter_path).read_bytes()).hexdigest()}
