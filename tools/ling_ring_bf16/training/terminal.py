#!/usr/bin/env python3
"""CPU-friendly terminal-expert LoRA trainer for Ling and Ring captures."""
from __future__ import annotations
import argparse, hashlib, json, math, os, struct, tempfile
from pathlib import Path
from typing import Any
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

MAGIC=b"STRALOR1"; AUTHOR="AI Identity Incorported <james@aiii.id>"
HIDDEN={"ling":1536,"ring":2048}; INTERMEDIATE=512

def _finite(t,name):
    if not torch.isfinite(t).all(): raise ValueError(f"non-finite tensor: {name}")
def _tensor(v,shape,name,dtype=torch.float32):
    t=torch.as_tensor(v,dtype=dtype)
    if tuple(t.shape)!=shape: raise ValueError(f"{name} shape {tuple(t.shape)} != {shape}")
    _finite(t,name); return t

class Capture:
    def __init__(self,rows,meta):
        self.rows,self.meta=rows,meta; self.hidden=int(meta["hidden"]); self.model=str(meta["model"]); self.expert=meta.get("expert"); self.data_hash=str(meta.get("data_sha256",""))
    def __len__(self): return self.rows.shape[0]

def load_features(path):
    p=Path(path)
    if p.is_dir(): dp,mp=p/"terminal.f32",p/"terminal-meta.json"
    elif p.name.endswith(".f32"): dp,mp=p,p.with_name("terminal-meta.json")
    else: raise ValueError("capture path must be directory or terminal.f32")
    meta=json.loads(mp.read_text(encoding="utf-8")); model=str(meta.get("model",meta.get("family",""))).lower()
    if model not in HIDDEN: raise ValueError("capture model must be ling or ring")
    hidden=int(meta.get("hidden",HIDDEN[model]));
    if hidden!=HIDDEN[model]: raise ValueError("unexpected hidden size")
    row_floats=11*hidden+8208
    if meta.get("schema") not in (None,"strata-terminal-v1"): raise ValueError("unsupported capture schema")
    if int(meta.get("intermediate",INTERMEDIATE))!=INTERMEDIATE or int(meta.get("topk",8))!=8: raise ValueError("capture metadata geometry mismatch")
    if int(meta.get("rowfloats",row_floats))!=row_floats: raise ValueError("capture rowfloats mismatch")
    raw=dp.read_bytes()
    if len(raw)%4 or len(raw)//4%row_floats: raise ValueError("capture has partial row")
    count=len(raw)//4//row_floats
    if "rows" in meta and int(meta["rows"])!=count: raise ValueError("capture row count mismatch")
    rows=torch.frombuffer(bytearray(raw),dtype=torch.float32).clone().reshape(-1,row_floats); _finite(rows,"capture")
    meta=dict(meta,model=model,hidden=hidden,data_sha256=hashlib.sha256(raw).hexdigest()); return Capture(rows,meta)

def _parts(rows,h):
    return (rows[:,:h],rows[:,h:2*h],rows[:,2*h:3*h],rows[:,3*h:3*h+8],rows[:,3*h+8:3*h+16],rows[:,3*h+16:3*h+16+4096].reshape(-1,8,512),rows[:,3*h+16+4096:3*h+16+8192].reshape(-1,8,512),rows[:,3*h+16+8192:].reshape(-1,8,h))

def _find_keys(checkpoint,model,expert):
    root=Path(checkpoint); idx=root/"model.safetensors.index.json"
    if idx.is_file(): names=list(json.loads(idx.read_text())["weight_map"])
    else:
        names=[]
        for s in sorted(root.glob("*.safetensors")):
            with safe_open(str(s),framework="pt",device="cpu") as f: names+=list(f.keys())
    layer=23 if model=="ling" else 19; out={}
    for proj in ("gate_proj","up_proj","down_proj"):
        c=[n for n in names if n.endswith(f"experts.{expert}.{proj}.weight") and f"layers.{layer}." in n]
        if not c: raise KeyError(f"cannot find expert {expert} {proj}")
        out[proj]=c[0]
    return out
def _read_indexed(checkpoint,name):
    root=Path(checkpoint); idx=root/"model.safetensors.index.json"; shard=root/json.loads(idx.read_text())["weight_map"][name] if idx.is_file() else next(root.glob("*.safetensors"))
    with safe_open(str(shard),framework="pt",device="cpu") as f: return f.get_tensor(name)

class TerminalTrainer(torch.nn.Module):
    def __init__(self,gate,up,down,model,expert,rank,alpha,device="cpu"):
        super().__init__(); h=HIDDEN[model]; self.model,self.expert,self.rank,self.alpha,self.hidden=model,expert,rank,float(alpha),h
        self.register_buffer("gate",_tensor(gate,(512,h),"gate")); self.register_buffer("up",_tensor(up,(512,h),"up")); self.register_buffer("down",_tensor(down,(h,512),"down"))
        def pair(a_shape,b_shape):
            # Random A / zero B keeps the initial function unchanged while
            # allowing B to receive a useful first gradient.
            return torch.nn.Parameter(torch.randn(a_shape,dtype=torch.float32)*0.01), torch.nn.Parameter(torch.zeros(b_shape,dtype=torch.float32))
        self.gate_a,self.gate_b=pair((rank,h),(512,rank)); self.up_a,self.up_b=pair((rank,h),(512,rank)); self.down_a,self.down_b=pair((rank,512),(h,rank)); self.to(device)
    @property
    def scale(self): return self.alpha/self.rank
    def _delta(self,a,b,x): return b@(a@x)*self.scale
    def forward(self,features,targets=None,selected=False):
        h=self.hidden; x,resid,shared,ids,weights,og,ou,oo=_parts(features,h); hidden=[]
        for n in range(x.shape[0]):
            slots=(ids[n].to(torch.int64)==int(self.expert)).nonzero(as_tuple=False).flatten()
            if slots.numel()>1: raise ValueError(f"capture row {n} contains duplicate selected expert {self.expert}")
            if slots.numel()==0:
                y=(weights[n,:,None]*oo[n]).sum(0)+shared[n]+resid[n]
                y=y+0.0*(self.gate_a.sum()+self.gate_b.sum()+self.up_a.sum()+self.up_b.sum()+self.down_a.sum()+self.down_b.sum())
            else:
                s=int(slots[0]); ex=oo[n].clone(); g=og[n,s]+self._delta(self.gate_a,self.gate_b,x[n]); u=ou[n,s]+self._delta(self.up_a,self.up_b,x[n]); act=torch.nn.functional.silu(g)*u; ex[s]=self.down@act+self._delta(self.down_a,self.down_b,act); y=(weights[n,:,None]*ex).sum(0)+shared[n]+resid[n]
            hidden.append(y*torch.rsqrt(y.square().mean()+1e-6))
        hidden=torch.stack(hidden)
        if hasattr(self,"norm_weight"): hidden=hidden*self.norm_weight
        if hasattr(self,"lm_head"):
            logits=torch.stack([torch.mv(self.lm_head,row) for row in hidden])
            if selected:
                if targets is None: raise ValueError("targets required")
                return logits.gather(1,targets[:,None]).squeeze(1)
            return logits
        return hidden
    def loss(self,features,targets): return torch.nn.functional.cross_entropy(self(features),targets)

def build_terminal(modelcheckpoint,model,expert,rank=8,alpha=16,device="cpu"):
    if model not in HIDDEN or not 1<=rank<=32 or expert<0: raise ValueError("invalid model/rank/expert")
    root=Path(modelcheckpoint); k=_find_keys(root,model,expert); t=TerminalTrainer(_read_indexed(root,k["gate_proj"]),_read_indexed(root,k["up_proj"]),_read_indexed(root,k["down_proj"]),model,expert,rank,alpha,device)
    # Norm and head are optional for feature-only gradient fixtures.  If the
    # checkpoint exposes them, retain them as frozen CPU buffers.
    idx=root/"model.safetensors.index.json"; names=list(json.loads(idx.read_text())["weight_map"]) if idx.is_file() else []
    if not names: raise ValueError("checkpoint index with norm and lm_head is required")
    found_norm=found_head=False
    for n in names:
        if n.endswith("model.norm.weight") or n.endswith("transformer.norm.weight"):
            t.register_buffer("norm_weight",_tensor(_read_indexed(root,n).float().to(device),(HIDDEN[model],),"norm_weight")); found_norm=True; break
    for n in names:
        if n.endswith("lm_head.weight"):
            t.register_buffer("lm_head",_tensor(_read_indexed(root,n).float().to(device),(157184,HIDDEN[model]),"lm_head")); found_head=True; break
    if not found_norm or not found_head: raise KeyError("terminal training requires model norm and lm_head")
    return t

def save_adapter(trainer,path,checkpoint=None,checkpoint_sha256=None):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tensors={k:v.detach().cpu().contiguous() for k,v in trainer.named_parameters()}
    for k,v in tensors.items(): _finite(v,k)
    raw=bytearray(MAGIC)+bytearray(struct.pack("<IIIf",1 if trainer.model=="ling" else 2,trainer.expert,trainer.rank,trainer.alpha))
    for k in ("gate_a","gate_b","up_a","up_b","down_a","down_b"): raw+=tensors[k].numpy().tobytes(order="C")
    payload=bytes(raw); meta={"format":"strata-terminal-lora-v1","author":AUTHOR,"model":trainer.model,"expert":trainer.expert,"rank":trainer.rank,"alpha":trainer.alpha,"hidden":trainer.hidden,"checkpoint":str(checkpoint) if checkpoint else None,"checkpoint_sha256":checkpoint_sha256,"payload_sha256":hashlib.sha256(payload).hexdigest()}
    fd,tmp=tempfile.mkstemp(prefix=p.name+".",dir=str(p.parent)); os.close(fd)
    try: Path(tmp).write_bytes(payload); os.replace(tmp,p)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    Path(str(p)+".json").write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n")
    return meta
def load_adapter(trainer,path,checkpoint_sha256=None):
    raw=Path(path).read_bytes()
    if len(raw)<24 or raw[:8]!=MAGIC: raise ValueError("bad adapter magic")
    model_id,expert,rank,alpha=struct.unpack_from("<IIIf",raw,8); pos=24
    if model_id!=(1 if trainer.model=="ling" else 2) or expert!=trainer.expert or rank!=trainer.rank or abs(alpha-trainer.alpha)>1e-6: raise ValueError("adapter binding mismatch")
    shapes=[(rank,trainer.hidden),(512,rank),(rank,trainer.hidden),(512,rank),(rank,512),(trainer.hidden,rank)]; names=("gate_a","gate_b","up_a","up_b","down_a","down_b")
    for name,shape in zip(names,shapes):
        n=math.prod(shape)*4; end=pos+n
        if end>len(raw): raise ValueError("truncated adapter")
        t=torch.frombuffer(bytearray(raw[pos:end]),dtype=torch.float32).clone().reshape(shape); _finite(t,name); pos=end
    if pos != len(raw): raise ValueError("adapter has trailing bytes")
    try: meta=json.loads(Path(str(path)+".json").read_text())
    except Exception as e: raise ValueError("missing adapter metadata") from e
    if meta.get("payload_sha256")!=hashlib.sha256(raw[:pos]).hexdigest() or meta.get("model")!=trainer.model or int(meta.get("expert",-1))!=trainer.expert or int(meta.get("rank",-1))!=trainer.rank or abs(float(meta.get("alpha",float("nan")))-trainer.alpha)>1e-6 or int(meta.get("hidden",-1))!=trainer.hidden or (checkpoint_sha256 is not None and meta.get("checkpoint_sha256")!=checkpoint_sha256): raise ValueError("adapter metadata binding mismatch")
    for name,t in zip(names,[torch.frombuffer(bytearray(raw[24+sum(math.prod(s)*4 for s in shapes[:i]):24+sum(math.prod(s)*4 for s in shapes[:i+1])]),dtype=torch.float32).clone().reshape(s) for i,(name,s) in enumerate(zip(names,shapes))]):
        getattr(trainer,name).data.copy_(t)
    return meta


def save_training_state(trainer,optimizer,path,*,step,iteration_cursor,data_hash,config_binding):
    p=Path(path); p.mkdir(parents=True,exist_ok=True); tensors={k:v.detach().cpu().contiguous() for k,v in trainer.named_parameters()}
    osd=optimizer.state_dict(); slots=[]
    if type(optimizer) is not torch.optim.Adam or len(osd["param_groups"])!=1: raise ValueError("only single-group Adam snapshots are supported")
    for k,v in osd["state"].items():
        for n,x in v.items():
            if not torch.is_tensor(x): raise ValueError(f"non-tensor optimizer slot: {k}.{n}")
            tensors[f"optim.{k}.{n}"]=x.detach().cpu().contiguous(); slots.append(f"optim.{k}.{n}")
    tensors["rng.cpu"]=torch.get_rng_state().cpu().contiguous()
    if torch.cuda.is_available():
        for i,state in enumerate(torch.cuda.get_rng_state_all()): tensors[f"rng.cuda.{i}"]=state.cpu().contiguous()
    save_file(tensors,str(p/"tensors.safetensors"),metadata={"author":AUTHOR})
    payload=(p/"tensors.safetensors").read_bytes()
    groups=json.loads(json.dumps(osd["param_groups"],sort_keys=True))
    meta={"format":"strata-terminal-training-v2","author":AUTHOR,"step":int(step),"iteration_cursor":int(iteration_cursor),"data_hash":data_hash,"config_binding":config_binding,"parameter_names":[n for n,_ in trainer.named_parameters()],"optimizer_class":"Adam","optimizer_param_groups":groups,"payload_sha256":hashlib.sha256(payload).hexdigest()}
    (p/"state.json").write_text(json.dumps(meta,sort_keys=True,indent=2)+chr(10))

def restore_training_state(trainer,optimizer,path,*,data_hash,config_binding):
    p=Path(path); meta=json.loads((p/"state.json").read_text())
    if meta.get("format")!="strata-terminal-training-v2" or meta.get("data_hash")!=data_hash or meta.get("config_binding")!=config_binding: raise ValueError("training state binding mismatch")
    payload=(p/"tensors.safetensors").read_bytes()
    if meta.get("payload_sha256")!=hashlib.sha256(payload).hexdigest(): raise ValueError("training payload digest mismatch")
    tensors=load_file(str(p/"tensors.safetensors")); names=[n for n,_ in trainer.named_parameters()]
    osd=optimizer.state_dict(); groups=json.loads(json.dumps(osd["param_groups"],sort_keys=True))
    if type(optimizer) is not torch.optim.Adam or len(osd["param_groups"])!=1: raise ValueError("only single-group Adam snapshots are supported")
    if meta.get("parameter_names")!=names or meta.get("optimizer_param_groups")!=groups: raise ValueError("training optimizer binding mismatch")
    expected=set(names)|{"rng.cpu"}|{f"optim.{i}.{slot}" for i in range(len(names)) for slot in ("step","exp_avg","exp_avg_sq")}
    if torch.cuda.is_available(): expected|={f"rng.cuda.{i}" for i in range(torch.cuda.device_count())}
    if set(tensors)!=expected: raise ValueError("training parameter/optimizer key set mismatch")
    if "rng.cpu" not in tensors or tensors["rng.cpu"].dtype!=torch.uint8 or tensors["rng.cpu"].ndim!=1 or tensors["rng.cpu"].numel()!=torch.get_rng_state().numel(): raise ValueError("missing or invalid CPU RNG state")
    for key in expected- set(names)-{f"optim.{i}.{slot}" for i in range(len(names)) for slot in ("step","exp_avg","exp_avg_sq")}-{"rng.cpu"}:
        if tensors[key].dtype!=torch.uint8: raise ValueError("invalid CUDA RNG state")
    param_by_idx=[p for g in optimizer.param_groups for p in g["params"]]
    for key in expected-set(names)-{"rng.cpu"}:
        if not key.startswith("optim."): continue
        parts=key.split(".",2)
        if len(parts)!=3 or not parts[1].isdigit() or int(parts[1])>=len(param_by_idx): raise ValueError("invalid optimizer slot key")
        x=tensors[key]; param=param_by_idx[int(parts[1])]
        if parts[2]=="step":
            if tuple(x.shape)!=() or x.dtype not in (torch.float32,torch.float64) or x.item()<0 or x.item()!=meta["step"]: raise ValueError(f"optimizer step mismatch: {key}")
        elif tuple(x.shape)!=tuple(param.shape) or x.dtype!=param.dtype: raise ValueError(f"optimizer slot shape/dtype mismatch: {key}")
        _finite(x.float(),key)
    for n in names:
        if tuple(tensors[n].shape)!=tuple(getattr(trainer,n).shape) or tensors[n].dtype!=getattr(trainer,n).dtype: raise ValueError(f"training tensor mismatch: {n}")
        _finite(tensors[n],n)
    for key in expected-set(names)-{"rng.cpu"}:
        if not key.startswith("optim."): continue
        _,idx,slot=key.split(".",2)
        # Adam keeps its scalar step on CPU for the default non-capturable
        # configuration; moving it to a CUDA parameter device changes the
        # optimizer state contract and can alter continuation behavior.
        target_device=torch.device("cpu") if slot=="step" else param_by_idx[int(idx)].device
        optimizer.state[param_by_idx[int(idx)]][slot]=tensors[key].to(device=target_device).clone()
    for n in names: getattr(trainer,n).data.copy_(tensors[n])
    torch.set_rng_state(tensors["rng.cpu"])
    if torch.cuda.is_available(): torch.cuda.set_rng_state_all([tensors[f"rng.cuda.{i}"] for i in range(torch.cuda.device_count())])
    return meta

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--terminal-capture",type=int,default=0); ap.add_argument("path",nargs="?"); ap.parse_args(argv); return 0
if __name__=="__main__": raise SystemExit(main())
