# AI Identity Incorported <james@aiii.id>
#!/usr/bin/env python3
"""Original Ling-3.0-tiny FP32 reference; one decoder layer resident."""
from __future__ import annotations
import argparse,ast,gc,hashlib,importlib.util,json,mmap,resource,struct,sys,time,types
from pathlib import Path
class ReferenceError(RuntimeError): pass
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def readj(p): return json.loads(Path(p).read_text(),object_pairs_hook=lambda pairs: {k:v for k,v in pairs})
class SafeTensorIndex:
 def __init__(self,root):
  self.root=Path(root); self.index=readj(self.root/'model.safetensors.index.json'); self.weight_map=self.index['weight_map']; self.headers={}; self.files={}
  for name in sorted(set(self.weight_map.values())):
   p=self.root/name
   if Path(name).is_absolute() or '..' in Path(name).parts or not p.is_file() or p.is_symlink(): raise ReferenceError('unsafe/missing shard '+name)
   with p.open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]; raw=struct.pack('<Q',n)+f.read(n)
   self.files[name]={'path':p,'size':p.stat().st_size,'sha256':sha(p),'header_sha256':hashlib.sha256(raw).hexdigest(),'header_bytes':len(raw)}; self.headers[name]=json.loads(raw[8:])
 def pin(self):
  return {'config':self._pin(self.root/'config.json'),'index':self._pin(self.root/'model.safetensors.index.json'),'shards':{n:self._pin(v['path'])|{'header_sha256':v['header_sha256'],'header_bytes':v['header_bytes']} for n,v in sorted(self.files.items())}}
 def _pin(self,p): return {'path':str(p),'size':p.stat().st_size,'sha256':sha(p)}
 def raw(self,name,torch,device):
  sh=self.weight_map[name]; e=self.headers[sh][name]; a,b=e['data_offsets']; width=2 if e['dtype']=='BF16' else 4
  with self.files[sh]['path'].open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm: payload=bytes(mm[self.files[sh]['header_bytes']+a:self.files[sh]['header_bytes']+b])
  dt=torch.bfloat16 if e['dtype']=='BF16' else torch.float32
  return torch.frombuffer(payload,dtype=dt).clone().to(torch.float32).reshape(tuple(e['shape'])).to(device)
def write_report(report, out):
 out=Path(out); out.mkdir(parents=True,exist_ok=True)
 names=['final_hidden.fp32','logits.fp32','hidden_layers.fp32','layers.json','router_meta.json']
 report['outputs']={n:{'bytes':(out/n).stat().st_size,'sha256':sha(out/n)} for n in names}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

def load_source(path):
 path=Path(path); pkg=types.ModuleType('ling_reference'); pkg.__path__=[str(path.parent)]; sys.modules[pkg.__name__]=pkg
 for n in ['configuration_bailing_moe_v3','modeling_bailing_moe_v3']:
  p=path.parent/(n+'.py'); spec=importlib.util.spec_from_file_location(pkg.__name__+'.'+n,p); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)
 return sys.modules[pkg.__name__+'.modeling_bailing_moe_v3']
def pin_source(root,source): return {'modeling':{'path':str(source),'size':source.stat().st_size,'sha256':sha(source)},'configuration':{'path':str(source.parent/'configuration_bailing_moe_v3.py'),'size':(source.parent/'configuration_bailing_moe_v3.py').stat().st_size,'sha256':sha(source.parent/'configuration_bailing_moe_v3.py')},'config':{'path':str(Path(root)/'config.json'),'size':(Path(root)/'config.json').stat().st_size,'sha256':sha(Path(root)/'config.json')}}
def build_report(a,cfg,toks,hidden,logits,srcpins2,pins2,torch,out,started):
 legacy={'path':str(a.source),'size':Path(a.source).stat().st_size,
         'sha256':srcpins2['modeling']['sha256'],
         'configuration_sha256':srcpins2['configuration']['sha256']}
 return {'status':'executed','model':'Ling-3.0-tiny','activation_dtype':'FP32',
  'quantized':False,'reference':f'original Ling-3.0-tiny; {a.device.upper()} FP32 widened BF16',
  'device':a.device,'tokens':toks,'layers':cfg.num_hidden_layers,
  'hidden_size':cfg.hidden_size,'vocab_size':cfg.vocab_size,
  'hidden_layers_shape':[cfg.num_hidden_layers,len(toks),cfg.hidden_size],
  'final_hidden_shape':list(hidden.shape),'logits_shape':list(logits.shape),
  'logits_tokens':int(logits.shape[1]),'logits_mode':a.logits_mode,
  'source':legacy,'source_details':srcpins2,'checkpoint':pins2,
  'config':json.loads((Path(a.checkpoint)/'config.json').read_text()),
  'runtime':{'python':sys.version.split()[0],'torch':torch.__version__,
   'torch_cuda':torch.version.cuda,'tf32_matmul':bool(torch.backends.cuda.matmul.allow_tf32),
   'tf32_cudnn':bool(torch.backends.cudnn.allow_tf32),
   'elapsed_seconds':time.monotonic()-started,
   'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}

def run(a):
 import torch
 root=Path(a.checkpoint); source=Path(a.source); out=Path(a.output)
 if out.exists(): raise ReferenceError('refusing existing output')
 if a.device=='cpu': raise ReferenceError('Ling full forward requires CUDA; CPU is import/meta/test-only')
 if a.device=='cuda':
  if not torch.cuda.is_available(): raise ReferenceError('CUDA unavailable')
  torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
  free,_=torch.cuda.mem_get_info()
  if free<6*1024**3: raise ReferenceError('less than 6 GiB free CUDA memory')
 dev=torch.device(a.device); started=time.monotonic(); mod=load_source(source); cfg=mod.BailingMoeV3Config.from_pretrained(str(root)); cfg._attn_implementation='eager'
 toks=[int(x) for x in a.tokens.split(',') if x];
 if not toks or len(toks)>1024 or any(x<0 or x>=cfg.vocab_size for x in toks): raise ReferenceError('tokens must be 1..1024 and in vocabulary')
 idx=SafeTensorIndex(root); pins=idx.pin(); srcpins=pin_source(root,source)
 ids=torch.tensor([toks],dtype=torch.long,device=dev); hidden=idx.raw('model.embed_tokens.weight',torch,dev)[ids] if 'model.embed_tokens.weight' in idx.weight_map else idx.raw('model.word_embeddings.weight',torch,dev)[ids]
 pos=torch.arange(len(toks),device=dev,dtype=torch.long).unsqueeze(0); causal=torch.triu(torch.full((1,1,len(toks),len(toks)),-float('inf'),device=dev),diagonal=1); rotary=mod.BailingMoeV3RotaryEmbedding(cfg).to(dev); pos_emb=rotary(hidden,pos); hs=[]; layers=[]
 with torch.no_grad():
  for ln in range(cfg.num_hidden_layers):
   prefix=f'model.layers.{ln}.'; state={k[len(prefix):]:idx.raw(k,torch,dev) for k in idx.weight_map if k.startswith(prefix)}
   with torch.device('meta'): layer=mod.BailingMoeV3DecoderLayer(cfg,ln)
   missing,extra=layer.load_state_dict(state,strict=False,assign=True); layer.eval()
   if missing or extra: raise ReferenceError(f'layer {ln} state mismatch')
   captured={}; original=None; capture=None
   if hasattr(layer.mlp,'gate'):
    original=layer.mlp.gate.forward
    def capture(x,_original=original):
     value=_original(x); captured['ids']=value[0].detach().cpu(); captured['weights']=value[1].detach().float().cpu(); return value
    layer.mlp.gate.forward=capture
   try: hidden=layer(hidden,attention_mask=causal,position_ids=pos,position_embeddings=pos_emb,use_cache=False,output_router_logits=False)[0]
   finally:
    if original is not None: layer.mlp.gate.forward=original
   capture=None
   hs.append(hidden.detach().cpu().float()); layers.append({'layer':ln,'router_ids':captured.get('ids',torch.empty(0,dtype=torch.long)).tolist(),'router_weights':captured.get('weights',torch.empty(0)).tolist()}); del layer,state,captured,original,capture; gc.collect()
   if a.device=='cuda': torch.cuda.empty_cache()
  norm=mod.BailingMoeV3RMSNorm(cfg.hidden_size,eps=cfg.rms_norm_eps).to(dev); norm.load_state_dict({'weight':idx.raw('model.norm.weight',torch,dev)}); hidden=norm(hidden); head_hidden=hidden if a.logits_mode=='all' else hidden[:,-1:,:]; logits=torch.matmul(head_hidden.float(),idx.raw('lm_head.weight',torch,dev).transpose(0,1).float())
 if a.device=='cuda' and torch.cuda.max_memory_reserved()>5*1024**3: raise ReferenceError('CUDA peak reserved exceeds 5 GiB')
 pins2=idx.pin(); srcpins2=pin_source(root,source)
 if pins!=pins2 or srcpins!=srcpins2: raise ReferenceError('checkpoint/source pins changed')
 out.mkdir(parents=True); hidden.detach().cpu().float().numpy().tofile(out/'final_hidden.fp32'); logits.detach().cpu().float().numpy().tofile(out/'logits.fp32'); torch.stack(hs).numpy().tofile(out/'hidden_layers.fp32'); (out/'layers.json').write_text(json.dumps(layers)); router_meta=[{'layer':x['layer'],'router_ids':bool(x['router_ids']),'tokens':len(x['router_ids']),'top_k':8 if x['router_ids'] else 0} for x in layers]; (out/'router_meta.json').write_text(json.dumps(router_meta,indent=2)); report=build_report(a,cfg,toks,hidden,logits,srcpins2,pins2,torch,out,started)
 write_report(report,out); return report
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--source',required=True); p.add_argument('--tokens',required=True); p.add_argument('--output',required=True); p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--logits-mode',choices=['all','last'],default='all'); a=p.parse_args()
 try: print(json.dumps(run(a),indent=2))
 except (ReferenceError,OSError,ValueError,RuntimeError) as e: p.error(str(e))
if __name__=='__main__': main()
