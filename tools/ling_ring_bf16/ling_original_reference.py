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
def readj(p):
 try:
  return json.loads(Path(p).read_text(encoding='utf-8'),object_pairs_hook=_unique)
 except (OSError,UnicodeError,json.JSONDecodeError) as exc:
  raise ReferenceError(f'invalid JSON {p}: {exc}') from exc

def _unique(pairs):
 result={}
 for key,value in pairs:
  if key in result: raise ReferenceError(f'duplicate JSON key: {key}')
  result[key]=value
 return result

class SafeTensorIndex:
 def __init__(self,root):
  self.root=Path(root); self.index=readj(self.root/'model.safetensors.index.json')
  self.weight_map=self.index.get('weight_map',{})
  if not isinstance(self.weight_map,dict) or not self.weight_map or any(not isinstance(k,str) or not isinstance(v,str) or not k or not v for k,v in self.weight_map.items()): raise ReferenceError('invalid safetensors weight map')
  self.files={}; self.headers={}
  for name in sorted(set(self.weight_map.values())):
   path=self.root/name
   if not isinstance(name,str) or Path(name).is_absolute() or '..' in Path(name).parts or not path.is_file() or path.is_symlink(): raise ReferenceError(f'unsafe or missing shard: {name}')
   with path.open('rb') as f:
    raw_len=f.read(8)
    if len(raw_len)!=8: raise ReferenceError(f'truncated shard: {name}')
    header_len=struct.unpack('<Q',raw_len)[0]
    if header_len>64*1024*1024 or header_len+8>path.stat().st_size: raise ReferenceError(f'invalid header span: {name}')
    raw=raw_len+f.read(header_len)
   if len(raw)!=header_len+8: raise ReferenceError(f'truncated header: {name}')
   self.files[name]={'path':path,'size':path.stat().st_size,'sha256':sha(path),'header_sha256':hashlib.sha256(raw).hexdigest(),'header_bytes':len(raw)}
   header=json.loads(raw[8:].decode('utf-8'),object_pairs_hook=_unique)
   if not isinstance(header,dict): raise ReferenceError(f'invalid shard header: {name}')
   self.headers[name]=header
  for tensor,shard in self.weight_map.items():
   if shard not in self.headers or tensor not in self.headers[shard]: raise ReferenceError(f'index tensor absent from header: {tensor}')
 def pin(self):
  return {'config':self._pin(self.root/'config.json'),'index':self._pin(self.root/'model.safetensors.index.json'),'shards':{n:self._pin(v['path'])|{'header_sha256':v['header_sha256'],'header_bytes':v['header_bytes']} for n,v in sorted(self.files.items())}}
 def _pin(self,p): return {'path':str(p),'size':p.stat().st_size,'sha256':sha(p)}
 def raw(self,name,torch,device):
  if name not in self.weight_map: raise ReferenceError(f'tensor not indexed: {name}')
  shard=self.weight_map[name]; entry=self.headers[shard].get(name)
  if not isinstance(entry,dict): raise ReferenceError(f'invalid tensor header: {name}')
  dtype=entry.get('dtype')
  if dtype not in {'BF16','F32'}: raise ReferenceError(f'unsupported dtype for {name}: {dtype}')
  shape=entry.get('shape'); offsets=entry.get('data_offsets')
  if not isinstance(shape,list) or not all(type(x) is int and x>=0 for x in shape): raise ReferenceError(f'invalid shape for {name}')
  if not isinstance(offsets,list) or len(offsets)!=2 or not all(type(x) is int for x in offsets): raise ReferenceError(f'invalid offsets for {name}')
  start,end=offsets; info=self.files[shard]
  if start<0 or end<start: raise ReferenceError(f'invalid offsets for {name}')
  width=2 if dtype=='BF16' else 4; count=1
  for dim in shape: count*=dim
  if end-start!=count*width or info['header_bytes']+end>info['size']: raise ReferenceError(f'tensor bounds/length mismatch for {name}')
  with info['path'].open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm: payload=bytes(mm[info['header_bytes']+start:info['header_bytes']+end])
  dt=torch.bfloat16 if dtype=='BF16' else torch.float32
  return torch.frombuffer(payload,dtype=dt).clone().to(torch.float32).reshape(tuple(shape)).to(device)
def write_report(report, out):
 out=Path(out); out.mkdir(parents=True,exist_ok=True)
 names=['final_hidden.fp32','logits.fp32','hidden_layers.fp32','layers.json','router_meta.json']
 if (out/'route-probe.json').is_file(): names.append('route-probe.json')
 report['outputs']={n:{'bytes':(out/n).stat().st_size,'sha256':sha(out/n)} for n in names}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

def load_source(path):
 path=Path(path); pkg=types.ModuleType('ling_reference'); pkg.__path__=[str(path.parent)]; sys.modules[pkg.__name__]=pkg
 for n in ['configuration_bailing_moe_v3','modeling_bailing_moe_v3']:
  p=path.parent/(n+'.py'); spec=importlib.util.spec_from_file_location(pkg.__name__+'.'+n,p); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)
 return sys.modules[pkg.__name__+'.modeling_bailing_moe_v3']
def pin_source(root,source): return {'modeling':{'path':str(source),'size':source.stat().st_size,'sha256':sha(source)},'configuration':{'path':str(source.parent/'configuration_bailing_moe_v3.py'),'size':(source.parent/'configuration_bailing_moe_v3.py').stat().st_size,'sha256':sha(source.parent/'configuration_bailing_moe_v3.py')},'config':{'path':str(Path(root)/'config.json'),'size':(Path(root)/'config.json').stat().st_size,'sha256':sha(Path(root)/'config.json')},'reference':{'path':str(Path(__file__)),'size':Path(__file__).stat().st_size,'sha256':sha(Path(__file__))}}
def build_report(a,cfg,toks,hidden,logits,srcpins2,pins2,torch,out,started,decoding_mode="full",cache_metadata=None):
 legacy={'path':str(a.source),'size':Path(a.source).stat().st_size,
         'sha256':srcpins2['modeling']['sha256'],
         'configuration_sha256':srcpins2['configuration']['sha256']}
 return {'status':'executed','model':'Ling-3.0-tiny','activation_dtype':'FP32',
  'quantized':False,'reference':f'original Ling-3.0-tiny; {a.device.upper()} FP32 widened BF16',
  'device':a.device,'tokens':toks,'layers':cfg.num_hidden_layers,
  'hidden_size':cfg.hidden_size,'vocab_size':cfg.vocab_size,
  'hidden_layers_shape':[cfg.num_hidden_layers,len(toks),cfg.hidden_size],
  'final_hidden_shape':list(hidden.shape),'logits_shape':list(logits.shape),
  'logits_tokens':int(logits.shape[1]),'logits_mode':a.logits_mode,'decoding_mode':decoding_mode,'cache_metadata':cache_metadata or {},
  'source':legacy,'source_details':srcpins2,'checkpoint':pins2,
  'config':json.loads((Path(a.checkpoint)/'config.json').read_text()),
  'runtime':{'python':sys.version.split()[0],'torch':torch.__version__,
   'torch_cuda':torch.version.cuda,'tf32_matmul':bool(torch.backends.cuda.matmul.allow_tf32),
   'tf32_cudnn':bool(torch.backends.cudnn.allow_tf32),
   'elapsed_seconds':time.monotonic()-started,
   'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}

def validate_cached_trace(layer_rows,route_rows,tokens,layers):
 if len(layer_rows)!=layers or len(route_rows)!=layers: raise ReferenceError('cached trace layer count mismatch')
 for ln in range(layers):
  if len(layer_rows[ln])!=tokens: raise ReferenceError(f'cached layer {ln} token count mismatch')
  expected=0 if ln==0 else tokens
  if len(route_rows[ln]['router_ids'])!=expected or len(route_rows[ln]['router_weights'])!=expected: raise ReferenceError(f'cached layer {ln} route count mismatch')
 return True

def _parse_route_probe(text, token_count, layer_count):
 if not isinstance(text, str) or text.count(',') != 1:
  raise ReferenceError('route_probe must be POS,LAYER')
 try: pos, layer = (int(x, 10) for x in text.split(','))
 except (TypeError, ValueError) as exc:
  raise ReferenceError('route_probe must be POS,LAYER') from exc
 if pos < 0 or pos >= token_count:
  raise ReferenceError('route_probe position exceeds evaluated tokens')
 if layer < 1 or layer >= layer_count:
  raise ReferenceError('route_probe layer must be a sparse layer')
 return pos, layer

def run_cached(a,root,source,out,torch,mod,cfg,idx,pins,srcpins,dev,started,toks):
 if getattr(a,'logits_mode','all') != 'all': raise ReferenceError('cached decoding requires --logits-mode all')
 from transformers.cache_utils import DynamicCache
 embed_name='model.embed_tokens.weight' if 'model.embed_tokens.weight' in idx.weight_map else 'model.word_embeddings.weight'
 embeddings=idx.raw(embed_name,torch,dev)
 token_hiddens=[embeddings[torch.tensor([[tok]],dtype=torch.long,device=dev)].clone() for tok in toks]; del embeddings; gc.collect()
 route_rows=[{'layer':ln,'router_ids':[],'router_weights':[]} for ln in range(cfg.num_hidden_layers)]
 parsed_probe = getattr(a, 'route_probe', None)
 parsed_probe = _parse_route_probe(parsed_probe, len(toks), cfg.num_hidden_layers) if parsed_probe is not None else None
 route_probe_data = [None]
 layer_rows=[[] for _ in range(cfg.num_hidden_layers)]; rotary=mod.BailingMoeV3RotaryEmbedding(cfg).to(dev)
 with torch.no_grad():
  for ln in range(cfg.num_hidden_layers):
   prefix=f'model.layers.{ln}.'; state={k[len(prefix):]:idx.raw(k,torch,dev) for k in idx.weight_map if k.startswith(prefix)}
   with torch.device('meta'): layer=mod.BailingMoeV3DecoderLayer(cfg,ln)
   missing,extra=layer.load_state_dict(state,strict=False,assign=True); layer.eval()
   if missing or extra: raise ReferenceError(f'layer {ln} state mismatch')
   cache=DynamicCache(); next_hiddens=[]; probe_calls=[0]
   for pos_i,hidden in enumerate(token_hiddens):
    position_ids=torch.tensor([[pos_i]],dtype=torch.long,device=dev); cache_position=position_ids.view(-1)
    captured={}; original=None; capture=None
    probe_hidden=None; post_capture={}; post_handle=None
    if parsed_probe is not None and pos_i == parsed_probe[0] and ln == parsed_probe[1]:
     probe_hidden=hidden.detach().float().cpu().reshape(-1).tolist()
     def capture_post(_module,args):
      if not args or args[0].numel() != cfg.hidden_size:
       raise ReferenceError('cached post-attention probe geometry mismatch')
      post_capture['value']=args[0].detach().float().cpu().reshape(-1).tolist()
     post_handle=layer.post_attention_layernorm.register_forward_pre_hook(capture_post)
    if hasattr(layer.mlp,'gate'):
     original=layer.mlp.gate.forward
     def capture(x,_original=original,_layer_no=ln):
      value=_original(x); captured['ids']=value[0].detach().cpu(); captured['weights']=value[1].detach().float().cpu()
      if parsed_probe is not None and _layer_no == parsed_probe[1]:
       flat_x=x.reshape(-1,x.shape[-1]); flat_logits=value[2].reshape(-1,value[2].shape[-1])
       if flat_x.shape[0] != 1 or probe_calls[0] >= len(toks):
        raise ReferenceError('cached route probe call geometry mismatch')
       row=probe_calls[0]; probe_calls[0] += 1
       if row == parsed_probe[0]:
        logits_device=flat_logits[0].detach().float(); scores_device=torch.sigmoid(logits_device)
        bias_device=layer.mlp.gate.expert_bias.detach().float().reshape(-1)
        corrected_device=scores_device+bias_device
        route_probe_data[0]={'position':parsed_probe[0],'layer':parsed_probe[1],
         'score_reconstruction':'torch.sigmoid(router_logits) and bias correction recomputed on the original logits device from the same gate output',
         'input_hidden':probe_hidden,
         'post_attention_hidden':post_capture.get('value'),
         'gate_input':flat_x[0].detach().float().cpu().tolist(),
         'router_logits':logits_device.cpu().tolist(),
         'sigmoid_scores':scores_device.cpu().tolist(),
         'expert_bias':bias_device.cpu().tolist(),
         'bias_corrected_scores':corrected_device.cpu().tolist(),
         'selected_expert_ids':value[0].reshape(-1,value[0].shape[-1])[0].detach().cpu().tolist(),
         'selected_weights':value[1].reshape(-1,value[1].shape[-1])[0].detach().float().cpu().tolist()}
      return value
     layer.mlp.gate.forward=capture
    try:
     pos_emb=rotary(hidden,position_ids)
     hidden=layer(hidden,attention_mask=None,position_ids=position_ids,cache_position=cache_position,position_embeddings=pos_emb,use_cache=True,past_key_value=cache,output_router_logits=False)[0]
    finally:
     if original is not None: layer.mlp.gate.forward=original
     if post_handle is not None: post_handle.remove()
    if parsed_probe is not None and pos_i == parsed_probe[0] and ln == parsed_probe[1] and 'value' not in post_capture:
     raise ReferenceError('cached post-attention probe was not captured')
    capture=None
    if captured:
     ids=captured['ids'].reshape(-1,8).tolist()[0]; weights=captured['weights'].reshape(-1,8).tolist()[0]; route_rows[ln]['router_ids'].append(ids); route_rows[ln]['router_weights'].append(weights)
    next_hiddens.append(hidden.detach()); layer_rows[ln].append(hidden.detach().cpu().float().reshape(-1))
    del captured,original,capture
   token_hiddens=next_hiddens
   del cache,layer,state
   gc.collect()
   if a.device=='cuda': torch.cuda.empty_cache()
  norm=mod.BailingMoeV3RMSNorm(cfg.hidden_size,eps=cfg.rms_norm_eps).to(dev); norm.load_state_dict({'weight':idx.raw('model.norm.weight',torch,dev)})
  lm_head=idx.raw('lm_head.weight',torch,dev); finals=[]; logits_rows=[]
  for hidden in token_hiddens:
   final=norm(hidden); finals.append(final.detach().cpu().float().reshape(-1)); logits_rows.append(torch.matmul(final.float(),lm_head.transpose(0,1).float()).detach().cpu().float().reshape(-1))
  validate_cached_trace(layer_rows,route_rows,len(toks),cfg.num_hidden_layers)
  if parsed_probe is not None and route_probe_data[0] is None: raise ReferenceError('route probe position was not captured')
  hidden_out=torch.stack(finals).unsqueeze(0); logits=torch.stack(logits_rows).unsqueeze(0); hidden_layers=torch.stack([torch.stack(x) for x in layer_rows])
 if a.device=='cuda' and torch.cuda.max_memory_reserved()>5*1024**3: raise ReferenceError('CUDA peak reserved exceeds 5 GiB')
 pins2=idx.pin(); srcpins2=pin_source(root,source)
 if pins!=pins2 or srcpins!=srcpins2: raise ReferenceError('checkpoint/source pins changed')
 out.mkdir(parents=True); hidden_out.numpy().tofile(out/'final_hidden.fp32'); logits.numpy().tofile(out/'logits.fp32'); hidden_layers.numpy().tofile(out/'hidden_layers.fp32'); (out/'layers.json').write_text(json.dumps(route_rows)); (out/'router_meta.json').write_text(json.dumps([{'layer':x['layer'],'router_ids':bool(x['router_ids']),'tokens':len(x['router_ids']),'top_k':8 if x['router_ids'] else 0} for x in route_rows],indent=2))
 if route_probe_data[0] is not None: (out/'route-probe.json').write_text(json.dumps(route_probe_data[0],sort_keys=True)+chr(10))
 report=build_report(a,cfg,toks,hidden_out,logits,srcpins2,pins2,torch,out,started,'cached',{'cache_type':'DynamicCache','prefill_tokens':0,'decode_tokens':len(toks),'tokenwise':True,'layer_major':True})
 if parsed_probe is not None: report['route_probe']={'position':parsed_probe[0],'layer':parsed_probe[1],'output':'route-probe.json'}
 write_report(report,out); return report

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
 if getattr(a,'route_probe',None) is not None and getattr(a,'decoding_mode','full') != 'cached': raise ReferenceError('route probe currently requires cached decoding')
 if getattr(a,'decoding_mode','full')=='cached': return run_cached(a,root,source,out,torch,mod,cfg,idx,pins,srcpins,dev,started,toks)
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
 out.mkdir(parents=True); hidden.detach().cpu().float().numpy().tofile(out/'final_hidden.fp32'); logits.detach().cpu().float().numpy().tofile(out/'logits.fp32'); torch.stack(hs).numpy().tofile(out/'hidden_layers.fp32'); (out/'layers.json').write_text(json.dumps(layers)); router_meta=[{'layer':x['layer'],'router_ids':bool(x['router_ids']),'tokens':len(x['router_ids']),'top_k':8 if x['router_ids'] else 0} for x in layers]; (out/'router_meta.json').write_text(json.dumps(router_meta,indent=2)); report=build_report(a,cfg,toks,hidden,logits,srcpins2,pins2,torch,out,started,'full',{'cache_type':None,'prefill_tokens':len(toks),'chunk_mode':'source-selected'})
 write_report(report,out); return report
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--source',required=True); p.add_argument('--tokens',required=True); p.add_argument('--output',required=True); p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--logits-mode',choices=['all','last'],default='all'); p.add_argument('--decoding-mode',choices=['full','cached'],default='full'); p.add_argument('--route-probe',help='capture one cached POS,LAYER router state'); a=p.parse_args()
 try: print(json.dumps(run(a),indent=2))
 except (ReferenceError,OSError,ValueError,RuntimeError) as e: p.error(str(e))
if __name__=='__main__': main()
