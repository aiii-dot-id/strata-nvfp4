"""Strict CPU rehydration of sealed tool-workflow rollout captures.
AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations
import hashlib,json,os,math
from pathlib import Path
import numpy as np
import torch

MODEL_GEOMETRY={'ling':(1536,157184),'ring':(2048,157184)}
class ArchiveError(ValueError): pass

def _sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def _json(path):
 try:
  with Path(path).open(encoding='utf-8') as f:return json.load(f)
 except Exception as e: raise ArchiveError(f'invalid JSON: {path}') from e
def _rel(root,value):
 p=Path(value)
 # Reject symlink components before canonicalization; resolve() alone can hide them.
 raw=root/p if not p.is_absolute() else p
 cur=Path(raw.anchor or "/")
 for part in raw.parts[1:] if raw.is_absolute() else raw.parts:
  cur=cur/part
  if cur.is_symlink(): raise ArchiveError("symlink path refused")
 if p.is_absolute():
  try:p=p.resolve().relative_to(root.resolve())
  except ValueError:raise ArchiveError('path escapes sealed root')
 else:p=Path(os.path.normpath(str(p)))
 if str(p)=='.' or str(p).startswith('../') or p.is_absolute():raise ArchiveError('path escapes sealed root')
 cur=root
 for part in p.parts:
  cur=cur/part
  if cur.is_symlink():raise ArchiveError('symlink path refused')
 q=(root/p).resolve()
 try:q.relative_to(root.resolve())
 except ValueError:raise ArchiveError('path escapes sealed root')
 return q,p.as_posix()
def _manifest(root,mapping):
 if not isinstance(mapping,dict) or not mapping:raise ArchiveError('sealed manifest required')
 out={}
 for key,val in mapping.items():
  if not isinstance(key,str) or Path(key).is_absolute() or '..' in Path(key).parts:raise ArchiveError('invalid manifest path')
  p,rel=_rel(root,key)
  if not isinstance(val,(str,dict)):raise ArchiveError('invalid manifest entry')
  sha=val if isinstance(val,str) else val.get('sha256'); size=None if isinstance(val,str) else val.get('size')
  if not isinstance(sha,str) or len(sha)!=64:raise ArchiveError('manifest hash')
  if not p.is_file() or p.is_symlink():raise ArchiveError('manifest file missing')
  if size is not None and (type(size)is not int or p.stat().st_size!=size):raise ArchiveError('manifest size mismatch')
  if _sha(p)!=sha:raise ArchiveError('manifest hash mismatch')
  out[rel]=(p,sha)
 return out
def _need(root,man,value):
 p,rel=_rel(root,value)
 if rel not in man:raise ArchiveError('referenced file absent from sealed manifest: '+rel)
 return p,rel
def _field(d,*keys):
 for k in keys:
  if isinstance(d,dict) and k in d:return d[k]
 return None
def _stage_prompt(run):
 e=run.get('execution',{}) if isinstance(run,dict) else {}
 g=run.get('generation',{}) if isinstance(run,dict) else {}
 ids=g.get('prompt_token_ids')
 if not isinstance(ids,list) or not ids or any(type(x)is not int or x<0 for x in ids): raise ArchiveError('native prompt token IDs missing')
 return len(ids)
def _stage_generation(run):
 g=run.get('generation') if isinstance(run,dict) else None
 if not isinstance(g,dict):raise ArchiveError('native generation missing')
 ids=g.get('generated_token_ids')
 sampling=g.get('sampling')
 lps=sampling.get('sampled_logprobs') if isinstance(sampling,dict) else None
 if not isinstance(ids,list) or not isinstance(lps,list) or len(ids)!=len(lps) or not ids:raise ArchiveError('native generation IDs/logprobs')
 if any(type(x)is not int or x<0 for x in ids) or any(type(x) not in (int,float) or not math.isfinite(float(x)) for x in lps):raise ArchiveError('native generation values')
 return tuple(ids),tuple(float(x) for x in lps)
def load_rollout(root, manifest, *, model=None):
 raw_root=Path(root).absolute()
 cur=Path(raw_root.anchor)
 for part in raw_root.parts[1:]:
  cur=cur/part
  if cur.is_symlink():raise ArchiveError("symlink root refused")
 root=raw_root.resolve()
 if not root.is_dir():raise ArchiveError('sealed episode root required')
 man=_manifest(root,manifest)
 rp,rr=_need(root,man,'ROLLOUT.json'); rec=_json(rp)
 if not isinstance(rec,dict) or rec.get('schema')!='strata-identity-tool-rollout-v1':raise ArchiveError('rollout schema')
 mdl=model or rec.get('model')
 if mdl not in MODEL_GEOMETRY or rec.get('model')!=mdl:raise ArchiveError('model')
 hidden,vocab=MODEL_GEOMETRY[mdl]
 if type(hidden)is not int or type(vocab)is not int:raise ArchiveError('geometry')
 stages=rec.get('native_stages')
 if not isinstance(stages,list) or not stages:raise ArchiveError('native stages')
 all_ids=[];all_lps=[];all_pos=[];feature_parts=[];logit_parts=[];groups=[];seen_stages=set();prev_stage=-1
 for si,entry in enumerate(stages):
  if not isinstance(entry,dict) or not isinstance(entry.get('path'),str):raise ArchiveError('stage record')
  sp,srel=_rel(root,entry['path'])
  if not sp.is_dir(): raise ArchiveError('stage directory')
  if srel in seen_stages: raise ArchiveError('duplicate or out-of-order stage')
  seen_stages.add(srel)
  runp,runrel=_need(root,man,sp/'RUN.json'); run=_json(runp)
  if run.get('model')!=mdl:raise ArchiveError('native model mismatch')
  if run.get('passed') is not True or run.get('source_dirty') is not False or not isinstance(run.get('source_state_before'),dict) or run['source_state_before'].get('dirty') is not False:raise ArchiveError('native run not qualified')
  ids,lps=_stage_generation(run); prompt_len=_stage_prompt(run)
  transcript=rec.get('transcript_ids')
  if not isinstance(transcript,list) or transcript[:prompt_len]!=run['generation']['prompt_token_ids']:raise ArchiveError('native prompt transcript binding')
  if tuple(transcript[prompt_len:prompt_len+len(ids)])!=ids:raise ArchiveError('native action transcript binding')
  retp,retrel=_need(root,man,sp/'RETENTION.json'); retention=_json(retp)
  retained=retention.get('retained') if isinstance(retention,dict) else None
  if not isinstance(retained,dict):raise ArchiveError('retention metadata')
  term=retained.get('selected-terminal.f32'); logm=retained.get('selected-logits.f32')
  if not isinstance(term,dict) or not isinstance(logm,dict):raise ArchiveError('selected arrays missing')
  tp,trel=_need(root,man,sp/'selected-terminal.f32'); lp,lrel=_need(root,man,sp/'selected-logits.f32')
  for meta,rel in ((term,trel),(logm,lrel)):
   if not isinstance(meta.get('sha256'),str) or meta.get('sha256')!=man[rel][1]: raise ArchiveError('selected array hash binding')
   if type(meta.get('bytes')) is not int or meta.get('bytes')!=Path(_rel(root,rel)[0]).stat().st_size: raise ArchiveError('selected array size binding')
  n=len(ids); expected_feat=n*(hidden*11+8208); expected_log=n*vocab
  if tp.stat().st_size!=expected_feat*4 or lp.stat().st_size!=expected_log*4:raise ArchiveError('selected array geometry')
  f=np.fromfile(tp,dtype='<f4').reshape(n,hidden*11+8208); l=np.fromfile(lp,dtype='<f4').reshape(n,vocab)
  if not np.isfinite(f).all() or not np.isfinite(l).all():raise ArchiveError('nonfinite selected arrays')
  positions=term.get('positions'); lpositions=logm.get('positions')
  if positions!=lpositions or not isinstance(positions,list) or positions!=list(range(prompt_len-1,prompt_len-1+n)):raise ArchiveError('selected positions')
  if not isinstance(entry.get('run_sha256'),str) or entry['run_sha256']!=man[runrel][1]:raise ArchiveError('RUN hash binding')
  if not isinstance(entry.get('retention_sha256'),str) or entry['retention_sha256']!=man[retrel][1]:raise ArchiveError('retention hash binding')
  all_ids.extend(ids);all_lps.extend(lps);all_pos.extend([x+1 for x in positions]);feature_parts.append(f);logit_parts.append(l);groups.extend([entry.get('stage',si)]*n)
 rid=rec.get('ids'); rlp=rec.get('sampling_logprobs'); rpos=rec.get('action_positions')
 if tuple(rid or ())!=tuple(all_ids) or tuple(float(x) for x in (rlp or ()))!=tuple(all_lps) or tuple(rpos or ())!=tuple(all_pos):raise ArchiveError('ROLLOUT action binding')
 if any(all_pos[i]>=all_pos[i+1] for i in range(len(all_pos)-1)):raise ArchiveError('global action position order')
 if tuple(rec['transcript_ids'][i] for i in all_pos)!=tuple(all_ids):raise ArchiveError('transcript action binding')
 features=torch.from_numpy(np.concatenate(feature_parts,axis=0).copy())
 logits=np.concatenate(logit_parts,axis=0).copy()
 return {'features':features,'logits':logits,'ids':tuple(all_ids),'logprobs':tuple(all_lps),'sampling_logprobs':tuple(all_lps),'action_positions':tuple(all_pos),'path':root,'group':rec.get('group',rec.get('task')),'model':mdl,'stage_groups':tuple(groups),'rollout':rec,**{k:rec[k] for k in ('task','split','family','seed','grade','success','complete','policy_adapter_sha256','native_stages') if k in rec}}
