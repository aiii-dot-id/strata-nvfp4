"""Native structured tool-episode sampler; orchestration only on import.
AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import numpy as np
try:
 from . import completion,native,terminal
except ImportError:
 import completion,native,terminal

@dataclass(frozen=True)
class ToolSample:
 raw_text:str; final_text:str; transcript_ids:tuple[int,...]; sampled_ids:tuple[int,...]
 sampled_logprobs:tuple[float,...]; action_positions:tuple[int,...]; inserted_ids:tuple[int,...]
 forced_close:bool; complete:bool; format_error:str|None
 rows:tuple[dict[str,Any],...]; stages:tuple[dict[str,Any],...]

def _sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
def _finite(a,name):
 x=np.asarray(a)
 if not np.isfinite(x).all(): raise ValueError("nonfinite "+name)
 return x

class NativeToolSampler:
 def __init__(self,model,*,encode,decode,output_dir,eos_ids,executable=None,checkpoint_binding=None,
              stage_runner=None,closure_check=None,release=None,max_new_tokens=32,thinking_tokens=128,
              answer_tokens=96,context_capacity=4096,temperature=1.0,timeout=300):
  if model not in ("ling","ring"): raise ValueError("model")
  if isinstance(temperature,bool) or not math.isfinite(float(temperature)) or temperature<=0: raise ValueError("temperature")
  if stage_runner is None and executable is None: raise ValueError("qualified executable required")
  self.model=model;self.encode=encode;self.decode=decode;self.output_dir=Path(output_dir)
  self.eos_ids=tuple(eos_ids);self.executable=executable;self.checkpoint_binding=checkpoint_binding
  self.stage_runner=stage_runner or self._native_stage;self.closure_check=closure_check or (lambda:None)
  self.release=release or (lambda:None);self.timeout=timeout
  self.config=completion.CompletionConfig(model=model,max_new_tokens=max_new_tokens,thinking_tokens=thinking_tokens,
      answer_tokens=answer_tokens,context_capacity=context_capacity,eos_token_ids=self.eos_ids)
  self.temperature=float(temperature);self._counter=0

 def _new_stage_dir(self,stage):
  self.output_dir.mkdir(parents=True,exist_ok=True);self._counter+=1
  d=self.output_dir/f"{self._counter:02d}-{stage}"
  if d.exists(): raise FileExistsError(d)
  return d

 def _native_stage(self,prompt,*,max_new_tokens,adapter,stage):
  d=self._new_stage_dir(stage)
  rec=native.run(self.model,prompt,d,executable=self.executable,adapter=adapter,capture=True,
      generate=max_new_tokens,eos=self.eos_ids,timeout=self.timeout,checkpoint_binding=self.checkpoint_binding,
      sampling_temperature=self.temperature,sampling_seed=self._seed+self._counter-1)
  gen=rec.get("generation")
  if not isinstance(gen,dict): raise ValueError("native generation receipt missing")
  return self._read_stage(d,prompt,gen)

 def _read_stage(self,directory,prompt,generation):
  d=Path(directory);execution=json.loads((d/"execution.json").read_text());n=execution["tokens"]
  ids=tuple(generation["generated_token_ids"]);eos=generation["finish_reason"]=="eos"
  lps=tuple(float(x) for x in generation["sampling"]["sampled_logprobs"])
  cap=terminal.load_features(d);features=cap.rows.detach().cpu().numpy()
  logits=np.memmap(d/"logits.f32",dtype="<f4",mode="r",shape=(n,157184))
  visible=ids[:-1] if eos else ids
  return {"ids":list(ids),"sampled_logprobs":list(lps),"text":self.decode(visible),
      "eos":eos,"finish_reason":generation["finish_reason"],"prompt_ids":list(prompt),
      "logits":np.asarray(logits),"features":features,"directory":str(d)}

 def _compact(self,stage,payload,inserted):
  prompt=tuple(payload["prompt_ids"]);ids=tuple(payload["ids"]);lps=tuple(float(x) for x in payload["sampled_logprobs"])
  logits=_finite(payload["logits"],"logits");features=_finite(payload["features"],"terminal")
  if len(ids)!=len(lps) or not ids or logits.ndim!=2 or features.ndim!=2: raise ValueError("stage payload shape")
  if logits.shape[0]!=len(prompt)+len(ids)-1 or logits.shape[1]!=157184: raise ValueError("logit row geometry")
  expected=11*native.MODELS[self.model]["hidden"]+8208
  if features.shape!=(logits.shape[0],expected): raise ValueError("terminal feature geometry")
  selected=[]
  for i,token in enumerate(ids):
   row=len(prompt)-1+i
   selected.append({"features":features[row].copy(),"logits":logits[row].copy(),"id":token,
      "logprob":lps[i],"stage":stage,"position":row})
  directory=payload.get("directory")
  if directory:
   d=Path(directory);retained={};removed={}
   for kind,key in (("terminal","features"),("logits","logits")):
    src=d/(kind+".f32"); arr=np.asarray([x[key] for x in selected],dtype="<f4")
    q=d/("selected-"+kind+".f32"); q.write_bytes(arr.tobytes())
    retained[q.name]={"sha256":_sha(q),"bytes":q.stat().st_size,"positions":[x["position"] for x in selected]}
    if src.exists():
     removed[src.name]={"sha256":_sha(src),"bytes":src.stat().st_size};src.unlink()
    for x in selected:x[kind+"_sha256"]=retained[q.name]["sha256"]
   (d/"RETENTION.json").write_text(json.dumps({"retained":retained,"removed_validated_transient_floats":removed},sort_keys=True,indent=2)+"\n")
  return selected

 def sample(self,prompt_ids,adapter=None,*,seed=0):
  self._seed=seed;self.output_dir.mkdir(parents=True,exist_ok=True);records=[]
  def run_stage(prompt,max_new_tokens,adapter,stage):
   self.closure_check();self.release()
   inserted=() if stage!="answer" else tuple(self.encode(self.config.close_text))
   try:
    out=self.stage_runner(tuple(prompt),max_new_tokens=max_new_tokens,adapter=adapter,stage=stage)
    rows=self._compact(stage,out,inserted);records.append((stage,rows));return out
   finally:self.release()
  result=completion.run_completion(tuple(prompt_ids),adapter,run_stage,self.encode,self.decode,self.config)
  flat=[x for _,rs in records for x in rs]
  if len(flat)!=len(result.sampled_ids): raise ValueError("sampled/action row count mismatch")
  final=result.text
  if self.model=="ring" and "</think>" in final: final=final.split("</think>",1)[1].strip()
  return ToolSample(result.text,final,result.transcript_ids,result.sampled_ids,result.sampled_logprobs,
      result.action_positions,result.inserted_ids,result.forced_close,result.complete,result.format_error,tuple(flat),
      tuple({"stage":s,"rows":len(rs),"positions":[x["position"] for x in rs]} for s,rs in records))

 def teacher_forced(self,prompt_ids,adapter=None,*,seed=0):
  self.closure_check();self.release();self._seed=seed;d=self._new_stage_dir("teacher")
  try:
   rec=native.run(self.model,prompt_ids,d,executable=self.executable,adapter=adapter,capture=True,
       generate=0,eos=self.eos_ids,timeout=self.timeout,checkpoint_binding=self.checkpoint_binding)
   execution=rec.get("execution") or json.loads((d/"execution.json").read_text())
   if execution.get("token_ids")!=list(prompt_ids): raise ValueError("teacher prefix mismatch")
   cap=terminal.load_features(d);features=cap.rows.detach().cpu().numpy();n=len(prompt_ids)
   logits=np.memmap(d/"logits.f32",dtype="<f4",mode="r",shape=(n,157184))
   _finite(features,"teacher terminal");_finite(logits,"teacher logits")
   return {"prompt_ids":tuple(prompt_ids),"features":features,"logits":np.asarray(logits),"directory":str(d)}
  finally:self.release()
