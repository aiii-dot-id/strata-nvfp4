#!/usr/bin/env python3
"""AI Identity Incorported <james@aiii.id>. CPU generation alignment only."""
import argparse, hashlib, json, mmap, pathlib
import numpy as np
PROD_VOCAB=157184
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def fail(ok,msg):
 if not ok: raise ValueError(msg)
def load(path): return json.loads(pathlib.Path(path).read_text())
def top2(row):
 first=int(np.argmax(row)); v=float(row[first]); tmp=row.copy(); tmp[first]=-np.inf; second=int(np.argmax(tmp)); return first,second,v,float(row[second])
def readout(artifact, *, vocab_domain=None, expected_provenance_sha256=None, thinking_end_id=None):
 vocab=PROD_VOCAB if vocab_domain is None else vocab_domain; fail(type(vocab) is int and not isinstance(vocab,bool) and vocab>=2,"invalid vocabulary domain")
 root=pathlib.Path(artifact); pth=root/"provenance.json"; exep=root/"execution.json"; logits=root/"logits.f32"; fail(pth.is_file() and exep.is_file(),"missing provenance/execution")
 psha0,esha0=sha(pth),sha(exep); fail(logits.is_file() and not logits.is_symlink(),"logits file missing or linked"); lsha0,lsize0=sha(logits),logits.stat().st_size; p,e=load(pth),load(exep)
 fail(p.get("status")=="verified","provenance status is not verified"); fail(p.get("execution")==e,"provenance execution binding mismatch")
 if expected_provenance_sha256 is not None: fail(psha0==expected_provenance_sha256,"provenance hash mismatch")
 start=p.get("continuation",{}).get("start_position",0); fail(type(start) is int and not isinstance(start,bool) and start==0,"suffix/resumed trace is not admitted in v1")
 gen=p.get("generation"); fail(isinstance(gen,dict),"missing generation receipt"); P,G,eos=gen.get("prompt_token_ids"),gen.get("generated_token_ids"),gen.get("eos_token_ids"); valid=lambda xs:isinstance(xs,list) and bool(xs) and all(type(x) is int and not isinstance(x,bool) and 0<=x<vocab for x in xs)
 fail(valid(P),"invalid prompt ids"); fail(valid(G),"invalid generated ids"); fail(valid(eos),"invalid EOS ids"); mx=gen.get("max_new_tokens"); fail(type(mx) is int and not isinstance(mx,bool) and mx>0,"invalid max_new_tokens"); fail(len(G)<=mx,"generated sequence exceeds max_new_tokens")
 reason=gen.get("finish_reason"); fail(reason in ("eos","length"),"unfinished generation"); fail(not any(x in eos for x in G[:-1]),"interior EOS token"); fail(G[-1] in eos if reason=="eos" else len(G)==mx and G[-1] not in eos,"invalid termination")
 rows=e.get("logits_tokens"); fail(type(rows) is int and not isinstance(rows,bool) and rows==len(P)+len(G)-1,"logit row accounting mismatch"); fail(e.get("tokens")==rows,"execution token count mismatch"); fail(e.get("token_ids")==P+G[:-1],"execution token binding mismatch")
 expected=rows*vocab*4; fail(lsize0==expected,"logits shape/size mismatch"); md=p.get("native_outputs",{}).get("logits.f32"); fail(isinstance(md,dict) and md.get("size")==expected and md.get("sha256")==lsha0,"logits receipt hash mismatch")
 with logits.open("rb") as f:
  mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
  try:
   arr=np.frombuffer(mm,dtype="<f4",count=rows*vocab); fail(bool(np.isfinite(arr).all()),"nonfinite logits"); arr=arr.reshape(rows,vocab); out=[]
   for k,gid in enumerate(G):
    row=arr[len(P)-1+k]; a,b,av,bv=top2(row); fail(a==gid,f"argmax mismatch at generated index {k}: got {a}, expected {gid}"); item={"k":k,"absolute_position":len(P)-1+k,"current_input_token_id":P[-1] if k==0 else G[k-1],"generated_next_token_id":gid,"top2_ids":[a,b],"top2_values":[av,bv],"margin":av-bv,"eos":[{"id":int(x),"logit":float(row[x]),"rank":int(1+np.sum(row>row[x]))} for x in eos]}
    if thinking_end_id is not None:
     fail(type(thinking_end_id) is int and not isinstance(thinking_end_id,bool) and 0<=thinking_end_id<vocab,"invalid thinking end id"); item["thinking_end"]={"id":thinking_end_id,"logit":float(row[thinking_end_id]),"rank":int(1+np.sum(row>row[thinking_end_id]))}
    out.append(item)
  finally:
   row=None; arr=None; mm.close()
 psha1,esha1,lsha1,lsize1=sha(pth),sha(exep),sha(logits),logits.stat().st_size; fail(psha1==psha0 and esha1==esha0 and lsha1==lsha0 and lsize1==lsize0,"artifact changed during readout")
 return {"schema":"strata-generation-readout-v1","artifact":str(root),"provenance_sha256":psha0,"execution_sha256":esha0,"logits_sha256":lsha0,"model":p.get("model") or e.get("model") or p.get("model_id"),"vocab_domain":vocab,"prompt_tokens":len(P),"generated_tokens":len(G),"finish_reason":reason,"eos_token_ids":eos,"rows":out}
def main():
 a=argparse.ArgumentParser(); a.add_argument("artifact",type=pathlib.Path); a.add_argument("--expected-provenance-sha256"); a.add_argument("--thinking-end-id",type=int); a.add_argument("--output",type=pathlib.Path,required=True); x=a.parse_args(); fail(not x.output.resolve().is_relative_to(x.artifact.resolve()),"report must be outside the input artifact"); result=readout(x.artifact,expected_provenance_sha256=x.expected_provenance_sha256,thinking_end_id=x.thinking_end_id);
 with x.output.open("x") as f: f.write(json.dumps(result,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
