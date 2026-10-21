import pytest
from completion import CompletionConfig, CompletionError, run_completion

def enc(s): return list(s.encode())
def dec(ids): return bytes(ids).decode()
def row(ids,text,eos,reason,lps=None): return {"ids":ids,"text":text,"eos":eos,"finish_reason":reason,"sampled_logprobs":lps or [-.1]*len(ids)}
def cb(rows,seen):
 def run(prompt_ids,*,max_new_tokens,adapter,stage):
  seen.append((tuple(prompt_ids),max_new_tokens,adapter,stage)); return rows[stage]
 return run

def test_ling_natural_eos_and_actions():
 seen=[]; r=run_completion([1],"A",cb({"completion":row([2],["x"][0],True,"eos",[-.2])},seen),enc,dec,CompletionConfig(model="ling",max_new_tokens=4,context_capacity=8))
 assert r.complete and r.sampled_ids==(2,) and r.sampled_logprobs==(-.2,) and r.transcript_ids==(1,2) and not r.inserted_ids
 assert seen==[((1,),4,"A","completion")]

def test_ling_length_is_incomplete():
 r=run_completion([1],"A",cb({"completion":row([120],"x",False,"length")},[]),enc,dec,CompletionConfig(model="ling",max_new_tokens=2,context_capacity=8))
 assert not r.complete

def test_ring_natural_close_uses_no_inserted_actions():
 seen=[]; think=enc("a</think>"); r=run_completion([1],"A",cb({"thinking":row(think,"a</think>",False,"length",[-1.0]*len(think)),"answer":row([ord("V")],"V",True,"eos",[-.4])},seen),enc,dec,CompletionConfig(model="ring",thinking_tokens=16,answer_tokens=3,context_capacity=40))
 assert r.complete and not r.forced_close and r.inserted_ids==() and r.sampled_ids==tuple(think+[ord("V")])
 assert seen[1][0]==(1,*think) and seen[1][1:]==(3,"A","answer")

def test_ring_forced_close_is_context_not_action():
 close="\n</think>\n"; seen=[]
 # byte decoder makes inserted delimiter exact; sampled transcript includes it.
 r=run_completion([1],"A",cb({"thinking":row([97],"a",False,"length",[-.2]),"answer":row([98],"b",True,"eos",[-.3])},seen),enc,dec,CompletionConfig(model="ring",thinking_tokens=2,answer_tokens=2,context_capacity=30,close_text=close))
 assert r.complete and r.forced_close and r.sampled_ids==(97,98) and r.inserted_ids==tuple(enc(close))
 assert seen[1][0]==(1,97,*enc(close))
 assert r.transcript_ids==(1,97,*enc(close),98)

def test_capacity_checked_before_native():
 calls=[]
 with pytest.raises(CompletionError): run_completion([1,2],"A",cb({},calls),enc,dec,CompletionConfig(model="ring",thinking_tokens=5,answer_tokens=2,context_capacity=6))
 assert calls==[]

def test_bad_logprob_or_delimiter_rejected():
 with pytest.raises((TypeError,CompletionError)):
  run_completion([1],"A",cb({"completion":{"ids":[2],"text":"x","eos":True,"finish_reason":"eos","sampled_logprobs":[]}},[]),enc,dec,CompletionConfig(model="ling",max_new_tokens=2))
 malformed=run_completion([1],"A",cb({"completion":row([2],"<think>x",True,"eos")},[]),enc,dec,CompletionConfig(model="ling",max_new_tokens=2))
 assert malformed.complete and malformed.format_error and malformed.sampled_ids==(2,) and malformed.action_positions==(1,)

def test_adapter_error_propagates():
 def bad(*a,**k): raise RuntimeError("binding")
 with pytest.raises(RuntimeError): run_completion([1],"A",bad,enc,dec,CompletionConfig(model="ling",max_new_tokens=2))


def test_forced_and_natural_action_positions_skip_inserted_ids():
 close="\n</think>\n"; seen=[]
 r=run_completion([10],"A",cb({"thinking":row([97],"a",False,"length",[-.2]),"answer":row([98],"b",True,"eos",[-.3])},seen),enc,dec,CompletionConfig(model="ring",thinking_tokens=2,answer_tokens=2,context_capacity=30,close_text=close))
 assert r.action_positions==(1,2+len(enc(close)))
 assert r.stages[0].action_positions==(1,) and r.stages[1].action_positions==(2+len(enc(close)),)
 assert r.inserted_ids==tuple(enc(close))

def test_malformed_ring_stage_is_retained_without_artificial_close():
 seen=[]
 r=run_completion([1],"A",cb({"thinking":row([97],"a<|bad|>",False,"length",[-.2])},seen),enc,dec,CompletionConfig(model="ring",thinking_tokens=2,answer_tokens=2,context_capacity=30))
 assert not r.complete and r.format_error and r.sampled_ids==(97,) and r.inserted_ids==()
 assert len(seen)==1

def test_ring_eos_without_close_is_retained_no_inserted_context():
 r=run_completion([1],"A",cb({"thinking":row([97],"a",True,"eos",[-.2])},[]),enc,dec,CompletionConfig(model="ring",thinking_tokens=2,answer_tokens=2,context_capacity=30))
 assert r.complete and r.format_error=="missing closing thinking delimiter" and not r.inserted_ids


def test_bad_eos_contract_is_structural_failure():
 with pytest.raises(TypeError):
  run_completion([1],"A",cb({"completion":row([2],"x",True,"length")},[]),enc,dec,CompletionConfig(model="ling",max_new_tokens=2))
