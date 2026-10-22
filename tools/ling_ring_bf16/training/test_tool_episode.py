"""Contract tests: sampled corruption aborts; model failures remain scored failures."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import json
import numpy as np
import pytest
from tool_episode import ToolEpisodeConfig, EpisodeError, run_episode
EOS=9999
def enc(s):return list(s.encode())
def dec(ids):return bytes([x for x in ids if x!=EOS]).decode()
def out(prompt,text,eos=True,*,model="ling",forced=False):
    answer=enc(text);tail=answer+([EOS] if eos else [])
    if model=="ring":
        thinking=enc("checking \n")
        close=enc("\n</think>\n")
        full=list(prompt)+thinking+close+tail
        actions=thinking+tail
        pos=list(range(len(prompt),len(prompt)+len(thinking)))+list(range(len(prompt)+len(thinking)+len(close),len(full)))
        inserted=close
        raw=dec(thinking+close+answer)
    else:
        actions=tail;full=list(prompt)+tail;pos=list(range(len(prompt),len(full)));inserted=[];raw=text
    return {"ids":actions,"sampled_logprobs":[-.5]*len(actions),"eos":eos,"finish_reason":"eos" if eos else "length",
        "transcript_ids":full,"action_positions":pos,"inserted_ids":inserted,"raw_text":raw,"text":text,
        "features":np.ones((len(actions),3)),"logits":np.ones((len(actions),4))}
def config(**kwargs):
    return ToolEpisodeConfig(eos_token_ids=(EOS,),context_capacity=2048,model_max_new_tokens=224,**kwargs)
def frame(ids,**kw):return tuple(ids)+tuple(enc("\nOBS "+json.dumps(kw["result"])+"\nASSISTANT "))
def run(sample,tools=lambda n,a:{},grader=lambda v,c:True,**kw):
    return run_episode([1],sample,tools,grader,enc,dec,config=kw.pop("config",config()),frame_observation=frame,**kw)
def test_ling_success_and_delivered_actions():
    calls=[];seen=[]
    def sample(p,**kw):
        seen.append(p)
        return out(p,'{"tool":"ls","args":{}}' if len(seen)==1 else '{"submit":"ok"}')
    r=run(sample,lambda n,a:(calls.append((n,a)) or {"files":["x"]}))
    assert r.success and r.complete and r.tool_calls==r.tool_attempts==1
    assert len(r.delivered_tool_result_hashes)==1
    assert tuple(r.transcript_ids[p] for p in r.action_positions)==r.action_ids
    assert len(r.action_features)==len(r.action_ids)<len(r.transcript_ids)
def test_ring_middle_inserted_close_and_token_value_overlap():
    s=lambda p,**kw:out(p,'{"submit":{"text":"a b"}}',model="ring",forced=True)
    r=run(s,config=config(model="ring",require_capture=True))
    assert r.success
    seg=r.segments[0]
    assert set(seg.inserted_ids)&set(seg.sampled_ids) # Values can match; positions must not.
    assert tuple(r.transcript_ids[p] for p in r.action_positions)==r.action_ids
    assert len(r.transcript_ids)-1==len(r.action_ids)+len(seg.inserted_ids)
def test_duplicate_json_failure_retained_and_charged():
    count=0
    def sample(p,**kw):
        nonlocal count
        count+=1
        return out(p,'{"tool":"ls","tool":"grep","args":{}}' if count==1 else '{"submit":"bad"}')
    def forbidden(*args):raise AssertionError("malformed action executed")
    r=run(sample,forbidden,lambda v,c:False)
    assert r.complete and not r.success and r.reward==0 and r.tool_attempts==1 and r.tool_calls==0 and len(r.segments)==2
def test_length_json_never_executes_or_grades():
    def forbidden(*args):raise AssertionError("incomplete executed")
    r=run(lambda p,**kw:out(p,'{"submit":"ok"}',eos=False),forbidden,forbidden)
    assert not r.complete and not r.success and r.tool_attempts==1 and r.action_ids and "incomplete_generation" in r.errors
@pytest.mark.parametrize("change",["raw","position","logs","eos","nan"])
def test_sampler_corruption_aborts(change):
    def sample(p,**kw):
        x=out(p,'{"submit":"ok"}')
        if change=="raw":x["raw_text"]="forged"
        if change=="position":x["action_positions"][0]=0
        if change=="logs":x["sampled_logprobs"][0]=float("nan")
        if change=="eos":x["ids"][-1]=2
        if change=="nan":x["features"][0,0]=float("nan")
        return x
    with pytest.raises(EpisodeError):run(sample)
def test_dropped_observation_has_no_delivered_hash():
    c=ToolEpisodeConfig(eos_token_ids=(EOS,),context_capacity=128,model_max_new_tokens=64)
    r=run(lambda p,**kw:out(p,'{"tool":"ls","args":{}}'),lambda n,a:{"output":"X"*120},config=c)
    assert not r.complete and r.tool_calls==1 and not r.delivered_tool_result_hashes and "context_overflow" in r.errors
def test_bad_verifier_cannot_return_positive_failure():
    with pytest.raises(EpisodeError,match="verifier"):
        run(lambda p,**kw:out(p,'{"submit":"x"}'),grader=lambda v,c:{"success":False,"reward":1.})
def test_unknown_tool_is_charged_not_executed():
    count=0
    def s(p,**kw):
        nonlocal count
        count+=1
        return out(p,'{"tool":"shell","args":{}}' if count==1 else '{"submit":"x"}')
    r=run(s,grader=lambda v,c:False)
    assert r.tool_attempts==1 and r.tool_calls==0 and not r.success
def test_output_overflow_is_bounded_marker():
    r=run(lambda p,**kw:out(p,'{"tool":"ls","args":{}}'),lambda n,a:{"output":"X"*10000},config=config(max_turns=1,max_result_bytes=64))
    assert r.output_limited and not r.success and len(r.delivered_tool_result_hashes)==1
