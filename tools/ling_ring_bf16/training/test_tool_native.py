import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import tool_native

def fake_sampler(tmp_path,model):
    H=tool_native.native.MODELS[model]["hidden"]; calls=[]; releases=[]
    close="\n</think>\n"
    def encode(s): return [7,8] if s==close else [1]
    def decode(ids):
        ids=list(ids)
        if ids==[7,8]: return close
        if ids==[3,4]: return "think"
        if ids==[5,9]: return "answer"
        if ids==[3,4,7,8,5]: return "think\n</think>\nanswer"
        return "".join(str(x) for x in ids)
    def stage(prompt,*,max_new_tokens,adapter,stage):
        calls.append((stage,tuple(prompt),max_new_tokens,adapter))
        ids=[4,9] if model=="ling" else ([3,4] if stage=="thinking" else [5,9])
        if model=="ling": ids=[4,9]
        rows=len(prompt)+len(ids)-1;H2=11*H+8208
        return {"ids":ids,"sampled_logprobs":[-0.1]*len(ids),"text":decode(ids),
                "eos":True if model=="ling" or stage=="answer" else False,
                "finish_reason":"eos" if model=="ling" or stage=="answer" else "length",
                "prompt_ids":list(prompt),"logits":np.zeros((rows,157184),np.float32),
                "features":np.zeros((rows,H2),np.float32),"directory":None}
    s=tool_native.NativeToolSampler(model,encode=encode,decode=decode,output_dir=tmp_path,
        eos_ids=(9,),stage_runner=stage,closure_check=lambda: calls.append(("check",)),
        release=lambda: releases.append(1),thinking_tokens=4,answer_tokens=4,max_new_tokens=4)
    return s,calls,releases

def test_ling_alignment_and_release(tmp_path):
    s,c,r=fake_sampler(tmp_path,"ling");x=s.sample([1,2,3],adapter="a")
    assert x.raw_text=="4" and x.final_text=="4" and x.sampled_ids==(4,9)
    assert x.action_positions==(3,4) and [z["position"] for z in x.rows]==[2,3]
    assert len(r)==2 and c[0][0]=="check"

def test_ring_close_excluded_from_actions_and_full_prefix(tmp_path):
    s,c,r=fake_sampler(tmp_path,"ring");x=s.sample([1,2],adapter=None)
    assert x.inserted_ids==(7,8) and x.forced_close and x.final_text=="answer"
    assert x.sampled_ids==(3,4,5,9) and len(x.rows)==4
    assert x.action_positions==(2,3,6,7)
    assert [z["stage"] for z in x.rows]==["thinking","thinking","answer","answer"]
    assert len(r)==4 and [z[0] for z in c if z[0]=="check"]==["check","check"]

def test_nonfinite_capture_rejected(tmp_path):
    s,c,r=fake_sampler(tmp_path,"ling")
    old=s.stage_runner
    def bad(*a,**kw):
        x=old(*a,**kw);x["features"][0,0]=np.nan;return x
    s.stage_runner=bad
    try:s.sample([1,2,3])
    except ValueError as e:assert "nonfinite" in str(e)
    else:raise AssertionError("nonfinite capture accepted")

def test_default_native_runner_lifecycle_and_retention(tmp_path,monkeypatch):
    import json
    s,c,releases=fake_sampler(tmp_path,"ling")
    calls=[]
    def mocked_run(model,prompt,d,**kw):
        assert not Path(d).exists()
        assert kw["executable"]=="/qualified/strata_ling"
        assert kw["checkpoint_binding"]=="c"*64
        Path(d).mkdir()
        n=len(prompt)+1; H=tool_native.native.MODELS[model]["hidden"]; rf=11*H+8208
        np.zeros((n,rf),np.float32).tofile(Path(d)/"terminal.f32")
        np.zeros((n,157184),np.float32).tofile(Path(d)/"logits.f32")
        (Path(d)/"terminal-meta.json").write_text(json.dumps({"schema":"strata-terminal-v1","model":model,"hidden":H,"rows":n,"rowfloats":rf,"intermediate":512,"topk":8}))
        (Path(d)/"execution.json").write_text(json.dumps({"tokens":n}))
        calls.append((d,kw))
        return {"generation":{"generated_token_ids":[4,9],"finish_reason":"eos","sampling":{"sampled_logprobs":[-.1,-.1]}}}
    s.stage_runner=None;s.executable="/qualified/strata_ling";s.checkpoint_binding="c"*64;s.stage_runner=s._native_stage
    monkeypatch.setattr(tool_native.native,"run",mocked_run)
    x=s.sample([1,2,3],seed=4)
    assert x.sampled_ids==(4,9) and len(calls)==1
    d=Path(calls[0][0]);assert (d/"RETENTION.json").exists() and not (d/"terminal.f32").exists() and not (d/"logits.f32").exists()

def test_teacher_forced_default_path(tmp_path,monkeypatch):
    import json
    s,c,r=fake_sampler(tmp_path,"ling");seen=[]
    def mocked(model,prompt,d,**kw):
        assert kw["generate"]==0 and not Path(d).exists()
        Path(d).mkdir(); n=len(prompt); H=tool_native.native.MODELS[model]["hidden"];rf=11*H+8208
        np.zeros((n,rf),np.float32).tofile(Path(d)/"terminal.f32");np.zeros((n,157184),np.float32).tofile(Path(d)/"logits.f32")
        (Path(d)/"terminal-meta.json").write_text(json.dumps({"schema":"strata-terminal-v1","model":model,"hidden":H,"rows":n,"rowfloats":rf,"intermediate":512,"topk":8}))
        (Path(d)/"execution.json").write_text(json.dumps({"tokens":n,"token_ids":list(prompt)}));seen.append(d)
        return {"execution":{"token_ids":list(prompt)}}
    s.stage_runner=None;s.executable="/qualified/strata_ling";s.checkpoint_binding="d"*64;s.stage_runner=s._native_stage
    monkeypatch.setattr(tool_native.native,"run",mocked)
    x=s.teacher_forced([1,2,3],seed=9)
    assert x["prompt_ids"]==(1,2,3) and x["features"].shape[0]==3 and len(seen)==1
