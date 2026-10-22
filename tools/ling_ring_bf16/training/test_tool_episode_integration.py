"""CPU integration: real tokenizers, production framing, canonical Go tools.
AI Identity Incorported <james@aiii.id>.
No model or GPU calls are made.
"""
import hashlib,json
from pathlib import Path
import pytest
from transformers import AutoTokenizer
from tool_episode import run_episode,ToolEpisodeConfig,EpisodeError
from tool_framing import ExactChatFramer
from tool_environment import CanonicalToolEnvironment,sha
from tool_workflow import SCHEMAS,PROTOCOL
from completion import CompletionConfig,run_completion

BRIDGE=Path("/work/AI/experiments/strata-capability-20261021-r2/tool-strategy-prep/aii-tool-bridge/aii-tool-bridge")
PIN="211f546da0b2518738fda47ed04698241d517e54c7e82504e762396113d76175"
@pytest.fixture(params=["ling","ring"])
def context(request,tmp_path):
    model=request.param
    cp="/work/AI/checkpoints/"+("Ling-3.0-tiny" if model=="ling" else "Ring-mini-2.0")
    tok=AutoTokenizer.from_pretrained(cp,local_files_only=True)
    encode=lambda s:tok.encode(s,add_special_tokens=False)
    decode=lambda ids:tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    root=tmp_path/"fixture";root.mkdir()
    (root/"a.txt").write_text("unrelated\nnoise\n")
    (root/"b.txt").write_text("first\nverified_target = 42\n")
    files=[{"relative_path":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(root.iterdir())]
    expected={"path":"b.txt","line":2,"text":"verified_target = 42"}
    env=CanonicalToolEnvironment(BRIDGE,PIN,root,files,expected)
    framer=ExactChatFramer(tok,model=model,schemas=SCHEMAS,context_capacity=4096,eos_token_ids=(tok.eos_token_id,))
    prompt=framer.start(PROTOCOL+"Find the exact line containing verified_target.").transcript_ids
    cfg=ToolEpisodeConfig(model=model,eos_token_ids=(tok.eos_token_id,),context_capacity=4096,model_max_new_tokens=229)
    def sample(prompt,text):
        def stage(ids,*,max_new_tokens,adapter,stage):
            if model=="ring" and stage=="thinking":
                generated=encode("Need evidence.")
                return {"ids":generated,"sampled_logprobs":[-.2]*len(generated),"text":decode(generated),"eos":False,"finish_reason":"length"}
            generated=encode(text)+[tok.eos_token_id]
            return {"ids":generated,"sampled_logprobs":[-.2]*len(generated),"text":decode(generated[:-1]),"eos":True,"finish_reason":"eos"}
        r=run_completion(prompt,None,stage,encode,decode,CompletionConfig(model=model,max_new_tokens=96,thinking_tokens=128,answer_tokens=96,context_capacity=4096,eos_token_ids=(tok.eos_token_id,)))
        return {"ids":r.sampled_ids,"sampled_logprobs":r.sampled_logprobs,"transcript_ids":r.transcript_ids,
                "action_positions":r.action_positions,"inserted_ids":r.inserted_ids,"raw_text":r.text,
                "text":r.text.split("</think>",1)[1].strip() if model=="ring" else r.text,
                "eos":r.complete,"finish_reason":"eos" if r.complete else "length","format_error":r.format_error}
    return model,tok,encode,decode,env,framer,prompt,cfg,sample

def test_real_tools_production_framing_and_action_positions(context):
    model,tok,enc,dec,env,framer,prompt,cfg,sample=context
    calls=[];samples=[]
    def sampling(p,**kw):
        calls.append(tuple(p))
        action={"tool":"grep","args":{"pattern":"verified_target","literal":True}} if len(calls)==1 else {"submit":env.expected}
        r=sample(p,json.dumps(action,separators=(",",":")));samples.append(r);return r
    r=run_episode(prompt,sampling,env,env.grade,enc,dec,config=cfg,
                  frame_observation=lambda ids,**kw:framer.append_tool(ids,**kw).transcript_ids)
    assert r.success and r.complete and r.reward==1.075
    assert r.tool_calls==1 and r.tool_attempts==1 and len(env.calls)==1
    assert r.delivered_tool_result_hashes==(env.calls[0]["result_sha256"],)
    assert str(env.root/"b.txt")+":2:verified_target = 42" in dec(calls[1])
    assert tuple(r.transcript_ids[p] for p in r.action_positions)==r.action_ids
    for original,segment in zip(samples,r.segments):
        assert segment.sampled_ids==tuple(original["ids"])
        assert segment.action_positions==tuple(original["action_positions"])
        all_positions=set(range(len(segment.prompt_ids),len(original["transcript_ids"])))
        inserted_positions=sorted(all_positions-set(segment.action_positions))
        assert tuple(original["transcript_ids"][p] for p in inserted_positions)==segment.inserted_ids
        if model=="ring":
            assert segment.inserted_ids==tuple(enc("\n</think>\n"))
            assert inserted_positions and not set(inserted_positions).intersection(segment.action_positions)
        else: assert not inserted_positions

def test_unobserved_correct_answer_gets_zero(context):
    model,tok,enc,dec,env,framer,prompt,cfg,sample=context
    r=run_episode(prompt,lambda p,**kw:sample(p,json.dumps({"submit":env.expected})),env,env.grade,enc,dec,config=cfg,
                  frame_observation=lambda ids,**kw:framer.append_tool(ids,**kw).transcript_ids)
    assert r.complete and not r.success and r.reward==0 and not env.calls

def test_forged_sampler_text_is_qualification_error(context):
    model,tok,enc,dec,env,framer,prompt,cfg,sample=context
    def bad(p,**kw):
        r=sample(p,'{"tool":"ls","args":{}}');r["raw_text"]="forged";return r
    with pytest.raises(EpisodeError,match="raw decoded"):
        run_episode(prompt,bad,env,env.grade,enc,dec,config=cfg,
                    frame_observation=lambda ids,**kw:framer.append_tool(ids,**kw).transcript_ids)
    assert not env.calls
