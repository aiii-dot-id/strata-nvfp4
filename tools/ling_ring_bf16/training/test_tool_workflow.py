import sys,types
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import tool_workflow as w
from tool_native import ToolSample

class FakeTok:
    eos_token_id=99
    def encode(self,s,**kw): return [ord(x)%1000 for x in s]
    def decode(self,ids,**kw): return ''.join(chr(x) for x in ids)
    def apply_chat_template(self,messages,**kw): return [1,2,3]

class FakeFramer:
    def __init__(self,*a,**kw): pass
    def start(self,text): return types.SimpleNamespace(transcript_ids=(1,2,3))
    def append_tool(self,existing_ids,**kw):
        obs=tuple(existing_ids)+(7,8)
        return types.SimpleNamespace(transcript_ids=obs)

class FakeEnv:
    def __init__(self,*a,**kw): self.calls=[]
    def __call__(self,tool,args): self.calls.append({"tool":tool,"arguments":args,"wall_seconds":0.0,"result_sha256":"a"*64}); return {"tool":tool,"result":{"Output":"b.txt:2:verified_target = 42\n"}}
    def grade(self,value,episode): return {"success":value=={"path":"b.txt","line":2,"text":"verified_target = 42"},"reward":1.0}

class FakeSampler:
    instances=[]
    def __init__(self,*a,**kw): self.calls=[];self.__class__.instances.append(self)
    def sample(self,prompt,adapter=None,seed=0):
        self.calls.append((tuple(prompt),adapter,seed))
        if len(self.calls)==1: text='{"tool":"ls","args":{}}'
        else: text='{"submit":{"path":"b.txt","line":2,"text":"verified_target = 42"}}'
        ids=tuple(ord(x) for x in text)+(99,); full=tuple(prompt)+ids
        rows=tuple({"features":np.zeros(4,np.float32),"logits":np.zeros(3,np.float32)} for _ in ids)
        return ToolSample(text,text,full,ids,tuple(-.1 for _ in ids),tuple(range(len(prompt),len(full))),(),False,True,None,rows,({"stage":"completion","rows":len(ids),"positions":[]},))

def setup(monkeypatch,tmp_path):
    root=tmp_path/"root";root.mkdir();(root/"a.txt").write_text("x")
    manifest=tmp_path/"manifest";manifest.write_text("m");exe=tmp_path/"exe";exe.write_text("e")
    monkeypatch.setattr(w.native,"MODELS",{"ling":{"manifest":str(manifest),"checkpoint":str(tmp_path/"cp"),"capacity":128,"hidden":4}})
    monkeypatch.setattr(w.native,"sha",lambda p: __import__("hashlib").sha256(Path(p).read_bytes()).hexdigest())
    monkeypatch.setattr(w.native,"_source_closure",lambda root:{"a":{"sha256":"x"}})
    monkeypatch.setattr(w.native,"_git_state",lambda root:{"dirty":False,"commit":"c","status":[]})
    monkeypatch.setattr(w,"ExactChatFramer",FakeFramer);monkeypatch.setattr(w,"CanonicalToolEnvironment",FakeEnv);monkeypatch.setattr(w,"NativeToolSampler",FakeSampler)
    mod=types.ModuleType("generate_ling");mod._eos_ids=lambda cp:((99,),"fixture");monkeypatch.setitem(sys.modules,"generate_ling",mod)
    return root,manifest,exe

def test_native_sample_dict_keeps_all_action_rows(tmp_path):
    rows=tuple({"features":np.zeros(2),"logits":np.ones(3)} for _ in range(2))
    x=ToolSample("raw","final",(1,2),(3,4),(-.1,-.2),(1,2),(),False,True,None,rows,())
    d=w.native_sample_dict(x)
    assert d["ids"]==(3,4) and d["text"]=="final" and d["raw_text"]=="raw"
    assert d["features"].shape==(2,2) and d["logits"].shape==(2,3)

def test_workflow_wires_sampler_environment_and_receipt(tmp_path,monkeypatch):
    root,manifest,exe=setup(monkeypatch,tmp_path)
    files=[{"relative_path":"a.txt","bytes":1,"sha256":w.native.sha(root/"a.txt")}]
    out=tmp_path/"out"
    result=w.sample_tool_task("ling",FakeTok(),{"id":"task-1","split":"teacher_selection","family":"x","prompt":"find"},
        root,files,{"path":"b.txt","line":2,"text":"verified_target = 42"},binary=str(tmp_path/"bridge"),binary_sha256="b"*64,
        output=out,executable=exe,source_closure={"a":{"sha256":"x"}},max_calls=2)
    assert result["success"] and result["tool_attempts"]==1 and len(result["ids"])>0
    assert result["tool_records"] and result["delivered_tool_result_hashes"]
    assert (out/"ROLLOUT.json").exists()
    assert FakeSampler.instances[-1].calls[0][2]==1000000

def test_selection_gate_rejects_quality_regression_and_accepts_gains():
    b=[{"task":"a","split":"teacher_selection","success":False,"complete":True,"tool_attempts":2},{"task":"b","split":"teacher_selection","success":False,"complete":True,"tool_attempts":2},{"task":"c","split":"teacher_selection","success":True,"complete":True,"tool_attempts":3}]
    c=[{"task":"a","split":"teacher_selection","success":True,"complete":True,"tool_attempts":2},{"task":"b","split":"teacher_selection","success":True,"complete":True,"tool_attempts":1},{"task":"c","split":"teacher_selection","success":True,"complete":True,"tool_attempts":1}]
    g=w.selection_gate(b,c)
    assert g["select"] and g["quality_gains"]==["a","b"]
    c[2]["success"]=False
    g2=w.selection_gate(b,c); assert not g2["select"] and g2["quality_regressions"]==["c"]
