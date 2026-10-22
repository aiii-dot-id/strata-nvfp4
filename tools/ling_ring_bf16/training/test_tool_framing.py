import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tool_framing import ExactChatFramer,FramingError
class Tok:
 def apply_chat_template(self,messages,**kw):
  return [ord(c) for m in messages for c in ("<"+m["role"]+">"+m["content"])]
def test_exact_prefix_and_suffix():
 f=ExactChatFramer(Tok(),model="ling",schemas={"ls":{}},context_capacity=1024)
 a=f.start("query"); b=f.append_tool(a.transcript_ids,assistant_text='{"tool":"ls","args":{}}',tool_name="ls",args={},result={"x":1})
 assert b.transcript_ids[:len(a.transcript_ids)]==a.transcript_ids and b.suffix_ids

def test_prefix_drift_rejected():
 f=ExactChatFramer(Tok(),model="ling",schemas={},context_capacity=1024); a=f.start("q")
 try: f.append_tool(tuple(list(a.transcript_ids[:-1])+[999]),assistant_text="x",tool_name="ls",args={},result={})
 except FramingError: pass
 else: raise AssertionError("drift accepted")
def test_ring_close_is_explicit_context_suffix():
 f=ExactChatFramer(Tok(),model="ring",schemas={},context_capacity=1024); a=f.start("q")
 close=tuple(ord(c) for c in "</think>\n")
 b=f.append_ring_close(a.transcript_ids,close_ids=close,assistant_text="",tool_name="ls",args={},result={})
 assert any(b.suffix_ids[i:i+len(close)]==close for i in range(len(b.suffix_ids)-len(close)+1))

def test_real_checkpoint_tokenizers_exact_prefix():
    import os
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return
    for model,path in (("ling","/work/AI/checkpoints/Ling-3.0-tiny"),("ring","/work/AI/checkpoints/Ring-mini-2.0")):
        if not os.path.exists(path+"/tokenizer.json"): return
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=True,use_fast=True)
        f=ExactChatFramer(tok,model=model,schemas={"ls":{}},context_capacity=4096)
        start=f.start("read the fixture")
        turn=f.append_tool(start.transcript_ids,assistant_text='{"tool":"ls","args":{}}',tool_name="ls",args={},result={"ok":True})
        assert turn.transcript_ids[:len(start.transcript_ids)]==start.transcript_ids
        assert turn.suffix_ids

def test_real_checkpoint_two_turn_sampled_prefix_and_opening():
    import os
    from transformers import AutoTokenizer
    from tool_framing import ExactChatFramer
    for model,path in (("ling","/work/AI/checkpoints/Ling-3.0-tiny"),("ring","/work/AI/checkpoints/Ring-mini-2.0")):
        if not os.path.exists(path+"/tokenizer.json"): return
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=True,use_fast=True)
        eos=tok.eos_token_id; f=ExactChatFramer(tok,model=model,schemas={"ls":{}},context_capacity=4096,eos_token_ids=(eos,))
        first=f.start("read the fixture")
        action='{"tool":"ls","args":{}}'; action_ids=tuple(tok(action,add_special_tokens=False).input_ids)+(eos,)
        # The sampled transcript contains the action and explicit EOS before the observation framing.
        pre=first.transcript_ids+action_ids
        turn=f.append_tool(pre,assistant_text=action,tool_name="ls",args={},result={"ok":True})
        assert turn.transcript_ids[:len(pre)]==pre
        second='{"submit":"ok"}'; second_ids=tuple(tok(second,add_special_tokens=False).input_ids)+(eos,)
        turn2=f.append_tool(turn.transcript_ids,assistant_text=second,tool_name="submit",args={},result={"accepted":True})
        assert turn2.transcript_ids[:len(turn.transcript_ids)]==turn.transcript_ids
        assert turn2.suffix_ids
