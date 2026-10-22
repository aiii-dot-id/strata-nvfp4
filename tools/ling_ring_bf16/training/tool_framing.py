"""Exact token-prefix framing for injected tool turns.
AI Identity Incorported <james@aiii.id>.

The tokenizer/chat template is supplied by the caller and must be pinned by the
campaign. This module never decodes and re-encodes an already accepted prefix.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

class FramingError(ValueError): pass
class FramingCapacityError(FramingError): pass

@dataclass(frozen=True)
class FramedTurn:
    transcript_ids: tuple[int,...]
    suffix_ids: tuple[int,...]
    messages: tuple[dict[str,Any],...]
    prompt_tokens: int


def _ids(value):
    if hasattr(value, "tolist"): value=value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list): value=value[0]
    if not isinstance(value, (list,tuple)) or any(type(x) is not int or x<0 for x in value):
        raise FramingError("tokenizer returned invalid IDs")
    return tuple(value)

class ExactChatFramer:
    def __init__(self, tokenizer, *, model:str, schemas:dict, context_capacity:int, eos_token_ids=()):
        if model not in ("ling","ring") or not isinstance(schemas,dict): raise ValueError("model/schemas")
        if type(context_capacity) is not int or context_capacity<=0: raise ValueError("context capacity")
        if not callable(getattr(tokenizer,"apply_chat_template",None)): raise TypeError("tokenizer chat template")
        self.tokenizer=tokenizer; self.model=model; self.schemas=schemas; self.capacity=context_capacity
        declared=eos_token_ids if eos_token_ids else ((getattr(tokenizer,"eos_token_id",None),) if getattr(tokenizer,"eos_token_id",None) is not None else ())
        self.eos_token_ids=tuple(x for x in declared if type(x) is int and x>=0)
        self.messages: list[dict[str,Any]]=[]
    def _render(self, messages):
        kw={"tokenize":True,"add_generation_prompt":True}
        if self.model=="ling": kw["enable_thinking"]=False
        try: rendered=self.tokenizer.apply_chat_template(messages,**kw)
        except TypeError:
            kw.pop("enable_thinking",None); rendered=self.tokenizer.apply_chat_template(messages,**kw)
        return _ids(rendered)
    def start(self, user_text:str)->FramedTurn:
        if not isinstance(user_text,str) or not user_text: raise ValueError("user text")
        self.messages=[{"role":"system","content":self._schema_header()},{"role":"user","content":user_text}]
        ids=self._render(self.messages)
        if len(ids)>self.capacity: raise FramingCapacityError("context capacity")
        return FramedTurn(ids,ids,tuple(self.messages),len(ids))
    def _schema_header(self):
        import json
        return ("Tool actions must be strict JSON. Available schemas:\n"+
                json.dumps(self.schemas,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False))
    def append_tool(self, existing_ids, *, assistant_text:str, tool_name:str, args:dict, result:Any)->FramedTurn:
        if not isinstance(assistant_text,str) or not isinstance(tool_name,str) or not isinstance(args,dict): raise ValueError("tool turn")
        if not self.messages: raise FramingError("start required")
        # Tool observations are represented as a user turn for both templates;
        # this is the pinned compatibility surface used by the campaign.
        import json
        observation=json.dumps({"tool":tool_name,"args":args,"result":result},sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
        assistant_content=assistant_text
        if self.model=="ring" and not assistant_content.startswith("<think>"):
            assistant_content="<think>\n"+assistant_content
        msgs=self.messages+[ {"role":"assistant","content":assistant_content}, {"role":"user","content":observation} ]
        full=self._render(msgs)
        old=_ids(existing_ids)
        oi=fi=0
        while oi<len(old) and fi<len(full):
            if old[oi]==full[fi]: oi+=1; fi+=1; continue
            if self.model=="ring" and old[oi] in self.eos_token_ids and (oi+1==len(old) or old[oi+1]==full[fi]):
                oi+=1; continue
            break
        if oi!=len(old): raise FramingError("chat template does not preserve exact transcript prefix")
        boundary=fi
        suffix=full[boundary:]
        transcript=old+suffix
        if len(transcript)>self.capacity: raise FramingCapacityError("context capacity")
        self.messages=msgs
        return FramedTurn(transcript,suffix,tuple(msgs),len(transcript))
    def append_ring_close(self, existing_ids, *, close_ids, assistant_text:str, tool_name:str, args:dict, result:Any)->FramedTurn:
        close=_ids(close_ids)
        if not close: raise FramingError("empty close IDs")
        # The caller must include the exact sampled reasoning/action text. Close
        # IDs are inserted context and are excluded from policy action rows.
        turn=self.append_tool(existing_ids,assistant_text=assistant_text+"</think>\n",tool_name=tool_name,args=args,result=result)
        if not any(turn.suffix_ids[i:i+len(close)]==close for i in range(len(turn.suffix_ids)-len(close)+1)):
            raise FramingError("inserted Ring close IDs do not match rendered suffix")
        return turn
