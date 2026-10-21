"""Bounded Ling/Ring completion controller with retained format failures.
AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math, re
from typing import Any, Callable, Sequence
_SPECIAL = re.compile(r"<\|[^>]*\|>")

@dataclass(frozen=True)
class CompletionConfig:
    model: str = "ling"
    max_new_tokens: int = 32
    thinking_tokens: int = 128
    answer_tokens: int = 96
    context_capacity: int = 4096
    close_text: str = "\n</think>\n"
    eos_token_ids: tuple[int, ...] = ()

@dataclass(frozen=True)
class CompletionStage:
    name: str
    prompt_ids: tuple[int, ...]
    sampled_ids: tuple[int, ...]
    sampled_logprobs: tuple[float, ...]
    text: str
    eos: bool
    finish_reason: str
    inserted_ids: tuple[int, ...]
    action_positions: tuple[int, ...]
    format_error: str | None = None

@dataclass(frozen=True)
class CompletionResult:
    text: str
    transcript_ids: tuple[int, ...]
    sampled_ids: tuple[int, ...]
    sampled_logprobs: tuple[float, ...]
    inserted_ids: tuple[int, ...]
    action_positions: tuple[int, ...]
    forced_close: bool
    complete: bool
    stages: tuple[CompletionStage, ...]
    format_error: str | None = None

class CompletionError(ValueError):
    pass

def _check_config(c, prompt, encode):
    if c.model not in ("ling", "ring"): raise ValueError("model must be ling or ring")
    for n in (c.max_new_tokens,c.thinking_tokens,c.answer_tokens,c.context_capacity):
        if type(n) is not int or n <= 0: raise ValueError("completion limits must be positive integers")
    if not prompt or any(type(x) is not int or x < 0 for x in prompt): raise TypeError("prompt_ids invalid")
    if any(type(x) is not int or x < 0 for x in c.eos_token_ids): raise TypeError("eos_token_ids invalid")
    if c.model == "ling":
        if len(prompt)+c.max_new_tokens > c.context_capacity: raise CompletionError("Ling completion exceeds context capacity")
        return ()
    if c.close_text.count("</think>") != 1 or "<think>" in c.close_text: raise ValueError("invalid close_text")
    close_ids=tuple(encode(c.close_text))
    if not close_ids or any(type(x) is not int or x < 0 for x in close_ids): raise TypeError("close encoder returned invalid IDs")
    if len(prompt)+c.thinking_tokens > c.context_capacity: raise CompletionError("Ring thinking reserve exceeds context capacity")
    return close_ids

def _invoke(run,prompt,limit,adapter,name,inserted):
    out=run(prompt,max_new_tokens=limit,adapter=adapter,stage=name)
    if not isinstance(out,dict): raise TypeError("native callback must return a dict")
    ids,lps,text=out.get("ids"),out.get("sampled_logprobs"),out.get("text"); eos,reason=out.get("eos"),out.get("finish_reason")
    if not isinstance(ids,(list,tuple)) or not ids or any(type(x) is not int or x<0 for x in ids): raise TypeError("native callback ids invalid")
    if len(ids)>limit: raise CompletionError("native callback exceeded token reserve")
    if not isinstance(lps,(list,tuple)) or len(lps)!=len(ids) or any(type(x) not in (int,float) or not math.isfinite(float(x)) for x in lps): raise TypeError("native sampled logprobs invalid")
    if not isinstance(text,str) or type(eos) is not bool or reason not in ("eos","length") or eos!=(reason=="eos"): raise TypeError("native completion fields invalid")
    return CompletionStage(name,tuple(prompt),tuple(ids),tuple(float(x) for x in lps),text,eos,reason,tuple(inserted),(),None)

def _issue(text,allow_close=True,require_close=False):
    if _SPECIAL.search(text): return "special token text leaked"
    if "<think>" in text: return "nested opening thinking delimiter"
    n=text.count("</think>")
    if n>1: return "multiple closing thinking delimiters"
    if not allow_close and n: return "unexpected closing thinking delimiter"
    if require_close and n!=1: return "missing closing thinking delimiter"
    return None

def _decode(decode,original,transcript,eos_ids,require_close):
    if transcript[:len(original)]!=original: raise CompletionError("transcript does not start with original prompt")
    ids=list(transcript[len(original):])
    if ids and ids[-1] in eos_ids: ids.pop()
    text=decode(ids)
    if not isinstance(text,str): raise TypeError("decoder must return text")
    return text,_issue(text,True,require_close)

def _result(original,stages,inserted,decode,eos_ids,forced,complete,extra_issue=None):
    # Prompt IDs are authoritative; the final stage prompt plus its sampled IDs
    # contains the original prompt, prior sampled actions, and any inserted IDs.
    if len(stages)==1: transcript=stages[0].prompt_ids+stages[0].sampled_ids
    else: transcript=stages[-1].prompt_ids+stages[-1].sampled_ids
    actions=tuple(x for st in stages for x in st.sampled_ids); logs=tuple(x for st in stages for x in st.sampled_logprobs)
    positions=[]; fixed_stages=[]
    for st in stages:
        local=tuple(range(len(st.prompt_ids),len(st.prompt_ids)+len(st.sampled_ids)))
        positions.extend(local); fixed_stages.append(replace(st,action_positions=local))
    text,decoded_issue=_decode(decode,original,transcript,eos_ids,False)
    issue=extra_issue or decoded_issue or next((st.format_error for st in stages if st.format_error),None)
    return CompletionResult(text,tuple(transcript),actions,logs,tuple(inserted),tuple(positions),forced,complete,tuple(fixed_stages),issue)

def run_completion(prompt_ids,adapter,run,encode,decode,config=CompletionConfig()):
    original=tuple(prompt_ids); close_ids=_check_config(config,original,encode)
    if config.model=="ring" and decode(close_ids)!=config.close_text: raise CompletionError("close encoder/decoder binding mismatch")
    if config.model=="ling":
        st=_invoke(run,original,config.max_new_tokens,adapter,"completion",())
        st=replace(st,format_error=_issue(st.text,False))
        return _result(original,(st,),(),decode,config.eos_token_ids,False,st.eos)
    first=_invoke(run,original,config.thinking_tokens,adapter,"thinking",())
    first_issue=_issue(first.text,True)
    first=replace(first,format_error=first_issue)
    nclose=first.text.count("</think>")==1
    if first.eos:
        return _result(original,(first,),(),decode,config.eos_token_ids,False,True,first_issue or (None if nclose else "missing closing thinking delimiter"))
    if first_issue and first_issue!="missing closing thinking delimiter":
        return _result(original,(first,),(),decode,config.eos_token_ids,False,False,first_issue)
    inserted=() if nclose else close_ids
    second_prompt=original+first.sampled_ids+inserted
    if len(second_prompt)+config.answer_tokens>config.context_capacity: raise CompletionError("Ring answer reserve exceeds context capacity")
    second=_invoke(run,second_prompt,config.answer_tokens,adapter,"answer",inserted)
    second_issue=_issue(second.text,False)
    second=replace(second,format_error=second_issue)
    return _result(original,(first,second),inserted,decode,config.eos_token_ids,bool(inserted),second.eos,second_issue)
