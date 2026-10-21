"""Private-oracle CPU task suite for on-policy protocol controls."""
from __future__ import annotations
import ast, json, operator, re
from dataclasses import dataclass
from pathlib import Path

_SPLITS = ("training", "development", "held_out", "retention")
_ALIAS = {"heldout":"held_out", "dev":"development"}
_ALLOWED = (int,)

@dataclass(frozen=True)
class Task:
    id: str
    split: str
    family: str
    text: str
    _raw: dict
    def __repr__(self):
        return f"Task(id={self.id!r}, split={self.split!r}, family={self.family!r})"

class Suite:
    def __init__(self, tasks):
        self._tasks = tuple(tasks)
        self._by_id = {t.id:t for t in tasks}
    @property
    def tasks(self):
        return self._tasks
    def __iter__(self): return iter(self._tasks)
    def get(self, task):
        if isinstance(task, Task): return task
        if isinstance(task, str): return self._by_id[task]
        if isinstance(task, dict): return self._by_id[task["id"]]
        raise TypeError("task must be Task, id, or public task dict")

def load_suite(path):
    data=json.loads(Path(path).read_text())
    groups={}
    for key, vals in data.items():
        split=_ALIAS.get(key,key)
        if split in _SPLITS:
            if not isinstance(vals,list): raise ValueError(f"{split} must be a list")
            groups.setdefault(split,[]).extend(vals)
    missing=[s for s in ("training","held_out","retention") if s not in groups]
    if missing: raise ValueError("missing required splits: "+", ".join(missing))
    seen=set(); tasks=[]
    for split in _SPLITS:
        if split not in groups: continue
        for raw in groups[split]:
            tid=raw.get("id"); text=raw.get("prompt")
            if not isinstance(tid,str) or not tid or tid in seen: raise ValueError("duplicate or invalid task id")
            if not isinstance(text,str) or not text.strip(): raise ValueError(f"invalid prompt for {tid}")
            if len(text.split()) >= 70: raise ValueError(f"prompt too long: {tid}")
            fam=raw.get("family", "retention" if split=="retention" else None)
            if fam not in ("math","extraction","retention"): raise ValueError(f"invalid family: {tid}")
            # Strip oracle-bearing verifier content from the public task object.
            public={"id":tid,"split":split,"family":fam,"prompt":text}
            candidate=Task(tid,split,fam,text,{})
            try: derived=_derive(candidate)
            except Exception as exc: raise ValueError(f"oracle derivation failed for {tid}: {exc}") from exc
            verifier=raw.get("verifier")
            if not isinstance(verifier,dict) or "private_answer" not in verifier:
                raise ValueError(f"missing private verifier answer for {tid}")
            if str(verifier["private_answer"]) != derived:
                raise ValueError(f"private verifier mismatch for {tid}")
            tasks.append(candidate)
            seen.add(tid)
    return Suite(tasks)

def prompt(task):
    if not isinstance(task,Task): raise TypeError("prompt requires Task from load_suite")
    return task.text

def _safe_expr(expr):
    tree=ast.parse(expr,mode="eval")
    def ev(n):
        if isinstance(n,ast.Constant) and type(n.value) is int: return n.value
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)): return +ev(n.operand) if isinstance(n.op,ast.UAdd) else -ev(n.operand)
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult)): return {ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul}[type(n.op)](ev(n.left),ev(n.right))
        raise ValueError("unsupported arithmetic")
    return ev(tree.body)

def _derive(task):
    t=task.text
    if task.family=="math":
        m=re.search(r"Compute\s+([0-9()+*\- ]+)\.",t,re.I)
        if m: return str(_safe_expr(m.group(1).strip()))
        m=re.search(r"(\d+)\s+rows?\s+of\s+(\d+)\s+coins?\.\s+Remove\s+(\d+)",t,re.I)
        if m: return str(int(m.group(1))*int(m.group(2))-int(m.group(3)))
        m=re.search(r"(\d+)\s+stacks?\s+of\s+(\d+)\s+books?\.\s+Remove\s+(\d+)",t,re.I)
        if m: return str(int(m.group(1))*int(m.group(2))-int(m.group(3)))
        raise ValueError("unrecognized math prompt")
    if task.family=="extraction":
        blocks=re.findall(r"`([^`]*)`",t)
        if len(blocks)<1: raise ValueError("extraction requires backtick data")
        keym=re.search(r"value after `([^=`]+)=`|return the ([A-Za-z_][\w-]*) value",t,re.I)
        key=keym.group(1) if keym and keym.group(1) else (keym.group(2) if keym else None)
        key={"department":"dept"}.get(key,key)
        if not key: raise ValueError("missing extraction key")
        data=next((b for b in blocks if not b.endswith("=")),blocks[0])
        m=re.search(r"(?:^|[;| ])"+re.escape(key)+r"[:=]([^;| ]+)",data)
        if not m: raise ValueError("key absent")
        return m.group(1)
    if task.family=="retention":
        if re.search(r"third letter",t,re.I):
            w=re.search(r"word `([^`]+)`",t,re.I).group(1); return w[2]
        if re.search(r"how many vowels",t,re.I):
            w=re.search(r"`([^`]+)`",t).group(1); return str(sum(c.lower() in "aeiou" for c in w))
        if re.search(r"sort the letters",t,re.I):
            w=re.search(r"`([^`]+)`",t).group(1); return ",".join(sorted(w.replace(" ","")))
        if re.search(r"last digit",t,re.I): return re.search(r"of (\d+)",t,re.I).group(1)[-1]
    raise ValueError("unrecognized task")

def reference(task):
    if not isinstance(task,Task): raise TypeError("reference requires Task from load_suite")
    if task.split != "training": raise PermissionError("private reference unavailable outside training split")
    return _derive(task)

def grade(task,text,complete):
    if not isinstance(task,Task): raise TypeError("grade requires Task from load_suite")
    if not isinstance(text,str): return {"reward":0,"reason":"non-string output"}
    if type(complete) is not bool: raise TypeError("complete must be bool")
    if not complete: return {"reward":0,"reason":"truncated_or_missing_eos"}
    out=text.strip()
    if not out: return {"reward":0,"reason":"empty_output"}
    if "```" in out or "<tool" in out.lower() or "grader" in out.lower() or "correct" in out.lower(): return {"reward":0,"reason":"verdict_or_injection_text"}
    if "\n" in out or "\r" in out: return {"reward":0,"reason":"extra_lines"}
    try: expected=_derive(task)
    except Exception as e: raise ValueError("oracle_parse_error") from e
    if task.family=="math" and not re.fullmatch(r"-?\d+",out): return {"reward":0,"reason":"not_bare_integer"}
    if task.family in ("extraction","retention") and not re.fullmatch(r"[A-Za-z0-9,._-]+",out): return {"reward":0,"reason":"not_bare_literal"}
    return {"reward":int(out==expected),"reason":"exact_match" if out==expected else "wrong_answer"}
