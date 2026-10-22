"""Read-only canonical tool execution and evidence-grounded task rewards.
AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations
import hashlib, json, math, os, re, signal, subprocess, time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

def unique_object(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise ValueError("duplicate JSON field")
        out[k]=v
    return out

def strict_loads(value):
    return json.loads(value, object_pairs_hook=unique_object,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",",":"), allow_nan=False)

def relative_path(value):
    if not isinstance(value,str) or not value or "\x00" in value or "\\" in value: raise ValueError("relative path required")
    p=Path(value)
    if p.is_absolute() or any(x in ("..",".") for x in value.split("/")): raise ValueError("fixture-relative path required")
    return p.as_posix()

def verify_fixture(root, files):
    root=Path(root)
    if root.is_symlink() or not root.is_dir() or root.resolve()!=root.absolute(): raise ValueError("real fixture directory required")
    if not isinstance(files,list) or not files: raise ValueError("fixture manifest required")
    expected={}
    for ent in files:
        rel=relative_path(ent["relative_path"])
        if rel in expected: raise ValueError("duplicate fixture file")
        expected[rel]=ent
    actual={}
    for p in root.rglob("*"):
        if p.is_symlink() or not (p.is_file() or p.is_dir()): raise ValueError("special fixture entry")
        if p.is_file(): actual[p.relative_to(root).as_posix()]=p
    if actual.keys()!=expected.keys(): raise ValueError("fixture file set differs")
    for rel,p in actual.items():
        e=expected[rel]
        if type(e.get("bytes")) is not int or e["bytes"]!=p.stat().st_size or sha(p)!=e.get("sha256"): raise ValueError("fixture bytes differ")
    return expected

class CanonicalToolEnvironment:
    """One episode; tool process is owned, deadline-bound, and read-only."""
    def __init__(self, binary, binary_sha256, root, files, expected, *, max_calls=4, timeout=3., max_result_bytes=4096):
        self.binary=Path(binary).resolve(); self.root=Path(root).absolute()
        if sha(self.binary)!=binary_sha256: raise ValueError("tool binary pin")
        if type(max_calls) is not int or max_calls<1: raise ValueError("call budget")
        if type(timeout) not in (float,int) or not math.isfinite(timeout) or timeout<=0: raise ValueError("deadline")
        if type(max_result_bytes) is not int or not 64<=max_result_bytes<=65536: raise ValueError("result budget")
        self.binary_sha=binary_sha256; self.files=files; self.expected=dict(expected); self.max_calls=max_calls
        self.timeout=float(timeout); self.max_result_bytes=max_result_bytes; self.calls=[]; self.observed=set()
        verify_fixture(self.root,self.files)
        self._validate_expected()

    def _validate_expected(self):
        e=self.expected
        if set(e)!={"path","line","text"} or type(e["line"])is not int or e["line"]<1 or not isinstance(e["text"],str): raise ValueError("oracle schema")
        rel=relative_path(e["path"]); p=self.root/rel
        if rel not in {x["relative_path"] for x in self.files}: raise ValueError("oracle outside fixture")
        lines=p.read_text(encoding="utf-8").split("\n")
        if e["line"]>len(lines) or lines[e["line"]-1]!=e["text"]: raise ValueError("oracle contradicts fixture")

    def __call__(self, tool, args):
        if len(self.calls)>=self.max_calls:
            entry={"ordinal":len(self.calls)+1,"tool":tool,"arguments":args,"delivered":False,
                   "error":"ValueError:tool call budget exhausted"}
            self.calls.append(entry)
            return {"error":entry["error"]}
        entry={"ordinal":len(self.calls)+1,"tool":tool,"arguments":args,"delivered":False}
        self.calls.append(entry)  # Attempts count even when invalid.
        started=time.monotonic()
        try:
            verify_fixture(self.root,self.files)
            if sha(self.binary)!=self.binary_sha: raise ValueError("tool binary changed")
            if tool not in ("read","grep","ls") or not isinstance(args,dict): raise ValueError("read-only tool required")
            request=(canonical({"tool":tool,"arguments":args})+"\n").encode()
            if len(request)>32768: raise ValueError("request budget")
            env={"PATH":"/usr/bin:/bin","HOME":str(self.root),"LANG":"C.UTF-8","LC_ALL":"C.UTF-8"}
            proc=subprocess.Popen([str(self.binary),str(self.root)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,cwd=self.root,start_new_session=True)
            try:
                stdout,stderr=proc.communicate(request,timeout=self.timeout)
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGKILL)
                proc.wait(timeout=3.)
                raise
            if proc.returncode!=0: raise ValueError("tool process failed")
            if len(stdout)>65537 or len(stderr)>4096: raise ValueError("bridge output budget")
            lines=stdout.decode("utf-8").splitlines()
            if len(lines)!=1: raise ValueError("one tool response required")
            result=strict_loads(lines[0])
            if not isinstance(result,dict) or result.get("tool")!=tool: raise ValueError("tool response binding")
            if "result" not in result and "error" not in result: raise ValueError("tool result envelope")
            verify_fixture(self.root,self.files)
            entry["response_sha256"]=hashlib.sha256(stdout).hexdigest()
            if len(canonical(result).encode())>self.max_result_bytes:
                result={"error":"tool result exceeds episode byte budget","output_limited":True}
            entry["result_sha256"]=hashlib.sha256(canonical(result).encode()).hexdigest()
            entry["result"]=result; entry["delivered"]=True
            entry["evidence"]=sorted(self._extract_evidence(tool,args,result))
            return result
        except Exception as exc:
            entry["error"]=type(exc).__name__+":"+str(exc)
            result={"error":entry["error"]}
            entry["result"]=result; entry["delivered"]=True
            return result
        finally:
            entry["wall_seconds"]=time.monotonic()-started

    def _extract_evidence(self,tool,args,response):
        found=set()
        r=response.get("result")
        if not isinstance(r,dict) or r.get("Error") or r.get("Refused") or r.get("Truncated"): return found
        text=r.get("Output")
        if not isinstance(text,str): return found
        if tool=="read":
            try: rel=relative_path(args["file_path"])
            except (ValueError,KeyError): return found
            if rel not in {x["relative_path"] for x in self.files}: return found
            for line in text.split("\n"):
                m=re.fullmatch(r" *([1-9][0-9]*)\t(.*)",line)
                if m: found.add((rel,int(m[1]),m[2]))
        elif tool=="grep":
            for line in text.split("\n"):
                for ent in self.files:
                    rel=ent["relative_path"]; prefix=str(self.root/rel)+":"
                    if line.startswith(prefix):
                        rest=line[len(prefix):]; m=re.fullmatch(r"([1-9][0-9]*):(.*)",rest)
                        if m: found.add((rel,int(m[1]),m[2]))
        return found

    def grade(self, submission, episode):
        verify_fixture(self.root,self.files)
        if not isinstance(episode,dict): raise ValueError("episode receipt required")
        delivered=episode.get("delivered_tool_result_hashes")
        if not isinstance(delivered,(list,tuple)) or any(not isinstance(x,str) or not re.fullmatch(r"[0-9a-f]{64}",x) for x in delivered):
            raise ValueError("delivered tool-result hashes required")
        accepted=set(delivered); self.observed=set()
        for entry in self.calls:
            if entry.get("delivered") and entry.get("result_sha256") in accepted:
                self.observed.update(tuple(x) for x in entry.get("evidence",()))
        valid=(isinstance(submission,dict) and set(submission)=={"path","line","text"} and
               type(submission.get("line")) is int and submission==self.expected)
        supported=valid and (submission["path"],submission["line"],submission["text"]) in self.observed
        reported=episode.get("tool_attempts")
        if type(reported) is not int or reported<0: raise ValueError("tool attempt receipt required")
        attempts=max(len(self.calls),reported)
        success=bool(supported and 1<=attempts<=self.max_calls)
        reward=1.+0.1*(self.max_calls-attempts)/self.max_calls if success else 0.
        return {"success":success,"reward":reward,"tool_attempts":attempts,
                "reason":"verified_delivered_observation" if success else "wrong_or_undelivered_answer",
                "calls_executed":len(self.calls)}
