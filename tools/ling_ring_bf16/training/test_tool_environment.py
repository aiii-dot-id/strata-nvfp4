"""CPU tests using the pinned canonical Go read/grep/ls bridge."""
import json, os
from pathlib import Path
import pytest
from tool_environment import CanonicalToolEnvironment, sha, verify_fixture

BRIDGE=Path("/work/AI/experiments/strata-capability-20261021-r2/tool-strategy-prep/aii-tool-bridge/aii-tool-bridge")
PIN="211f546da0b2518738fda47ed04698241d517e54c7e82504e762396113d76175"
def env(tmp_path,max_result_bytes=4096):
    root=tmp_path/"fixture";root.mkdir()
    (root/"a.txt").write_text("unrelated\nother\n")
    (root/"b.txt").write_text("first\nverified_target = 42\n")
    files=[{"relative_path":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(root.iterdir())]
    return CanonicalToolEnvironment(BRIDGE,PIN,root,files,{"path":"b.txt","line":2,"text":"verified_target = 42"},max_result_bytes=max_result_bytes)

def ep(e,attempts=None,delivered=True):
    if attempts is None: attempts=len(e.calls)
    hs=[x.get("result_sha256") for x in e.calls if x.get("result_sha256")] if delivered else []
    return {"tool_attempts":attempts,"delivered_tool_result_hashes":hs}

def test_requires_actual_observation(tmp_path):
    e=env(tmp_path)
    assert not e.grade(e.expected,ep(e,0))["success"]
    r=e("grep",{"pattern":"verified_target","literal":True})
    assert r["result"]["Output"]==str(e.root/"b.txt")+":2:verified_target = 42\n"
    g=e.grade(e.expected,ep(e,1))
    assert g["success"] and g["reward"]==1.075
def test_extra_calls_cost_only_success(tmp_path):
    e=env(tmp_path); e("ls",{});e("read",{"file_path":"b.txt"})
    assert e.grade(e.expected,ep(e,2))["reward"]==1.05
    assert e.grade({"path":"b.txt","line":1,"text":"first"},ep(e,2))["reward"]==0
    assert e.grade(e.expected,ep(e,5))["reward"]==0
def test_refusal_and_invalid_attempts_count(tmp_path):
    e=env(tmp_path)
    assert "error" in e("shell",{"command":"true"})
    r=e("read",{"file_path":"../../etc/passwd"})
    assert r.get("error") or r.get("result",{}).get("Refused") or r.get("result",{}).get("Error")
    assert not e.observed and len(e.calls)==2
def test_mutation_and_wrong_oracle_rejected(tmp_path):
    e=env(tmp_path);(e.root/"b.txt").write_text("changed\n")
    with pytest.raises(ValueError,match="fixture bytes"): e.grade(e.expected,{})
    assert "error" in e("ls",{})
def test_bool_oracle_and_fake_text_fail(tmp_path):
    e=env(tmp_path);e("read",{"file_path":"b.txt"})
    assert not e.grade({"path":"b.txt","line":True,"text":"verified_target = 42"},ep(e,1))["success"]
    assert not e.grade({"path":"../b.txt","line":2,"text":"verified_target = 42"},ep(e,1))["success"]
def test_clipped_output_not_evidence(tmp_path):
    e=env(tmp_path,max_result_bytes=64)
    r=e("read",{"file_path":"b.txt"})
    assert r["output_limited"] is True and not e.observed
    assert not e.grade(e.expected,ep(e,1))["success"]
def test_fixture_special_entries_rejected(tmp_path):
    e=env(tmp_path);(e.root/"link").symlink_to("/etc/passwd")
    with pytest.raises(ValueError,match="special"):verify_fixture(e.root,e.files)

def test_evidence_requires_delivered_hash(tmp_path):
    e=env(tmp_path);e("grep",{"pattern":"verified_target","literal":True})
    assert not e.grade(e.expected,{"tool_attempts":1,"delivered_tool_result_hashes":[]})["success"]
    assert not e.grade(e.expected,{"tool_attempts":1,"delivered_tool_result_hashes":["0"*64]})["success"]
    h=e.calls[0]["result_sha256"]
    assert e.grade(e.expected,{"tool_attempts":1,"delivered_tool_result_hashes":[h]})["success"]

def test_process_deadline_kills_owned_child(tmp_path):
    root=tmp_path/"fixture";root.mkdir();(root/"a.txt").write_text("x")
    files=[{"relative_path":"a.txt","bytes":1,"sha256":sha(root/"a.txt")}]
    sleep=tmp_path/"sleep.sh";sleep.write_text("#!/bin/sh\nsleep 5\n");sleep.chmod(0o755)
    e=CanonicalToolEnvironment(sleep,sha(sleep),root,files,{"path":"a.txt","line":1,"text":"x"},timeout=.05)
    r=e("read",{"file_path":"a.txt"})
    assert "error" in r and e.calls[0]["wall_seconds"] < 2
