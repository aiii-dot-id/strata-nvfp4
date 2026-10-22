"""CPU tests for the independent held-out tool teacher gate.
AI Identity Incorported <james@aiii.id>.
"""
import copy
import math
import pytest
import tool_teacher_gate as g

TEACHER = "a" * 64
PLAN = "b" * 64

def row(task, split, success, *, adapter=None, seconds=1.0, attempts=1, calls=1, complete=True):
    return {"task": task, "split": split, "success": success, "complete": complete,
            "tool_calls": calls, "tool_attempts": attempts, "wall_seconds": seconds,
            "policy_adapter_sha256": adapter, "plan_sha256": PLAN,
            "grade": {"verified_success": success}}

def development():
    b = [row("d0", "teacher_selection", False, adapter=None),
         row("d1", "teacher_selection", False, adapter=None)]
    c = [row("d0", "teacher_selection", True, adapter=TEACHER),
         row("d1", "teacher_selection", True, adapter=TEACHER)]
    return b, c

def heldout():
    b = [row(f"h{i}", "held_out", i >= 2, adapter=None, attempts=2) for i in range(4)]
    c = [row(f"h{i}", "held_out", True, adapter=TEACHER, attempts=1) for i in range(4)]
    return b, c

def test_limited_pilot_gate_requires_development_and_heldout_evidence():
    db, dc = development(); hb, hc = heldout()
    result = g.teacher_gate(db, dc, hb, hc, declared_task_ids=["h0","h1","h2","h3"],
                            teacher_adapter_sha256=TEACHER, plan_sha256=PLAN, declared_selection_ids=["d0","d1"])
    assert result["status"] == "limited-pilot-qualified"
    assert result["promotion_allowed"] is False and result["promoted"] is False
    assert result["held_out"]["paired_gains"] == ["h0", "h1"]
    assert result["held_out"]["paired_losses"] == []

def test_baseline_adapter_may_be_none_but_candidate_must_bind_teacher():
    db, dc = development(); hb, hc = heldout()
    hc[0]["policy_adapter_sha256"] = "c" * 64
    with pytest.raises(g.TeacherGateError, match="teacher adapter"):
        g.teacher_gate(db, dc, hb, hc, declared_task_ids=["h0","h1","h2","h3"],
                       teacher_adapter_sha256=TEACHER, plan_sha256=PLAN, declared_selection_ids=["d0","d1"])

@pytest.mark.parametrize("mutator, message", [
    (lambda b,c: b.pop(), "coverage"),
    (lambda b,c: b.append(copy.deepcopy(b[0])), "coverage"),
    (lambda b,c: b[0].update(split="retention"), "split"),
    (lambda b,c: b[0].update(tool_attempts=True), "integer"),
    (lambda b,c: b[0].update(wall_seconds=float("nan")), "finite"),
    (lambda b,c: b[0].update(success=True, complete=False, grade={"verified_success": True}), "successful"),
])
def test_invalid_heldout_receipts_rejected(mutator, message):
    db, dc = development(); hb, hc = heldout()
    mutator(hb, hc)
    with pytest.raises(g.TeacherGateError, match=message):
        g.teacher_gate(db, dc, hb, hc, declared_task_ids=["h0","h1","h2","h3"],
                       teacher_adapter_sha256=TEACHER, plan_sha256=PLAN, declared_selection_ids=["d0","d1"])

def test_plan_hash_and_declared_ids_are_strict():
    db, dc = development(); hb, hc = heldout()
    hc[0]["plan_sha256"] = "wrong"
    with pytest.raises(g.TeacherGateError, match="plan"):
        g.teacher_gate(db, dc, hb, hc, declared_task_ids=["h0","h1","h2","h3"],
                       teacher_adapter_sha256=TEACHER, plan_sha256=PLAN, declared_selection_ids=["d0","d1"])
    hb, hc = heldout()
    with pytest.raises(g.TeacherGateError, match="coverage"):
        g.teacher_gate(db, dc, hb, hc, declared_task_ids=["h0","h1","h2"],
                       teacher_adapter_sha256=TEACHER, plan_sha256=PLAN, declared_selection_ids=["d0","d1"])

def test_baseline_correct_regression_and_more_incomplete_reject():
    db, dc = development(); hb, hc = heldout()
    hc[2].update(success=False, complete=False, grade={"verified_success": False}, tool_calls=0, tool_attempts=0)
    result = g.teacher_gate(db, dc, hb, hc, declared_task_ids=["h0","h1","h2","h3"],
                            teacher_adapter_sha256=TEACHER, plan_sha256=PLAN, declared_selection_ids=["d0","d1"])
    assert result["status"] == "limited-pilot-rejected"
    assert "h2" in result["held_out"]["paired_losses"]

def test_efficiency_alternative_requires_twelve_shared_successes():
    ids = [f"h{i}" for i in range(12)]
    hb = [row(x, "held_out", True, adapter=None, seconds=4.0, attempts=4) for x in ids]
    hc = [row(x, "held_out", True, adapter=TEACHER, seconds=1.0, attempts=3) for x in ids]
    db, dc = development()
    result = g.teacher_gate(db, dc, hb, hc, declared_task_ids=ids,
                            teacher_adapter_sha256=TEACHER, plan_sha256=PLAN, declared_selection_ids=["d0","d1"])
    assert result["held_out"]["criteria_efficiency"] is True

def test_hash_shape_rejected_without_claiming_hash_proof():
    db, dc = development(); hb, hc = heldout()
    with pytest.raises(g.TeacherGateError):
        g.teacher_gate(db, dc, hb, hc, declared_task_ids=["h0","h1","h2","h3"],
                       teacher_adapter_sha256="not-a-sha", plan_sha256=PLAN)
    with pytest.raises(g.TeacherGateError):
        g.teacher_gate(db, dc, hb, hc, declared_task_ids=["h0","h1","h2","h3"],
                       teacher_adapter_sha256=TEACHER, plan_sha256="not-a-sha")


def test_zero_to_two_correct_result_is_json_serializable_without_infinity():
    db, dc = development()
    hb = [row(f"h{i}", "held_out", False, adapter=None) for i in range(4)]
    hc = [row(f"h{i}", "held_out", True, adapter=TEACHER) for i in range(4)]
    result = g.teacher_gate(db, dc, hb, hc, declared_task_ids=["h0","h1","h2","h3"],
                            teacher_adapter_sha256=TEACHER, plan_sha256=PLAN,
                            declared_selection_ids=["d0","d1"])
    import json
    json.dumps(result, allow_nan=False)
    assert result["held_out"]["verified_per_hour_change"] is None
    assert result["held_out"]["criteria_newly_correct"] is True

def test_nonzero_baseline_adapter_requires_explicit_original_pin():
    db, dc = development(); hb, hc = heldout()
    hb[0]["policy_adapter_sha256"] = "c" * 64
    with pytest.raises(g.TeacherGateError, match="baseline adapter"):
        g.teacher_gate(db, dc, hb, hc, declared_task_ids=["h0","h1","h2","h3"],
                       teacher_adapter_sha256=TEACHER, plan_sha256=PLAN,
                       declared_selection_ids=["d0","d1"])
    result = g.teacher_gate(db, dc, hb, hc, declared_task_ids=["h0","h1","h2","h3"],
                            teacher_adapter_sha256=TEACHER, plan_sha256=PLAN,
                            declared_selection_ids=["d0","d1"], original_adapter_sha256="c"*64)
    assert result["held_out"]["baseline"]["correctness"] == 2

def test_omitted_development_task_is_rejected_against_declared_ids():
    db, dc = development(); hb, hc = heldout()
    with pytest.raises(g.TeacherGateError, match="coverage"):
        g.teacher_gate(db, dc, hb, hc, declared_task_ids=["h0","h1","h2","h3"],
                       teacher_adapter_sha256=TEACHER, plan_sha256=PLAN,
                       declared_selection_ids=["d0","d1","d2"])


def test_development_teacher_and_plan_bindings_are_strict():
    db, dc = development(); hb, hc = heldout()
    dc[0]["policy_adapter_sha256"] = "c" * 64
    with pytest.raises(g.TeacherGateError, match="teacher adapter"):
        g.teacher_gate(db, dc, hb, hc, declared_task_ids=["h0","h1","h2","h3"],
                       teacher_adapter_sha256=TEACHER, plan_sha256=PLAN,
                       declared_selection_ids=["d0","d1"])
    db, dc = development(); hb, hc = heldout()
    dc[0]["plan_sha256"] = "c" * 64
    with pytest.raises(g.TeacherGateError, match="plan"):
        g.teacher_gate(db, dc, hb, hc, declared_task_ids=["h0","h1","h2","h3"],
                       teacher_adapter_sha256=TEACHER, plan_sha256=PLAN,
                       declared_selection_ids=["d0","d1"])
