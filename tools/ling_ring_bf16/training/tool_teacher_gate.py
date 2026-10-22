"""Strict held-out teacher gate for real tool-workflow receipts.
AI Identity Incorported <james@aiii.id>.
This computes a limited pilot gate only. It never promotes an adapter.
"""
from __future__ import annotations
import hashlib
import math
import re
from typing import Any, Iterable
import tool_workflow

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED = frozenset({
    "task", "split", "success", "complete", "tool_calls", "tool_attempts",
    "wall_seconds", "grade", "policy_adapter_sha256", "plan_sha256",
})

class TeacherGateError(ValueError):
    pass

def _sha(value: Any, name: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise TeacherGateError(f"{name} must be a lowercase SHA-256 string")
    return value

def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TeacherGateError(f"{name} must be boolean")
    return value

def _count(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TeacherGateError(f"{name} must be a nonnegative integer")
    return value

def _seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TeacherGateError("wall_seconds must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise TeacherGateError("wall_seconds must be finite and positive")
    return value

def _row(row: Any, expected_split: str, teacher_sha: str, plan_sha: str, *, require_teacher: bool, original_sha: str | None = None) -> dict[str, Any]:
    if not isinstance(row, dict) or not _REQUIRED.issubset(row):
        raise TeacherGateError("receipt missing required teacher-gate fields")
    if type(row["task"]) is not str or not row["task"]:
        raise TeacherGateError("task must be a nonempty string")
    if row["split"] != expected_split:
        raise TeacherGateError("receipt split mismatch")
    success = _bool(row["success"], "success")
    complete = _bool(row["complete"], "complete")
    calls = _count(row["tool_calls"], "tool_calls")
    attempts = _count(row["tool_attempts"], "tool_attempts")
    seconds = _seconds(row["wall_seconds"])
    adapter_sha = row["policy_adapter_sha256"]
    if require_teacher:
        _sha(adapter_sha, "policy_adapter_sha256")
        if adapter_sha != teacher_sha:
            raise TeacherGateError("receipt teacher adapter binding mismatch")
    elif adapter_sha is not None:
        _sha(adapter_sha, "policy_adapter_sha256")
        if original_sha is None or adapter_sha != original_sha:
            raise TeacherGateError("baseline adapter binding mismatch")
    if row["plan_sha256"] != plan_sha:
        raise TeacherGateError("receipt plan binding mismatch")
    grade = row["grade"]
    if not isinstance(grade, dict) or type(grade.get("verified_success")) is not bool:
        raise TeacherGateError("grade verified_success must be boolean")
    if bool(grade["verified_success"]) != success:
        raise TeacherGateError("success and verified_success disagree")
    if success and (not complete or calls < 1 or attempts < 1):
        raise TeacherGateError("successful receipt lacks complete actual tool call")
    if attempts < calls:
        raise TeacherGateError("tool_attempts cannot be below tool_calls")
    return {
        "task": row["task"], "success": success, "complete": complete,
        "tool_calls": calls, "tool_attempts": attempts, "seconds": seconds,
    }

def _paired(baseline: Iterable[dict], candidate: Iterable[dict], task_ids: Iterable[str], split: str, teacher_sha: str, plan_sha: str, original_sha: str | None = None):
    expected = list(task_ids)
    if len(expected) != len(set(expected)) or not expected or any(type(x) is not str or not x for x in expected):
        raise TeacherGateError("declared task IDs must be unique nonempty strings")
    b = [_row(x, split, teacher_sha, plan_sha, require_teacher=False, original_sha=original_sha) for x in baseline]
    c = [_row(x, split, teacher_sha, plan_sha, require_teacher=True) for x in candidate]
    if len(b) != len(expected) or len(c) != len(expected):
        raise TeacherGateError("held-out coverage is incomplete")
    bm = {x["task"]: x for x in b}
    cm = {x["task"]: x for x in c}
    if len(bm) != len(b) or len(cm) != len(c) or set(bm) != set(expected) or set(cm) != set(expected):
        raise TeacherGateError("duplicate or omitted held-out task")
    return bm, cm

def _metrics(bm, cm):
    ids = list(bm)
    regressions = [x for x in ids if bm[x]["success"] and not cm[x]["success"]]
    gains = [x for x in ids if not bm[x]["success"] and cm[x]["success"]]
    shared = [x for x in ids if bm[x]["success"] and cm[x]["success"]]
    b_incomplete = sum(not bm[x]["complete"] for x in ids)
    c_incomplete = sum(not cm[x]["complete"] for x in ids)
    b_seconds = sum(bm[x]["seconds"] for x in ids)
    c_seconds = sum(cm[x]["seconds"] for x in ids)
    b_verified = sum(bm[x]["success"] for x in ids)
    c_verified = sum(cm[x]["success"] for x in ids)
    b_vph = b_verified * 3600.0 / b_seconds
    c_vph = c_verified * 3600.0 / c_seconds
    b_attempts = sum(bm[x]["tool_attempts"] for x in shared)
    c_attempts = sum(cm[x]["tool_attempts"] for x in shared)
    attempt_reduction = (1.0 - c_attempts / b_attempts) if b_attempts else 0.0
    gain_vph = (c_vph / b_vph - 1.0) if b_vph else None
    criteria_a = len(gains) >= 2 and c_vph >= b_vph
    criteria_b = gain_vph is not None and gain_vph >= 0.15 and attempt_reduction >= 0.15 and len(shared) >= 12
    passed = not regressions and c_incomplete <= b_incomplete and (criteria_a or criteria_b)
    return {
        "baseline": {"correctness": b_verified, "attempts": sum(bm[x]["tool_attempts"] for x in ids),
                     "seconds": b_seconds, "verified_per_hour": b_vph, "incomplete": b_incomplete},
        "candidate": {"correctness": c_verified, "attempts": sum(cm[x]["tool_attempts"] for x in ids),
                      "seconds": c_seconds, "verified_per_hour": c_vph, "incomplete": c_incomplete},
        "paired_successes": shared, "paired_gains": gains, "paired_losses": regressions,
        "attempt_reduction_on_paired_successes": attempt_reduction,
        "verified_per_hour_change": gain_vph,
        "criteria_newly_correct": criteria_a, "criteria_efficiency": criteria_b,
        "passed": passed,
    }

def teacher_gate(development_baseline, development_candidate, heldout_baseline, heldout_candidate,
                 *, declared_task_ids, teacher_adapter_sha256, plan_sha256,
                 declared_selection_ids=None, original_adapter_sha256=None):
    teacher_sha = _sha(teacher_adapter_sha256, "teacher_adapter_sha256")
    plan_sha = _sha(plan_sha256, "plan_sha256")
    if declared_selection_ids is None:
        raise TeacherGateError("declared selection IDs required")
    selection_ids = list(declared_selection_ids)
    if len(selection_ids) != len(set(selection_ids)) or not selection_ids:
        raise TeacherGateError("declared selection IDs must be unique")
    if original_adapter_sha256 is not None:
        _sha(original_adapter_sha256, "original_adapter_sha256")
    _paired(development_baseline, development_candidate, selection_ids,
            "teacher_selection", teacher_sha, plan_sha, original_adapter_sha256)
    # Development is intentionally delegated to the existing development-only gate.
    development = tool_workflow.selection_gate(development_baseline, development_candidate)
    bm, cm = _paired(heldout_baseline, heldout_candidate, declared_task_ids,
                      "held_out", teacher_sha, plan_sha, original_adapter_sha256)
    heldout = _metrics(bm, cm)
    qualified = bool(development["select"] and heldout["passed"])
    return {
        "schema": "strata-tool-teacher-gate-v1",
        "status": "limited-pilot-qualified" if qualified else "limited-pilot-rejected",
        "promotion_allowed": False, "promoted": False,
        "teacher_adapter_sha256": teacher_sha, "plan_sha256": plan_sha,
        "development_selection": development,
        "held_out": heldout,
        "scope": "sealed teacher comparison on declared held-out task IDs; no promotion",
    }

