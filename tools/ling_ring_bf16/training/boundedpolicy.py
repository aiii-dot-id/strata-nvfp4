"""Strict opt-in bounded update policy parsing.
AI Identity Incorported <james@aiii.id>.
"""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path

_KEYS = frozenset({"learning_rates", "max_mean_kl", "max_max_kl", "max_selected_log_ratio"})
_RATE_KEYS = frozenset({"rlvr", "self_opd"})

def _unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError("duplicate bounded policy key")
        out[key] = value
    return out

def _finite(x, *, positive=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError("bounded policy values must be numeric")
    x = float(x)
    if not math.isfinite(x) or (x <= 0 if positive else x < 0):
        raise ValueError("bounded policy values out of range")
    return x

def parse_policy(value):
    if isinstance(value, (str, Path)):
        raw = Path(value).read_bytes()
        try: obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs)
        except Exception as exc: raise ValueError("invalid bounded policy JSON") from exc
    elif isinstance(value, dict):
        raw = None
        obj = value
    else:
        raise ValueError("bounded policy must be a JSON object or path")
    if not isinstance(obj, dict) or set(obj) != _KEYS:
        raise ValueError("bounded policy keys must be exactly learning_rates,max_mean_kl,max_max_kl,max_selected_log_ratio")
    rates = obj["learning_rates"]
    if not isinstance(rates, dict) or set(rates) != _RATE_KEYS:
        raise ValueError("learning_rates keys must be exactly rlvr,self_opd")
    out = {
        "learning_rates": {"rlvr": _finite(rates["rlvr"], positive=True), "self_opd": _finite(rates["self_opd"], positive=True)},
        "max_mean_kl": _finite(obj["max_mean_kl"]),
        "max_max_kl": _finite(obj["max_max_kl"]),
        "max_selected_log_ratio": _finite(obj["max_selected_log_ratio"]),
    }
    return out

def canonical_bytes(policy):
    return (json.dumps(parse_policy(policy), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()

def policy_sha256(policy):
    return hashlib.sha256(canonical_bytes(policy)).hexdigest()

def bounds(policy, chunk_rows):
    from bounded_update import Bounds
    p = parse_policy(policy)
    return Bounds(p["max_mean_kl"], p["max_max_kl"], p["max_selected_log_ratio"], chunk_rows)

def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value

def receipt(policy, *, arm, round_no, steps_before, steps_after, current_adapter_sha256, batch_sha256, decision):
    p = parse_policy(policy)
    return {
        "schema": "strata-bounded-update-receipt-v1",
        "accepted": bool(decision["accepted"]),
        "arm": arm, "round": int(round_no),
        "steps_before": int(steps_before), "steps_after": int(steps_after),
        "trial_lr": p["learning_rates"][arm],
        "policy": p, "policy_sha256": policy_sha256(p),
        "current_adapter_sha256": current_adapter_sha256,
        "batch_sha256": batch_sha256,
        "reason": decision.get("reason"),
        "metrics": _json_safe(decision.get("metrics")),
        "bounds": decision.get("bounds"),
        "restored": bool(decision.get("restored", False)),
    }
