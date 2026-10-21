import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pytest
import boundedpolicy

def good():
    return {"learning_rates":{"rlvr":0.001,"self_opd":0.0003},
            "max_mean_kl":0.01,"max_max_kl":0.05,"max_selected_log_ratio":0.1}

def test_duplicate_json_keys_rejected():
    path=Path("/tmp/bounded-policy-duplicate.json")
    path.write_text('{"learning_rates":{"rlvr":0.001,"self_opd":0.0003},"max_mean_kl":0,"max_max_kl":0,"max_selected_log_ratio":0,"max_mean_kl":0}')
    with pytest.raises(ValueError):
        boundedpolicy.parse_policy(path)

def test_policy_exact_schema_and_canonical_digest():
    p=boundedpolicy.parse_policy(good())
    assert p["learning_rates"]["rlvr"]==0.001
    assert len(boundedpolicy.policy_sha256(p))==64
    path=Path("/tmp/bounded-policy.json"); path.write_text(json.dumps(good()))
    assert boundedpolicy.parse_policy(path)==p

@pytest.mark.parametrize("bad",[
    {},
    {"learning_rates":{"rlvr":1,"self_opd":1},"max_mean_kl":0,"max_max_kl":0,"max_selected_log_ratio":0,"extra":1},
    {"learning_rates":{"rlvr":1},"max_mean_kl":0,"max_max_kl":0,"max_selected_log_ratio":0},
    {"learning_rates":{"rlvr":0,"self_opd":1},"max_mean_kl":0,"max_max_kl":0,"max_selected_log_ratio":0},
    {"learning_rates":{"rlvr":1,"self_opd":1},"max_mean_kl":-1,"max_max_kl":0,"max_selected_log_ratio":0},
    {"learning_rates":{"rlvr":True,"self_opd":1},"max_mean_kl":0,"max_max_kl":0,"max_selected_log_ratio":0},
])
def test_policy_rejects_malformed(bad):
    with pytest.raises(ValueError): boundedpolicy.parse_policy(bad)

def test_receipt_preserves_rejection_step_count_and_policy_binding():
    p=good()
    r=boundedpolicy.receipt(p,arm="rlvr",round_no=2,steps_before=1,steps_after=1,
        current_adapter_sha256="a"*64,batch_sha256="b"*64,
        decision={"accepted":False,"reason":"mean KL bound exceeded",
                  "metrics":{"mean_kl":1.0},"bounds":{"max_mean_kl":0.01},"restored":True})
    assert not r["accepted"] and r["steps_before"]==r["steps_after"]==1
    assert r["policy_sha256"]==boundedpolicy.policy_sha256(p)
    assert r["reason"]=="mean KL bound exceeded"
