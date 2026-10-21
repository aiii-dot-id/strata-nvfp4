"""CPU evidence tests for the actual r3 compare_actions.py runner."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
import compare_actions as runner


def _records():
    return [
        {"case_id":"a","target_ids":[101,102]},
        {"case_id":"b","target_ids":[103]},
        {"case_id":"c","target_ids":[104,105,106]},
    ]


def test_weights_failure_two_success_one_normalize_same_budget():
    records=_records(); failures={"a":True,"b":True,"c":False}
    for arm in ("S","C"):
        w=runner.weights_for(records,failures,arm)
        assert len(w)==6 and torch_isclose(float(w.sum()),6.0)
    s=runner.weights_for(records,failures,"S")
    c=runner.weights_for(records,failures,"C")
    assert torch_isclose(float(c[0]),2.0*6.0/9.0)
    assert torch_isclose(float(c[2]),2.0*6.0/9.0)
    assert torch_isclose(float(c[3]),6.0/9.0)


def torch_isclose(a,b): return abs(a-b)<1e-6


def test_target_rows_are_exact_terminal_suffix_and_reject_bad_mask():
    record={"prompt_ids":[10,11,12],"target_ids":[20,21],"target_row_indices":[2,3],"target_boundary_verified":True}
    tokens,rows,targets=runner.rows_and_targets(record)
    assert tokens==[10,11,12,20] and rows==[2,3] and targets==[20,21]
    for change in ({"target_row_indices":[1,3]}, {"target_boundary_verified":False}):
        bad=dict(record);bad.update(change)
        with pytest.raises(ValueError): runner.rows_and_targets(bad)


def test_trained_forward_parity_gate_accepts_tolerance_and_rejects_argmax_change():
    expected=np.array([[0.,2.,1.]],dtype=np.float32)
    observed=expected.copy();observed[0,0]=.0005
    assert runner.qualify.compare_logits(observed,expected)["passed"] is True
    changed=expected.copy();changed[0,1]=0.;changed[0,2]=3.
    assert runner.qualify.compare_logits(changed,expected)["passed"] is False


def test_compaction_retains_selected_rows_and_removes_hashed_transients(tmp_path):
    n=3; hidden=runner.native.MODELS["ling"]["hidden"]
    widths={"logits.f32":157184,"terminal.f32":11*hidden+8208}
    payloads={}
    for name,width in widths.items():
        path=tmp_path/name; data=np.arange(n*width,dtype=np.float32); path.write_bytes(data.astype("<f4").tobytes()); payloads[name]={"sha256":runner.native.sha(path)}
    (tmp_path/"RUN.json").write_text(json.dumps({"model":"ling","execution":{"tokens":n},"native_payloads":payloads}))
    runner.compact(tmp_path,[0,2])
    for name,width in widths.items():
        assert not (tmp_path/name).exists()
        got=np.fromfile(tmp_path/('selected-'+name),dtype="<f4").reshape(2,width)
        original=np.arange(n*width,dtype=np.float32).reshape(n,width)
        assert np.array_equal(got,original[[0,2]])
    retention=json.loads((tmp_path/"RETENTION.json").read_text())
    assert set(retention["removed_transient_payloads"])==set(widths)
    assert all(x["positions"]==[0,2] for x in retention["retained"].values())


def test_runner_freezes_candidates_before_heldout_and_compacts_eval():
    source=Path(runner.__file__).read_text()
    assert source.index("FROZEN-ADAPTERS.json") < source.index("for i,r in enumerate(held+retention)")
    assert source.index("candidate_digests") < source.index("for i,r in enumerate(held+retention)")
    eval_start=source.index("for i,r in enumerate(held+retention)")
    assert source.index("compact(d,list(range",eval_start) < source.index("if any(native.sha(p)!=v",eval_start)
