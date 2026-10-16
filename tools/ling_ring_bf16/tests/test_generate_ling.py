import json
from pathlib import Path
import pytest
import generate_ling as g

def write(p,v): p.write_text(json.dumps(v)+'\n')
def trace(tmp, generated, reason='eos', maximum=3, eos=[9]):
    p=tmp/'generation.json'
    write(p,{'schema':'strata-ling-generation-v1','prompt_token_ids':[1,2],'generated_token_ids':generated,'eos_token_ids':eos,'max_new_tokens':maximum,'finish_reason':reason})
    return p

def test_generation_eos_and_length(tmp_path):
    assert g._require_generation(trace(tmp_path,[3,9]),[1,2],3,[9])['finish_reason']=='eos'
    assert g._require_generation(trace(tmp_path,[3,4,5],reason='length',maximum=3),[1,2],3,[9])['finish_reason']=='length'

@pytest.mark.parametrize('generated,reason', [([3,9,4],'eos'),([3],'eos'),([3,9],'length')])
def test_generation_rejects_bad_terminal_state(tmp_path,generated,reason):
    with pytest.raises(g.VerificationError):
        g._require_generation(trace(tmp_path,generated,reason=reason,maximum=3),[1,2],3,[9])

def test_generation_requires_eos_and_bound(tmp_path):
    with pytest.raises(g.VerificationError):
        g.run_verified(Path('/missing'),Path('/missing'),Path('/missing'),'1',tmp_path/'out',5,generate=1024,eos_ids=[9])
    with pytest.raises(g.VerificationError):
        g.run_verified(Path('/missing'),Path('/missing'),Path('/missing'),'1',tmp_path/'out2',5,generate=1,eos_ids=[])

def test_generation_rejects_boolean_budget_and_token(tmp_path):
    p=trace(tmp_path,[3,9])
    value=json.loads(p.read_text())
    value["max_new_tokens"]=True
    p.write_text(json.dumps(value)+"\n")
    with pytest.raises(g.VerificationError):
        g._require_generation(p,[1,2],3,[9])
    value["max_new_tokens"]=3
    value["generated_token_ids"]=[3,True]
    p.write_text(json.dumps(value)+"\n")
    with pytest.raises(g.VerificationError):
        g._require_generation(p,[1,2],3,[9])

def test_eos_resolution_override_then_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 156895}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"temperature": 0.7}))
    assert g._eos_ids(tmp_path) == ([156895], "config.json")
    assert g._eos_ids(tmp_path, [9]) == ([9], "explicit")
    (tmp_path / "config.json").write_text(json.dumps({}))
    with pytest.raises(g.VerificationError, match="no eos_token_id"):
        g._eos_ids(tmp_path)

def test_binding_includes_chat_template(tmp_path):
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
        (tmp_path / name).write_text("{}")
    assert "chat_template.jinja" in {p.name for p in g._binding_paths(tmp_path)}

def test_real_ling_tokenizer_template_dryrun():
    ck = Path(__import__("os").environ.get("LING_CHECKPOINT", "/work/AI/checkpoints/Ling-3.0-tiny"))
    if not ck.is_dir():
        pytest.skip("Ling checkpoint fixture absent")
    pytest.importorskip("transformers")
    tokenizer = g._tokenizer(ck)
    ids = g._ids(tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=True, add_generation_prompt=True))
    assert ids and all(type(x) is int for x in ids)
