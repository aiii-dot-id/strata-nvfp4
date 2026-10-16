import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_ring_manifest import EXPECTED, prepare

def make(root, mutate=None):
    c=dict(EXPECTED); c.update({"architectures":["BailingMoeV2ForCausalLM"]})
    if mutate: c.update(mutate)
    (root/"config.json").write_text(json.dumps(c))
    h=json.dumps({"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}).encode()
    (root/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{"x":"model.safetensors"}}))
    (root/"model.safetensors").write_bytes(len(h).to_bytes(8,"little")+h+b"abcd")

def test_prepare(tmp_path):
    make(tmp_path); out=tmp_path/"out"; r=prepare(tmp_path,out)
    row=(out/"manifest.tsv").read_text().strip().split("\t")
    assert row[:4]==["x","BF16","2","model.safetensors"] and row[5:]==["4","74"] and int(row[4]) > 8
    assert json.loads((out/"admission.json").read_text())["manifest_sha256"]==r["manifest_sha256"]

def test_geometry_rejected_before_payload_read(tmp_path):
    make(tmp_path,{"hidden_size":2049})
    try: prepare(tmp_path,tmp_path/"out")
    except ValueError as e: assert "hidden_size" in str(e)
    else: assert False

def test_every_critical_config_field_is_guarded(tmp_path):
    for key, value in EXPECTED.items():
        mutated = {key: ("wrong" if isinstance(value, str) else (value + 1 if isinstance(value, (int, float)) and not isinstance(value, bool) else not value))}
        make(tmp_path, mutated)
        try: prepare(tmp_path, tmp_path / ("out-" + key))
        except ValueError: pass
        else: assert False, key

def test_boolean_numeric_coercion_rejected(tmp_path):
    make(tmp_path, {"use_qk_norm": 1})
    try: prepare(tmp_path, tmp_path / "out")
    except ValueError: pass
    else: assert False
