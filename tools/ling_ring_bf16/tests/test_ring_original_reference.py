import gc
import json
import shutil
from types import SimpleNamespace
import struct
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import ring_original_reference as reference


def write_checkpoint(root: Path, payload=b"abcd"):
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"vocab_size": 8}))
    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, separators=(",", ":")).encode()
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model.safetensors"}}))
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)


def test_shard_pin_rereads_payload(tmp_path):
    root = tmp_path / "checkpoint"
    write_checkpoint(root)
    index = reference.SafeTensorIndex(root)
    before = index.pin()
    (root / "model.safetensors").write_bytes(
        (root / "model.safetensors").read_bytes()[:-1] + b"Z"
    )
    after = index.pin()
    assert before["shards"]["model.safetensors"]["sha256"] != after["shards"]["model.safetensors"]["sha256"]


def test_runtime_pins_bind_runner_source_and_config(tmp_path):
    root = tmp_path / "checkpoint"
    write_checkpoint(root)
    source = tmp_path / "modeling_bailing_moe_v2.py"
    config = tmp_path / "configuration_bailing_moe_v2.py"
    source.write_text("source")
    config.write_text("config")
    pins = reference.runtime_pins(root, source)
    assert {"runner", "modeling", "configuration", "config"} == set(pins)
    source.write_text("changed")
    assert pins["modeling"]["sha256"] != reference.runtime_pins(root, source)["modeling"]["sha256"]



def test_reference_run_reduced_checkpoint_all_and_last(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    source = Path("/work/AI/checkpoints/Ring-mini-2.0/modeling_bailing_moe_v2.py")
    config_source = source.parent / "configuration_bailing_moe_v2.py"
    if not source.is_file() or not config_source.is_file():
        pytest.skip("authoritative Ring source is unavailable")
    module = reference.load_original(source)
    config = module.BailingMoeV2Config(
        vocab_size=32, hidden_size=8, intermediate_size=12,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=4, max_position_embeddings=16, rope_theta=100.0,
        pad_token_id=31, eos_token_id=31, num_experts=4,
        num_experts_per_tok=2, n_group=2, topk_group=1,
        moe_intermediate_size=4, first_k_dense_replace=1,
        routed_scaling_factor=2.5, torch_dtype="bfloat16",
    )
    config._attn_implementation = "eager"
    torch.manual_seed(17)
    oracle = module.BailingMoeV2ForCausalLM(config).eval()
    state = {}
    for name, value in oracle.state_dict().items():
        state[name] = value.float() if name.endswith("mlp.gate.expert_bias") else value.to(torch.bfloat16)
    # The oracle uses the exact saved BF16 values widened to FP32, matching
    # reference.run's FP32 working domain rather than native BF16 activation math.
    oracle.load_state_dict({name: value.float() for name, value in state.items()}, assign=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(config.to_dict()))
    (checkpoint / "modeling_bailing_moe_v2.py").write_bytes(source.read_bytes())
    (checkpoint / "configuration_bailing_moe_v2.py").write_bytes(config_source.read_bytes())
    tensors = {name: value.cpu().contiguous() for name, value in state.items()}
    safetensors_torch.save_file(tensors, str(checkpoint / "model.safetensors"))
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {},
        "weight_map": {name: "model.safetensors" for name in tensors},
    }))
    tokens = [1, 2, 3, 4]
    outputs = {}
    for mode in ("all", "last"):
        args = SimpleNamespace(
            checkpoint=checkpoint,
            source=checkpoint / "modeling_bailing_moe_v2.py",
            tokens=",".join(map(str, tokens)),
            output=tmp_path / ("out-" + mode),
            threads=1,
            device="cpu",
            logits_mode=mode,
        )
        outputs[mode] = reference.run(args)
        assert outputs[mode]["source"]["sha256"] == reference.sha(args.source)
        assert outputs[mode]["source_details"]["runner"]["sha256"] == reference.sha(Path(reference.__file__).resolve())
        assert outputs[mode]["checkpoint"]["shards"]["model.safetensors"]["sha256"] == reference.sha(checkpoint / "model.safetensors")
    ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        expected = oracle(ids, use_cache=False, return_dict=True).logits.float()
    all_logits = torch.frombuffer(bytearray((tmp_path / "out-all" / "logits.fp32").read_bytes()), dtype=torch.float32).reshape(1, 4, 32)
    last_logits = torch.frombuffer(bytearray((tmp_path / "out-last" / "logits.fp32").read_bytes()), dtype=torch.float32).reshape(1, 1, 32)
    assert torch.allclose(all_logits, expected, atol=1e-5, rtol=1e-4)
    assert torch.allclose(last_logits, expected[:, -1:], atol=1e-5, rtol=1e-4)
    assert outputs["all"]["logits_mode"] == "all"
    assert outputs["last"]["logits_mode"] == "last"
    del oracle, module
    gc.collect()

def test_original_two_layer_reduced_geometry_prefill_incremental_and_logits_modes():
    torch = pytest.importorskip("torch")
    source = Path("/work/AI/checkpoints/Ring-mini-2.0/modeling_bailing_moe_v2.py")
    if not source.is_file():
        pytest.skip("authoritative Ring source is unavailable")
    module = reference.load_original(source)
    config = module.BailingMoeV2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        rope_theta=100.0,
        pad_token_id=31,
        eos_token_id=31,
        num_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        moe_intermediate_size=4,
        first_k_dense_replace=1,
        routed_scaling_factor=2.5,
    )
    config._attn_implementation = "eager"
    model = module.BailingMoeV2ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        prefill = model(ids[:, :3], use_cache=True, return_dict=True)
        incremental = model(ids[:, 3:], past_key_values=prefill.past_key_values,
                            use_cache=True, return_dict=True)
        full = model(ids, use_cache=False, return_dict=True)
    assert prefill.logits.shape == (1, 3, 32)
    assert incremental.logits.shape == (1, 1, 32)
    assert full.logits.shape == (1, 4, 32)
    assert torch.isfinite(prefill.logits).all()
    assert torch.isfinite(incremental.logits).all()
    assert torch.isfinite(full.logits).all()
    assert torch.allclose(incremental.logits[:, -1], full.logits[:, -1], atol=1e-5, rtol=1e-4)
    assert incremental.past_key_values is not None
    # The runner's all/last contract is all positions versus only the final row.
    assert reference.select_logits_hidden(full.logits, "all").shape == (1, 4, 32)
    assert reference.select_logits_hidden(full.logits, "last").shape == (1, 1, 32)
    del incremental, prefill, full, model, module
    gc.collect()
