import json
import struct
import torch
import pytest
import hashlib
from safetensors.torch import load_file, save_file

from terminal import HIDDEN, TerminalTrainer, load_adapter, load_features, save_adapter, save_training_state, restore_training_state


def _trainer(model="ling"):
    h=HIDDEN[model]
    return TerminalTrainer(torch.randn(512,h),torch.randn(512,h),torch.randn(h,512),model,2,8,16)


def _features(model="ling", rows=2):
    h=HIDDEN[model]; x=torch.randn(rows,11*h+8208); x[:,3*h:3*h+8]=torch.tensor([[2,128,7,9,11,13,15,17]][:rows],dtype=torch.float32); x[:,3*h+8:3*h+16]=torch.softmax(torch.randn(rows,8),1); return x


def test_forward_and_b_gradients():
    torch.manual_seed(7); t=_trainer(); f=_features(); out=t(f); assert out.shape==(2,1536)
    out.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in t.parameters())
    assert t.gate_b.grad.abs().sum()>0 and t.down_b.grad.abs().sum()>0

def test_expert_id_is_not_slot_and_absent_is_passthrough():
    t=_trainer(); f=_features(rows=1); t.expert=128; assert t(f).shape==(1,1536)
    t.expert=127; out=t(f); out.square().mean().backward(); assert all(torch.isfinite(p.grad).all() for p in t.parameters())

def test_mixed_batch_only_exposed_rows_train():
    t=_trainer(); f=_features(rows=2); f[1,3*HIDDEN["ling"]:3*HIDDEN["ling"]+8]=torch.tensor([128,7,9,11,13,15,17,19])
    t(f).square().mean().backward(); assert t.gate_b.grad.abs().sum()>0


def test_capture_layout(tmp_path):
    f=_features("ring",1); (tmp_path/"terminal.f32").write_bytes(f.numpy().astype("<f4").tobytes()); (tmp_path/"terminal-meta.json").write_text(json.dumps({"schema":"strata-terminal-v1","model":"ring","hidden":2048,"intermediate":512,"topk":8,"rows":1,"rowfloats":11*2048+8208}))
    c=load_features(tmp_path); assert c.rows.shape==(1,11*2048+8208); assert c.data_hash


def test_adapter_roundtrip_and_tamper(tmp_path):
    a=_trainer(); path=tmp_path/"a.lora"; save_adapter(a,path,"ckpt")
    b=_trainer(); load_adapter(b,path)
    for x,y in zip(a.parameters(),b.parameters()): assert torch.equal(x,y)
    raw=bytearray(path.read_bytes()); raw[30]^=1; path.write_bytes(raw)
    try: load_adapter(b,path); assert False
    except ValueError: pass


def test_snapshot_restores_optimizer_and_rng(tmp_path):
    torch.manual_seed(3); a=_trainer(); opt=torch.optim.Adam(a.parameters(),lr=.001); loss=a(_features()).square().mean(); loss.backward(); opt.step(); save_training_state(a,opt,tmp_path,step=1,iteration_cursor=4,data_hash="d",config_binding={"x":1})
    b=_trainer(); ob=torch.optim.Adam(b.parameters(),lr=.001); meta=restore_training_state(b,ob,tmp_path,data_hash="d",config_binding={"x":1}); assert meta["iteration_cursor"]==4
    for x,y in zip(a.parameters(),b.parameters()): assert torch.equal(x,y)


def _update(trainer, optimizer, features):
    optimizer.zero_grad(set_to_none=True)
    trainer(features).square().mean().backward()
    optimizer.step()


def _optimizer_snapshot(optimizer):
    return {k: {name: value.detach().clone() for name, value in state.items()}
            for k, state in optimizer.state_dict()["state"].items()}


def test_three_updates_uninterrupted_equals_resume_and_rng(tmp_path):
    torch.manual_seed(101)
    features=[_features(rows=1) for _ in range(3)]
    uninterrupted=_trainer(); opt_u=torch.optim.Adam(uninterrupted.parameters(),lr=.001)
    _update(uninterrupted,opt_u,features[0])
    save_training_state(uninterrupted,opt_u,tmp_path,step=1,iteration_cursor=1,data_hash="rows",config_binding={"model":"ling"})
    expected_random=torch.randn(9)
    _update(uninterrupted,opt_u,features[1]); _update(uninterrupted,opt_u,features[2])
    expected_params=[p.detach().clone() for p in uninterrupted.parameters()]
    expected_slots=_optimizer_snapshot(opt_u)
    resumed=_trainer(); opt_r=torch.optim.Adam(resumed.parameters(),lr=.001)
    for name in ("gate", "up", "down"):
        getattr(resumed,name).copy_(getattr(uninterrupted,name))
    restore_training_state(resumed,opt_r,tmp_path,data_hash="rows",config_binding={"model":"ling"})
    assert torch.equal(torch.randn(9),expected_random)
    _update(resumed,opt_r,features[1]); _update(resumed,opt_r,features[2])
    for got,want in zip(resumed.parameters(),expected_params): assert torch.equal(got,want)
    got_slots=_optimizer_snapshot(opt_r)
    assert got_slots.keys()==expected_slots.keys()
    for key in expected_slots:
        assert got_slots[key].keys()==expected_slots[key].keys()
        for name in expected_slots[key]: assert torch.equal(got_slots[key][name],expected_slots[key][name])


def test_finite_difference_selected_b_gradient():
    torch.manual_seed(202); t=_trainer(); f=_features(rows=1)
    t.zero_grad(set_to_none=True); value=t(f)[0,:8].sum(); value.backward()
    parameter=t.gate_b; index=(0,0); analytic=parameter.grad[index].item(); original=parameter.data[index].item(); eps=1e-3
    with torch.no_grad(): parameter[index]=original+eps
    plus=t(f)[0,:8].sum().item()
    with torch.no_grad(): parameter[index]=original-eps
    minus=t(f)[0,:8].sum().item()
    with torch.no_grad(): parameter[index]=original
    numerical=(plus-minus)/(2*eps)
    assert abs(analytic-numerical) <= 2e-3*max(1.0,abs(analytic),abs(numerical))

def test_snapshot_rejects_missing_key_even_with_recomputed_digest(tmp_path):
    a=_trainer(); opt=torch.optim.Adam(a.parameters(),lr=.001); z=a(_features()).square().mean(); z.backward(); opt.step()
    save_training_state(a,opt,tmp_path,step=1,iteration_cursor=1,data_hash="d",config_binding={"x":1})
    tensors=load_file(str(tmp_path/"tensors.safetensors")); tensors.pop("optim.0.exp_avg"); save_file(tensors,str(tmp_path/"tensors.safetensors"),metadata={"author":"AI Identity Incorported <james@aiii.id>"})
    meta=json.loads((tmp_path/"state.json").read_text()); meta["payload_sha256"]=hashlib.sha256((tmp_path/"tensors.safetensors").read_bytes()).hexdigest(); (tmp_path/"state.json").write_text(json.dumps(meta))
    try: restore_training_state(_trainer(),torch.optim.Adam(_trainer().parameters(),lr=.001),tmp_path,data_hash="d",config_binding={"x":1}); assert False
    except ValueError: pass


def test_adapter_missing_metadata_rejects_without_mutating(tmp_path):
    source=_trainer(); path=tmp_path/"adapter.lora"; save_adapter(source,path,"ckpt")
    target=_trainer(); before=[p.detach().clone() for p in target.parameters()]
    (tmp_path/"adapter.lora.json").unlink()
    try: load_adapter(target,path); assert False
    except ValueError: pass
    for got,want in zip(target.parameters(),before): assert torch.equal(got,want)


def test_adapter_rejects_nan_header_and_metadata_without_mutating(tmp_path):
    source=_trainer(); path=tmp_path/"bad.lora"; save_adapter(source,path,"ckpt")
    target=_trainer(); before=[p.detach().clone() for p in target.parameters()]
    raw=bytearray(path.read_bytes()); raw[20:24]=struct.pack("<f",float("nan")); path.write_bytes(raw)
    with pytest.raises(ValueError): load_adapter(target,path)
    for got,want in zip(target.parameters(),before): assert torch.equal(got,want)
    save_adapter(source,path,"ckpt")
    meta=json.loads((tmp_path/"bad.lora.json").read_text()); meta["alpha"]=float("nan"); (tmp_path/"bad.lora.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError): load_adapter(target,path)
    for got,want in zip(target.parameters(),before): assert torch.equal(got,want)


def test_adapter_rejects_boolean_integer_metadata_without_mutating(tmp_path):
    source=_trainer(); path=tmp_path/"bad-int.lora"; save_adapter(source,path,"ckpt")
    target=_trainer(); before=[p.detach().clone() for p in target.parameters()]
    meta=json.loads((tmp_path/"bad-int.lora.json").read_text()); meta["expert"]=True; (tmp_path/"bad-int.lora.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError): load_adapter(target,path)
    for got,want in zip(target.parameters(),before): assert torch.equal(got,want)
