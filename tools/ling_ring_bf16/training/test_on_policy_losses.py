import itertools
import torch
import pytest

from on_policy_losses import self_opd, rlvr


def test_self_opd_full_vocab_temperature_and_finite_difference():
    torch.manual_seed(4); logits=torch.randn(2,4,requires_grad=True); teacher=torch.randn(2,4)
    loss,_=self_opd(logits,teacher,temperature=2.0); loss.backward(); index=(0,1); analytic=logits.grad[index].item(); eps=1e-3
    with torch.no_grad(): logits[index]+=eps
    plus=self_opd(logits,teacher,2.0)[0].item()
    with torch.no_grad(): logits[index]-=2*eps
    minus=self_opd(logits,teacher,2.0)[0].item()
    assert abs(analytic-(plus-minus)/(2*eps))<2e-3


def test_self_opd_teacher_has_no_gradient_and_not_sampled_scalar():
    s=torch.tensor([[1.,0.,-1.]],requires_grad=True); q=torch.tensor([[0.,2.,-2.]],requires_grad=True)
    loss,metrics=self_opd(s,q,1.0); expected=(torch.softmax(s,-1)*(torch.log_softmax(s,-1)-torch.log_softmax(q,-1))).sum()
    assert torch.allclose(loss,expected.double()); loss.backward(); assert q.grad is None; assert metrics["teacher_detached"]
    assert s.grad is not None and torch.count_nonzero(s.grad)==3


def test_self_opd_rejects_invalid_temperature_and_nonfinite():
    x=torch.zeros(1,3)
    for t in (False, True, 0,-1,float("inf"),float("nan")):
        with pytest.raises(ValueError): self_opd(x,x,t)
    bad=x.clone();bad[0,0]=float("nan")
    with pytest.raises(ValueError): self_opd(bad,x)
    with pytest.raises(ValueError): self_opd(x,x.double())


def _bandit_logits():
    return torch.tensor([[1.,0.,-1.],[0.5,-.5,0.],[0.,.5,-.5]],requires_grad=True)


def test_rlvr_leave_one_out_and_sequence_sum_no_length_normalization():
    logits=_bandit_logits(); tokens=torch.tensor([0,1,2]); loss,m=rlvr(logits,tokens,[2,1],[2.,0.],[7,7],temperature=1.0)
    assert torch.allclose(m["advantages"],torch.tensor([2.,-2.]))
    expected=-(2*m["sequence_logp"][0]-2*m["sequence_logp"][1])/2
    assert torch.allclose(loss,expected); assert m["response_tokens"]==3


def test_rlvr_reward_permutation_flips_advantages():
    x=_bandit_logits(); _,a=rlvr(x,torch.tensor([0,1,2]),[1,1,1],[3.,1.,2.],[4,4,4]); _,b=rlvr(x,torch.tensor([0,1,2]),[1,1,1],[1.,3.,2.],[4,4,4])
    assert torch.allclose(a["advantages"],torch.tensor([1.5,-1.5,0.])); assert torch.allclose(b["advantages"],-a["advantages"])


def test_rlvr_all_zero_rewards_has_exact_zero_gradient():
    logits=_bandit_logits(); loss,_=rlvr(logits,torch.tensor([0,1,2]),[1,1,1],[0.,0.,0.],[1,1,1]); loss.backward()
    assert loss.item()==0.0 and torch.count_nonzero(logits.grad)==0


def test_rlvr_all_one_rewards_has_exact_zero_gradient():
    logits=_bandit_logits(); loss,_=rlvr(logits,torch.tensor([0,1,2]),[1,1,1],[1.,1.,1.],[1,1,1]); loss.backward()
    assert loss.item()==0.0 and torch.count_nonzero(logits.grad)==0


def test_rlvr_invalid_outcome_zero_reward_still_includes_sampled_tokens():
    logits=torch.randn(4,5,requires_grad=True); tokens=torch.tensor([1,2,3,4]); loss,m=rlvr(logits,tokens,[2,2],[1.,0.],[9,9]); loss.backward()
    assert m["invalid_outcome_zero_reward_included"] and torch.count_nonzero(logits.grad[:2])>0 and torch.count_nonzero(logits.grad[2:])>0


def test_rlvr_token_rows_are_response_only_and_temperature_matches_sampling():
    logits=torch.tensor([[2.,0.],[0.,2.],[1.,0.],[0.,1.]],requires_grad=True); tokens=torch.tensor([0,1,0,1])
    loss,m=rlvr(logits,tokens,[2,2],[1.,0.],[3,3],temperature=2.0)
    manual=torch.log_softmax(logits/2,-1).gather(1,tokens[:,None]).squeeze(1)
    assert torch.allclose(m["sequence_logp"],torch.stack([manual[:2].sum(),manual[2:].sum()]))
    assert m["temperature"]==2.0


def test_rlvr_directional_reward_prefers_high_reward_trajectory():
    logits=_bandit_logits(); loss,_=rlvr(logits,torch.tensor([0,1,2]),[1,2],[5.,0.],[2,2]); loss.backward()
    # Gradient descent increases the selected log-probability of trajectory 0.
    assert logits.grad[0,0] < 0


def test_rlvr_unbiased_two_arm_enumeration_control():
    # Enumerate every pair of sampled Bernoulli actions.  The weighted
    # estimator gradient must equal the negative analytic gradient of expected
    # reward, which catches a scalar-sample or biased-baseline implementation.
    base=torch.tensor([.7,-.2],requires_grad=True); p=torch.softmax(base,0); rewards=torch.tensor([1.,0.])
    expected_reward=(p*rewards).sum(); analytic=-torch.autograd.grad(expected_reward,base)[0]
    expected=torch.zeros(2); probs=p.detach()
    for actions in itertools.product((0,1),repeat=2):
        pair_prob=probs[actions[0]]*probs[actions[1]]
        logits=base.detach().clone().repeat(2,1).requires_grad_(True)
        loss,_=rlvr(logits,torch.tensor(actions),[1,1],[float(rewards[actions[0]]),float(rewards[actions[1]])],[5,5])
        expected += pair_prob*torch.autograd.grad(loss,logits)[0].sum(0)
    assert torch.allclose(expected,analytic,atol=1e-6,rtol=1e-6)


def test_rlvr_rejects_incomplete_shapes_and_singleton_groups():
    with pytest.raises(ValueError): rlvr(torch.zeros(2,3),torch.tensor([0]),[2],[1.],[1])
    with pytest.raises(ValueError): rlvr(torch.zeros(1,3),torch.tensor([0]),[1],[1.],[1])


def test_self_opd_near_equal_distributions_keep_small_positive_signal():
    s=torch.tensor([[12.,1.,-3.,4.]],dtype=torch.float32,requires_grad=True)
    q=s.detach().clone();q[0,1]+=0.002
    loss,m=self_opd(s,q)
    sd=s.detach().double();qd=q.double()
    lp=sd.log_softmax(-1);lq=qd.log_softmax(-1)
    reference=(lp.exp()*(lp-lq)).sum()
    assert loss.dtype==torch.float64 and m['objective_dtype']=='float64'
    assert 0 < loss.item() < 1e-9
    assert torch.allclose(loss,reference,atol=1e-18,rtol=1e-8)
    loss.backward()
    assert torch.isfinite(s.grad).all() and s.grad.abs().max()>0

def test_self_opd_equal_distributions_do_not_create_training_signal():
    s=torch.tensor([[12.,1.,-3.,4.]],requires_grad=True)
    loss,_=self_opd(s,s.detach().clone())
    loss.backward()
    assert loss.item()==0 and s.grad.abs().max()<1e-15
