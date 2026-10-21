"""Exact self-OPD and on-policy RLVR losses for fixed-prefix rollouts."""
from __future__ import annotations

from typing import Sequence
import torch


def _check_logits(x: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(x) or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 2:
        raise ValueError(f"{name} must have shape [states,vocab]")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be finite floating point")


def _temperature(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not torch.isfinite(torch.tensor(float(value))) or float(value) <= 0:
        raise ValueError("temperature must be finite and positive")
    return float(value)


def self_opd(logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0):
    """Return exact full-vocabulary reverse KL and metrics.

    ``logits`` are student distributions at student-sampled fixed-prefix
    states.  The teacher is detached and is evaluated at the same states and
    temperature.  The objective is the sum over states of
    ``p_student * (log p_student - log q_frozen)``.
    """
    _check_logits(logits, "logits"); _check_logits(teacher_logits, "teacher_logits")
    if logits.shape != teacher_logits.shape: raise ValueError("logit shapes must match")
    if logits.dtype != teacher_logits.dtype or logits.device != teacher_logits.device: raise ValueError("student and teacher logits must share dtype and device")
    temp = _temperature(temperature)
    logp = torch.log_softmax(logits / temp, dim=-1)
    with torch.no_grad(): logq = torch.log_softmax(teacher_logits.detach() / temp, dim=-1)
    p = logp.exp()
    per_state = (p * (logp - logq)).sum(dim=-1)
    loss = per_state.sum()
    return loss, {"states": int(logits.shape[0]), "vocab": int(logits.shape[1]), "temperature": temp, "reverse_kl_sum": float(loss.detach()), "teacher_detached": True}


def rlvr(logitsflat: torch.Tensor, tokenidsflat: torch.Tensor, lengths: Sequence[int], rewards: Sequence[float], groupids: Sequence[int], temperature: float = 1.0):
    """Return one-update sequence-level leave-one-out policy-gradient loss.

    Rows in ``logitsflat`` are response-token states only; fixed-prefix rows
    are therefore excluded by construction.  ``lengths`` partitions rows by
    trajectory.  Every group must contain at least two trajectories.
    """
    _check_logits(logitsflat, "logitsflat"); temp = _temperature(temperature)
    n = len(lengths)
    if len(rewards) != n or len(groupids) != n or n < 2: raise ValueError("lengths, rewards, and groupids must have equal nonzero length")
    if any(type(x) is not int or x <= 0 for x in lengths): raise ValueError("trajectory lengths must be positive integers")
    if sum(lengths) != logitsflat.shape[0] or tokenidsflat.ndim != 1 or tokenidsflat.numel() != logitsflat.shape[0] or tokenidsflat.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64): raise ValueError("flat token shape mismatch")
    if tokenidsflat.numel() and (int(tokenidsflat.min()) < 0 or int(tokenidsflat.max()) >= logitsflat.shape[1]): raise ValueError("token id out of range")
    if not all(isinstance(g, int) for g in groupids): raise ValueError("group ids must be integers")
    reward = torch.as_tensor(rewards, dtype=logitsflat.dtype, device=logitsflat.device)
    if not torch.isfinite(reward).all(): raise ValueError("rewards must be finite")
    groups: dict[int, list[int]] = {}
    for i,g in enumerate(groupids): groups.setdefault(g, []).append(i)
    if any(len(items) < 2 for items in groups.values()): raise ValueError("each prompt group needs at least two trajectories")
    advantages = torch.empty(n, dtype=logitsflat.dtype, device=logitsflat.device)
    for items in groups.values():
        total = reward[items].sum()
        for i in items: advantages[i] = reward[i] - (total - reward[i]) / (len(items)-1)
    logp = torch.log_softmax(logitsflat / temp, dim=-1)
    selected = logp.gather(1, tokenidsflat.to(torch.long)[:, None]).squeeze(1)
    sequence_logp = []; start = 0
    for length in lengths:
        sequence_logp.append(selected[start:start+length].sum()); start += length
    sequence_logp = torch.stack(sequence_logp)
    loss = -(advantages * sequence_logp).mean()
    return loss, {"trajectories": n, "response_tokens": int(sum(lengths)), "groups": len(groups), "temperature": temp, "advantages": advantages.detach(), "sequence_logp": sequence_logp.detach(), "invalid_outcome_zero_reward_included": True}
