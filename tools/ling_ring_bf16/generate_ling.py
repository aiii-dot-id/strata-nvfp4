#!/usr/bin/env python3
"""Local-tokenizer greedy generation frontend for verified Ling."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from run_ling_verified import (
    VerificationError, _file, _read_json, _require_generation, _sha,
    run_verified, VOCAB, MAX_TOKENS,
)

def _binding_paths(checkpoint):
    paths = [checkpoint / "tokenizer.json", checkpoint / "tokenizer_config.json",
             checkpoint / "generation_config.json"]
    template = checkpoint / "chat_template.jinja"
    if template.exists():
        paths.append(template)
    special = checkpoint / "special_tokens_map.json"
    if special.exists():
        paths.append(special)
    return paths

def _snapshot(paths):
    return {str(_file(p, "frontend binding")): (_sha(p), p.stat().st_size) for p in paths}

def _validate_eos(value, source):
    vals = value if isinstance(value, list) else [value]
    if (not vals or any(type(x) is not int or isinstance(x, bool) or x < 0 or x >= VOCAB for x in vals)
            or len(set(vals)) != len(vals)):
        raise VerificationError(f"{source} must contain unique EOS vocabulary IDs")
    return vals

def _eos_ids(checkpoint, explicit=None):
    if explicit is not None:
        return _validate_eos(explicit, "explicit EOS override"), "explicit"
    generation = _read_json(_file(checkpoint / "generation_config.json", "generation config"), "generation config")
    if "eos_token_id" in generation and generation["eos_token_id"] is not None:
        return _validate_eos(generation["eos_token_id"], "generation_config.json eos_token_id"), "generation_config.json"
    config = _read_json(_file(checkpoint / "config.json", "config"), "config")
    if "eos_token_id" not in config:
        raise VerificationError("no eos_token_id in explicit override, generation_config.json, or config.json")
    return _validate_eos(config["eos_token_id"], "config.json eos_token_id"), "config.json"

def _tokenizer(checkpoint):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise VerificationError("local transformers is required; installation is not attempted") from exc
    try:
        return AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                             trust_remote_code=True, use_fast=True)
    except Exception as exc:
        raise VerificationError(f"cannot load local checkpoint tokenizer: {exc}") from exc

def _ids(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(type(x) is not int or isinstance(x, bool) or x < 0 or x >= VOCAB for x in value):
        raise VerificationError("tokenizer returned invalid token sequence")
    return value

def generate(checkpoint, manifest, executable, prompt, output, cache_mib, max_new_tokens,
             system=None, timeout=1800, eos_ids=None, thinking="auto", rope_frequency="native",
             rms_mode="native", matmul="bf16-gemv", pointwise="native", lm_head_residency="off"):
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    if max_new_tokens < 1 or max_new_tokens > MAX_TOKENS:
        raise VerificationError("max_new_tokens must be in 1..1024")
    if thinking not in {"auto", "on", "off"}:
        raise VerificationError("thinking must be auto, on, or off")
    if rope_frequency not in {"native", "original"}:
        raise VerificationError("rope_frequency must be native or original")
    if rms_mode not in {"native", "torch-cuda-order"}:
        raise VerificationError("rms_mode must be native or torch-cuda-order")
    if matmul not in {"bf16-gemv", "fp32-cublas"}:
        raise VerificationError("matmul must be bf16-gemv or fp32-cublas")
    if pointwise not in {"native", "torch-cuda"}:
        raise VerificationError("pointwise must be native or torch-cuda")
    if lm_head_residency not in {"off", "cuda"}:
        raise VerificationError("lm_head_residency must be off or cuda")
    binding = _binding_paths(checkpoint)
    before = _snapshot(binding)
    eos, eos_source = _eos_ids(checkpoint, eos_ids)
    tokenizer = _tokenizer(checkpoint)
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    if not hasattr(tokenizer, "apply_chat_template"):
        raise VerificationError("original tokenizer has no chat template")
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    if thinking != "auto":
        kwargs["enable_thinking"] = thinking == "on"
    try:
        prompt_ids = _ids(tokenizer.apply_chat_template(messages, **kwargs))
    except Exception as exc:
        raise VerificationError(f"original chat template failed: {exc}") from exc
    if not prompt_ids or len(prompt_ids) + max_new_tokens > MAX_TOKENS:
        raise VerificationError("prompt plus generation exceeds 1024-token evaluation window")
    if before != _snapshot(binding):
        raise VerificationError("frontend tokenizer/config bindings drifted during tokenization")
    receipt = run_verified(
        checkpoint, manifest, executable, ",".join(map(str, prompt_ids)), output,
        cache_mib, timeout, max_new_tokens, eos, binding_files=binding,
        rope_frequency=rope_frequency, rms_mode=rms_mode, matmul=matmul, pointwise=pointwise, lm_head_residency=lm_head_residency)
    generated = receipt["generation"]["generated_token_ids"]
    try:
        decoded = tokenizer.decode(generated, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
    except Exception as exc:
        raise VerificationError(f"original tokenizer decode failed: {exc}") from exc
    frontend = {
        "schema": "strata-ling-generation-frontend-v1",
        "prompt": prompt, "system": system, "thinking": thinking,
        "prompt_token_ids": prompt_ids, "generated_token_ids": generated,
        "finish_reason": receipt["generation"]["finish_reason"],
        "decoded": decoded, "eos_token_ids": eos, "eos_source": eos_source,
        "max_new_tokens": max_new_tokens, "rope_frequency": rope_frequency,
        "rms_mode": rms_mode, "matmul": matmul, "pointwise": pointwise, "lm_head_residency": lm_head_residency,
        "bindings": receipt["frontend"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "frontend.json").write_text(json.dumps(frontend, indent=2, sort_keys=True) + chr(10))
    (output / "generated.txt").write_text(decoded)
    return {"receipt": receipt, "frontend": frontend}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--executable", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--system")
    p.add_argument("--thinking", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--cache-mib", required=True, type=int)
    p.add_argument("--max-new-tokens", required=True, type=int)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--eos-ids", help="explicit comma-separated EOS override")
    p.add_argument("--rope-frequency", choices=("native", "original"), default="native")
    p.add_argument("--rms-mode", choices=("native", "torch-cuda-order"), default="native")
    p.add_argument("--matmul", choices=("bf16-gemv", "fp32-cublas"), default="bf16-gemv")
    p.add_argument("--pointwise", choices=("native", "torch-cuda"), default="native")
    p.add_argument("--lm-head-residency", choices=("off", "cuda"), default="off")
    args = p.parse_args()
    try:
        explicit = None if args.eos_ids is None else [int(x, 10) for x in args.eos_ids.split(",")]
        result = generate(args.checkpoint, args.manifest, args.executable, args.prompt,
                          args.output, args.cache_mib, args.max_new_tokens, args.system,
                          args.timeout, explicit, args.thinking, args.rope_frequency,
                          args.rms_mode, args.matmul, args.pointwise, args.lm_head_residency)
        print(result["frontend"]["decoded"], end="")
    except (OSError, ValueError, VerificationError) as exc:
        p.error(str(exc))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
