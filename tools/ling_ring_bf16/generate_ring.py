#!/usr/bin/env python3
"""Local-tokenizer greedy generation frontend for the verified Ring runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_ring_verified import VerificationError, _file, _read_json, _sha, run_verified


def _binding_paths(checkpoint: Path) -> list[Path]:
    paths = [checkpoint / "tokenizer.json", checkpoint / "tokenizer_config.json", checkpoint / "generation_config.json"]
    special = checkpoint / "special_tokens_map.json"
    if special.exists():
        paths.append(special)
    return paths


def _binding_snapshot(paths: list[Path]) -> dict[str, tuple[str, int]]:
    snapshot = {}
    for path in paths:
        path = _file(path, "frontend binding")
        snapshot[str(path)] = (_sha(path), path.stat().st_size)
    return snapshot


def _eos_ids(checkpoint: Path) -> list[int]:
    path = _file(checkpoint / "generation_config.json", "generation config")
    config = _read_json(path, "generation config")
    value = config.get("eos_token_id")
    values = value if isinstance(value, list) else [value]
    if not values or any(type(x) is not int or x < 0 or x >= 157184 for x in values) or len(set(values)) != len(values):
        raise VerificationError("generation_config.json must contain unique EOS vocabulary IDs")
    return values


def _tokenizer(checkpoint: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise VerificationError("local transformers is required for the original tokenizer; installation is not attempted") from exc
    try:
        return AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=True, use_fast=True)
    except Exception as exc:
        raise VerificationError(f"cannot load local checkpoint tokenizer: {exc}") from exc


def _ids(value) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(type(x) is not int or x < 0 or x >= 157184 for x in value):
        raise VerificationError("tokenizer returned an invalid token sequence")
    return value


def generate(checkpoint: Path, manifest: Path, executable: Path, prompt: str, output: Path,
             cache_mib: int, max_new_tokens: int, system: str | None = None,
             logits_mode: str = "all", timeout: float = 1800, rope_frequency: str = "native",
             attention_order: str = "native", attention_scale: str = "native", rms_mode: str = "native",
             matmul: str = "bf16-gemv", router: str = "native",
             attention_backend: str = "native", pointwise: str = "native") -> dict:
    checkpoint = checkpoint.resolve()
    if max_new_tokens < 1 or max_new_tokens > 4096:
        raise VerificationError("max_new_tokens must be in 1..4096")
    if logits_mode != "all":
        raise VerificationError("generation requires logits_mode all")
    if matmul not in {"bf16-gemv", "fp32-cublas"}:
        raise VerificationError("matmul must be bf16-gemv or fp32-cublas")
    if router not in {"native", "torch-cuda"}:
        raise VerificationError("router must be native or torch-cuda")
    if attention_backend not in {"native", "torch-cuda"}:
        raise VerificationError("attention backend must be native or torch-cuda")
    if pointwise not in {"native", "torch-cuda"}:
        raise VerificationError("pointwise backend must be native or torch-cuda")
    binding = _binding_paths(checkpoint)
    before_bindings = _binding_snapshot(binding)
    eos = _eos_ids(checkpoint)
    tokenizer = _tokenizer(checkpoint)
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    if not hasattr(tokenizer, "apply_chat_template"):
        raise VerificationError("original tokenizer has no chat template")
    try:
        prompt_ids = _ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    except Exception as exc:
        raise VerificationError(f"original chat template failed: {exc}") from exc
    if not prompt_ids or len(prompt_ids) + max_new_tokens > 4096:
        raise VerificationError("prompt plus generation exceeds the 4096-token evaluation window")
    if before_bindings != _binding_snapshot(binding):
        raise VerificationError("frontend tokenizer/config bindings drifted during tokenization")
    receipt = run_verified(checkpoint, manifest, executable, ",".join(map(str, prompt_ids)), output,
                           cache_mib, generate=max_new_tokens, eos_ids=eos,
                           logits_mode=logits_mode, binding_files=binding, timeout=timeout,
                           rope_frequency=rope_frequency, attention_order=attention_order, attention_scale=attention_scale,
                           rms_mode=rms_mode, matmul=matmul, router=router,
                           attention_backend=attention_backend, pointwise=pointwise)
    generation_receipt = receipt["generation"]
    generated_ids = generation_receipt["generated_token_ids"]
    try:
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except Exception as exc:
        raise VerificationError(f"original tokenizer decode failed: {exc}") from exc
    frontend = {
        "schema": "strata-ring-generation-frontend-v1",
        "prompt": prompt,
        "system": system,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated_ids,
        "finish_reason": generation_receipt["finish_reason"],
        "decoded": decoded,
        "eos_token_ids": eos,
        "max_new_tokens": max_new_tokens,
        "logits_mode": logits_mode,
        "rope_frequency": rope_frequency, "attention_value_order": attention_order, "attention_score_scale": attention_scale,
        "rms_mode": rms_mode, "matmul_backend": matmul, "router_backend": router,
        "attention_backend": attention_backend, "pointwise_backend": pointwise,
        "bindings": receipt["frontend"],
    }
    (output / "frontend.json").write_text(json.dumps(frontend, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "generated.txt").write_text(decoded, encoding="utf-8")
    return {"receipt": receipt, "frontend": frontend}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-mib", required=True, type=int)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--logits-mode", choices=("all", "last"), default="all")
    parser.add_argument("--rope-frequency", choices=("native", "original"), default="native")
    parser.add_argument("--attention-order", choices=("native", "normalized-first"), default="native")
    parser.add_argument("--attention-scale", choices=("native", "divide"), default="native")
    parser.add_argument("--rms-mode", choices=("native", "torch-cuda"), default="native")
    parser.add_argument("--matmul", choices=("bf16-gemv", "fp32-cublas"), default="bf16-gemv")
    parser.add_argument("--router", choices=("native", "torch-cuda"), default="native")
    parser.add_argument("--attention-backend", choices=("native", "torch-cuda"), default="native")
    parser.add_argument("--pointwise", choices=("native", "torch-cuda"), default="native")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    try:
        result = generate(args.checkpoint, args.manifest, args.executable, args.prompt, args.output,
                          args.cache_mib, args.max_new_tokens, args.system, args.logits_mode, args.timeout,
                          args.rope_frequency, args.attention_order, args.attention_scale, args.rms_mode,
                          args.matmul, args.router, attention_backend=args.attention_backend,
                          pointwise=args.pointwise)
        print(result["frontend"]["decoded"], end="")
    except (OSError, ValueError, VerificationError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
