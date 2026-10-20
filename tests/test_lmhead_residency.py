#!/usr/bin/env python3
"""CPU-only receipt controls for opt-in LM-head residency.

These tests stub manifest admission and only import wrapper Python. They do not
open checkpoint tensors, initialize CUDA, spawn an executable, or use a model.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "ling_ring_bf16"


def load_runner(kind: str):
    dependency = "prepare_ring_manifest" if kind == "ring" else "prepare_ling_manifest"
    stub = types.ModuleType(dependency)
    error_type = type("AdmissionError", (Exception,), {})
    setattr(stub, "CheckpointError" if kind == "ring" else "AdmissionError", error_type)
    stub.prepare = lambda *args, **kwargs: None
    previous = sys.modules.get(dependency)
    sys.modules[dependency] = stub
    try:
        spec = importlib.util.spec_from_file_location(f"lmhead_{kind}_runner", TOOLS / f"run_{kind}_verified.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            del sys.modules[dependency]
        else:
            sys.modules[dependency] = previous


def write_receipt(directory: Path, receipt: dict) -> Path:
    path = directory / "execution.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def ring_receipt():
    return {"status": "executed", "model": "Ring-mini-2.0", "activation_dtype": "FP32",
            "quantized": False, "layers": 20, "checkpoint_id": "a" * 64,
            "matmul_backend": "bf16-gemv", "tokens": 1, "token_ids": [7],
            "logits_mode": "all", "logits_tokens": 1}


def ling_receipt(runner):
    return {"status": "executed", "model": runner.MODEL, "activation_dtype": "FP32",
            "quantized": False, "layers": runner.LAYERS, "checkpoint_id": "b" * 64,
            "rope_frequency": "native", "rms_mode": "native", "matmul_backend": "bf16-gemv",
            "pointwise_mode": "native", "tokens": 1, "token_ids": [9],
            "logits_mode": "all", "logits_tokens": 1,
            "matmul_fp32_weight_scratch_bytes": 0,
            "workspace_bytes": 157184 * 1536 * 2 + 4608 * 4 + 157184 * 4}


class ResidencyReceiptControls(unittest.TestCase):
    def check(self, runner, kind, receipt, mode):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_receipt(Path(temporary), receipt)
            if kind == "ring":
                return runner._require_execution(path, [7], "a" * 64, 0, "bf16-gemv", mode)
            return runner._require_execution(path, [9], "b" * 64, 0, "native", "native", "bf16-gemv", "native", mode)

    def test_legacy_off_receipt_is_accepted_only_when_all_fields_absent(self):
        for kind in ("ring", "ling"):
            runner = load_runner(kind)
            base = ring_receipt() if kind == "ring" else ling_receipt(runner)
            self.assertEqual(self.check(runner, kind, base, "off")["tokens"], 1)
            with self.assertRaises(runner.VerificationError):
                self.check(runner, kind, base, "cuda")

    def test_cuda_rejects_missing_null_bool_float_and_negative_accounting(self):
        for kind in ("ring", "ling"):
            runner = load_runner(kind)
            base = ring_receipt() if kind == "ring" else ling_receipt(runner)
            expected_bytes = 157184 * (2048 if kind == "ring" else 1536) * 2
            good = dict(base, lm_head_residency="cuda", lm_head_resident_bytes=expected_bytes,
                        lm_head_resident_uploads=1, lm_head_resident_apply_calls=1)
            self.assertEqual(self.check(runner, kind, good, "cuda")["lm_head_resident_bytes"], expected_bytes)
            for field in ("lm_head_resident_bytes", "lm_head_resident_uploads", "lm_head_resident_apply_calls"):
                missing = dict(good); del missing[field]
                with self.assertRaises(runner.VerificationError, msg=f"{kind} missing {field}"):
                    self.check(runner, kind, missing, "cuda")
                for bad_value in (None, False, 1.0, -1):
                    malformed = dict(good); malformed[field] = bad_value
                    with self.assertRaises(runner.VerificationError, msg=f"{kind} {field}={bad_value!r}"):
                        self.check(runner, kind, malformed, "cuda")
            no_logit_count = dict(good); del no_logit_count["logits_tokens"]
            with self.assertRaises(runner.VerificationError):
                self.check(runner, kind, no_logit_count, "cuda")

    def test_off_rejects_partial_or_non_integer_zero_accounting(self):
        for kind in ("ring", "ling"):
            runner = load_runner(kind)
            base = ring_receipt() if kind == "ring" else ling_receipt(runner)
            all_zero = dict(base, lm_head_resident_bytes=0, lm_head_resident_uploads=0,
                            lm_head_resident_apply_calls=0)
            self.assertEqual(self.check(runner, kind, all_zero, "off")["tokens"], 1)
            partial = dict(base, lm_head_resident_bytes=0)
            with self.assertRaises(runner.VerificationError):
                self.check(runner, kind, partial, "off")
            for field in ("lm_head_resident_bytes", "lm_head_resident_uploads", "lm_head_resident_apply_calls"):
                for bad_value in (None, False, 0.0, -1, 1):
                    malformed = dict(all_zero); malformed[field] = bad_value
                    with self.assertRaises(runner.VerificationError, msg=f"{kind} off {field}={bad_value!r}"):
                        self.check(runner, kind, malformed, "off")

    def test_invalid_requested_mode_is_rejected_before_admission(self):
        for kind in ("ring", "ling"):
            runner = load_runner(kind)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.assertRaises(runner.VerificationError):
                    if kind == "ring":
                        runner.run_verified(root / "checkpoint", root / "manifest", root / "executable",
                                            "7", root / "output", 6, lm_head_residency="invalid")
                    else:
                        runner.run_verified(root / "checkpoint", root / "manifest", root / "executable",
                                            "9", root / "output", 5, lm_head_residency="invalid")


if __name__ == "__main__":
    unittest.main()
