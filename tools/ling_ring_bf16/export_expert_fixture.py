#!/usr/bin/env python3
"""Export three exact raw BF16 expert tensors from an inspected checkpoint.

This creates a small exact-byte fixture; it is not a full-weight integrity
check, GGUF/Strata pack, conversion, or quantization operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from inspect_checkpoint import CheckpointError, read_safetensors_header


MAX_READ = 8 * 1024 * 1024
NAMES = {
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
}


class ExportError(ValueError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ExportError(f"duplicate report JSON key: {key}")
        value[key] = item
    return value


def _load(path: Path, raw: bytes | None = None) -> dict[str, Any]:
    try:
        value = json.loads((raw if raw is not None else path.read_bytes()).decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExportError(f"JSON object required: {path}")
    return value


def _pin_files(root: Path, report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pins = {}
    for item in report.get("files", []):
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            pins[item["name"]] = item
    config = pins.get("config.json")
    if not config:
        raise ExportError("report has no config.json pin")
    config_path = root / "config.json"
    if config_path.stat().st_size != config.get("file_size"):
        raise ExportError("config file size changed since inspection")
    if _sha(config_path.read_bytes()) != config.get("sha256"):
        raise ExportError("config SHA-256 changed since inspection")
    return pins


def _safe_shard_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ExportError("shard name must be a string")
    candidate = Path(value)
    if not value or candidate.is_absolute() or ".." in candidate.parts:
        raise ExportError(f"unsafe shard path: {value}")
    return value


def export_fixture(checkpoint: str | Path, report_path: str | Path, output_dir: str | Path, family: str) -> dict[str, Any]:
    root, report_path, output = Path(checkpoint), Path(report_path), Path(output_dir)
    if output.exists():
        raise ExportError(f"refusing existing output directory: {output}")
    report_bytes = report_path.read_bytes()
    report = _load(report_path, report_bytes)
    tensors = report.get("tensors")
    if not isinstance(tensors, dict):
        raise ExportError("report lacks tensors manifest")
    prefix = "model.layers.1.mlp.experts.0."
    if family not in ("ling", "ring"):
        raise ExportError(f"unknown family: {family}")
    expected = {
        "gate_proj": [512, 1536] if family == "ling" else [512, 2048],
        "up_proj": [512, 1536] if family == "ling" else [512, 2048],
        "down_proj": [1536, 512] if family == "ling" else [2048, 512],
    }
    pins = _pin_files(root, report)
    output.mkdir(parents=True)
    entries: dict[str, Any] = {}
    total = 0
    try:
        for projection, short in NAMES.items():
            tensor_name = prefix + projection + ".weight"
            record = tensors.get(tensor_name)
            if not isinstance(record, dict):
                raise ExportError(f"report missing tensor: {tensor_name}")
            if record.get("dtype") != "BF16" or record.get("shape") != expected[projection]:
                raise ExportError(f"unexpected {tensor_name} dtype/shape")
            shard_name = _safe_shard_name(record.get("shard"))
            shard_pin = pins.get(shard_name)
            if not isinstance(shard_name, str) or not isinstance(shard_pin, dict):
                raise ExportError(f"missing shard pin for {tensor_name}")
            shard = root / shard_name
            if not shard.is_file() or shard.is_symlink():
                raise ExportError(f"unsafe or missing shard: {shard_name}")
            if shard.stat().st_size != shard_pin.get("file_size"):
                raise ExportError(f"shard size changed: {shard_name}")
            header_bytes = int(record.get("header_bytes", 0))
            if header_bytes < 8 or header_bytes > shard.stat().st_size:
                raise ExportError(f"invalid header span: {shard_name}")
            if header_bytes > 64 * 1024 * 1024:
                raise ExportError(f"header span exceeds 64 MiB: {shard_name}")
            try:
                header, header_raw = read_safetensors_header(shard)
            except CheckpointError as exc:
                raise ExportError(str(exc)) from exc
            if len(header_raw) != header_bytes or _sha(header_raw) != shard_pin.get("sha256"):
                raise ExportError(f"shard header changed: {shard_name}")
            pinned_entry = header.get(tensor_name)
            if not isinstance(pinned_entry, dict):
                raise ExportError(f"tensor missing from current header: {tensor_name}")
            if pinned_entry.get("dtype") != record.get("dtype") or pinned_entry.get("shape") != record.get("shape") or pinned_entry.get("data_offsets") != record.get("data_offsets"):
                raise ExportError(f"tensor header differs from report: {tensor_name}")
            with shard.open("rb") as source:
                offsets = record.get("data_offsets")
                if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(x) is int for x in offsets):
                    raise ExportError(f"invalid offsets: {tensor_name}")
                start, end = offsets
                absolute = header_bytes + start
                length = end - start
                if start < 0 or end < start or absolute + length > shard.stat().st_size:
                    raise ExportError(f"source range out of bounds: {tensor_name}")
                if length != expected[projection][0] * expected[projection][1] * 2 or length > MAX_READ:
                    raise ExportError(f"unexpected source length: {tensor_name}")
                try:
                    if sys.platform == "darwin":
                        import fcntl
                        fcntl.fcntl(source.fileno(), 48, 1)  # F_NOCACHE
                except OSError as exc:
                    raise ExportError(f"cannot enable F_NOCACHE: {exc}") from exc
                source.seek(absolute)
                data = source.read(length)
            if len(data) != length:
                raise ExportError(f"short payload read: {tensor_name}")
            target = output / f"{family}-{short}.bf16"
            target.write_bytes(data)
            entries[short] = {
                "tensor": tensor_name, "path": target.name, "dtype": "BF16",
                "shape": expected[projection], "bytes": length, "sha256": _sha(data),
                "source_shard": shard_name, "source_header_bytes": header_bytes,
                "source_data_offsets": offsets, "source_file_offset": absolute,
                "source_header_sha256": shard_pin["sha256"],
            }
            total += length
        if total > MAX_READ:
            raise ExportError("fixture payload exceeds bounded read budget")
        manifest = {
            "format": "raw-bf16-expert-fixture", "family": family,
            "sample_exact_bytes_only": True, "quantized": False,
            "checkpoint": str(root), "report": str(report_path),
            "report_sha256": _sha(report_bytes), "report_bytes": len(report_bytes),
            "config_sha256": pins["config.json"]["sha256"], "total_bytes": total,
            "tensors": entries,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest
    except Exception:
        # No partial fixture is left behind after a failed validation/export.
        for child in output.iterdir():
            if child.is_file():
                child.unlink()
        output.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--family", required=True, choices=("ling", "ring"))
    args = parser.parse_args()
    try:
        result = export_fixture(args.checkpoint, args.report, args.output_dir, args.family)
    except (ExportError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
