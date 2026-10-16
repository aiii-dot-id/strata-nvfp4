#!/usr/bin/env python3
"""Read-only, dependency-free checkpoint structure inspector.

Only JSON metadata and safetensors headers are read.  Tensor payloads are never
read or hashed, which makes this suitable for a CPU preflight of large models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any


class CheckpointError(ValueError):
    pass


_DTYPE_WIDTHS = {
    "BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "U32": 4,
    "I32": 4, "U64": 8, "I64": 8, "F8_E4M3": 1, "F8_E5M2": 1,
    "F16": 2, "BF16": 2, "F32": 4, "F64": 8, "C64": 8, "C128": 16,
}


def _file(path: Path) -> Path:
    if not path.exists() or not path.is_file() or path.is_symlink():
        raise CheckpointError(f"required regular file missing: {path.name}")
    return path


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_file(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"invalid JSON: {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"JSON object required: {path.name}")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CheckpointError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _small_record(path: Path, data: bytes) -> dict[str, Any]:
    return {
        "name": path.name,
        "file_size": path.stat().st_size,
        "hashed_bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], bytes]:
    """Return parsed header and exactly the bytes hashed (length + JSON header)."""
    path = _file(path)
    try:
        with path.open("rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) != 8:
                raise CheckpointError(f"truncated safetensors length: {path.name}")
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len > path.stat().st_size - 8 or header_len > 64 * 1024 * 1024:
                raise CheckpointError(f"invalid safetensors header length: {path.name}")
            header_bytes = handle.read(header_len)
    except OSError as exc:
        raise CheckpointError(f"cannot read {path.name}: {exc}") from exc
    if len(header_bytes) != header_len:
        raise CheckpointError(f"truncated safetensors header: {path.name}")
    try:
        header = json.loads(header_bytes.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"invalid safetensors header: {path.name}: {exc}") from exc
    if not isinstance(header, dict):
        raise CheckpointError(f"safetensors header must be object: {path.name}")
    return header, raw_len + header_bytes


def _product(shape: Any, name: str) -> int:
    if not isinstance(shape, list) or not all(type(x) is int and x >= 0 for x in shape):
        raise CheckpointError(f"invalid shape for tensor {name}")
    result = 1
    for dim in shape:
        result *= dim
    return result


def _validate_tensor(name: str, value: Any, payload_end: int) -> tuple[int, int, str, list[int]]:
    if not isinstance(value, dict) or not isinstance(value.get("dtype"), str):
        raise CheckpointError(f"invalid tensor header entry: {name}")
    dtype = value["dtype"]
    if dtype not in _DTYPE_WIDTHS:
        raise CheckpointError(f"unsupported dtype for tensor {name}: {dtype}")
    shape = value.get("shape")
    count = _product(shape, name)
    offsets = value.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(x) is int for x in offsets):
        raise CheckpointError(f"invalid offsets for tensor {name}")
    start, end = offsets
    if start < 0 or end < start or end > payload_end:
        raise CheckpointError(f"offset out of bounds for tensor {name}")
    nbytes = end - start
    if nbytes != count * _DTYPE_WIDTHS[dtype]:
        raise CheckpointError(f"dtype/shape byte length mismatch for tensor {name}")
    return count, nbytes, dtype, shape


def _is_expert(name: str) -> bool:
    if re.search(r"(?:^|[._/])shared[_./]?experts?(?:[._/]|$)", name, re.I):
        return False
    return bool(re.search(r"(?:^|[._/])(?:routed[_./]?)?experts?(?:[._/]|$)", name, re.I))


def _shard_name(name: str) -> str:
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise CheckpointError(f"unsafe shard path in index: {name}")
    if not name or candidate == Path("."):
        raise CheckpointError("empty shard path in index")
    return name


def _special_metadata(root: Path, config: dict[str, Any], generation: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "config_eos_token_id": config.get("eos_token_id"),
        "generation_config_eos_token_id": generation.get("eos_token_id") if generation else None,
    }
    for filename in ("tokenizer_config.json", "special_tokens_map.json"):
        path = root / filename
        if path.exists() and path.is_file() and not path.is_symlink():
            data = _json(path)
            result[filename] = {k: data[k] for k in data if "eos" in k.lower() or "special" in k.lower()}
    tok = root / "tokenizer.json"
    if tok.exists() and tok.is_file() and not tok.is_symlink():
        data = _json(tok)
        added = data.get("added_tokens", [])
        if isinstance(added, list):
            result["tokenizer.json"] = {
                "special_tokens": [x for x in added if isinstance(x, dict) and x.get("special")]
            }
    return result


def inspect_checkpoint(root: str | Path, compatibility_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(root)
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config_bytes = _file(config_path).read_bytes()
    config = _json(config_path)
    index_bytes = _file(index_path).read_bytes()
    index = _json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()):
        raise CheckpointError("index weight_map must be an object of tensor to shard names")
    if not weight_map:
        raise CheckpointError("index weight_map must not be empty")
    shard_names = sorted({_shard_name(name) for name in weight_map.values()})
    headers: dict[str, dict[str, Any]] = {}
    tensor_records: dict[str, dict[str, Any]] = {}
    records = [_small_record(config_path, config_bytes), _small_record(index_path, index_bytes)]
    dtype_counts: dict[str, int] = {}
    dtype_bytes: dict[str, int] = {}
    routed = {"parameters": 0, "payload_bytes": 0}
    other = {"parameters": 0, "payload_bytes": 0}
    total_params = total_payload = 0
    for shard_name in shard_names:
        shard = root / shard_name
        header, header_bytes = read_safetensors_header(shard)
        records.append({**_small_record(shard, header_bytes), "header_bytes": len(header_bytes), "hash_scope": "safetensors_header"})
        if any(name in headers for name in header if name != "__metadata__"):
            duplicate = next(name for name in header if name != "__metadata__" and name in headers)
            raise CheckpointError(f"duplicate tensor header: {duplicate}")
        payload_end = shard.stat().st_size - len(header_bytes)
        intervals: list[tuple[int, int, str]] = []
        for name, value in header.items():
            if name == "__metadata__":
                continue
            count, nbytes, dtype, _ = _validate_tensor(name, value, payload_end)
            start, end = value["data_offsets"]
            headers[name] = {"shard": shard_name, "count": count, "bytes": nbytes, "dtype": dtype}
            tensor_records[name] = {
                "shard": shard_name, "dtype": dtype, "shape": value["shape"],
                "data_offsets": [start, end], "payload_bytes": nbytes,
                "header_bytes": len(header_bytes),
            }
            intervals.append((start, end, name))
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count
            dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + nbytes
            total_params += count
            total_payload += nbytes
            bucket = routed if _is_expert(name) else other
            bucket["parameters"] += count
            bucket["payload_bytes"] += nbytes
        intervals.sort()
        cursor = 0
        for start, end, name in intervals:
            if start != cursor:
                raise CheckpointError(f"overlap or payload gap before tensor {name}")
            cursor = end
        if cursor != payload_end:
            raise CheckpointError(f"unindexed payload bytes in shard: {shard_name}")
    missing = sorted(set(weight_map) - set(headers))
    extra = sorted(set(headers) - set(weight_map))
    if missing:
        raise CheckpointError(f"index tensor missing from headers: {missing[0]}")
    if extra:
        raise CheckpointError(f"header tensor absent from index: {extra[0]}")
    for name, shard_name in weight_map.items():
        if headers[name]["shard"] != shard_name:
            raise CheckpointError(f"index shard mismatch for tensor: {name}")
    generation_path = root / "generation_config.json"
    generation = _json(generation_path) if generation_path.exists() else None
    result: dict[str, Any] = {
        "root": str(root),
        "architecture": config.get("architectures", config.get("model_type")),
        "model_type": config.get("model_type"),
        "context_length": config.get("max_position_embeddings", config.get("max_seq_len")),
        "parameter_count": total_params,
        "payload_bytes": total_payload,
        "dtype_counts": dtype_counts,
        "dtype_payload_bytes": dtype_bytes,
        "routed_expert": routed,
        "other": other,
        "shard_count": len(shard_names),
        "files": records,
        "tensors": {name: tensor_records[name] for name in sorted(tensor_records)},
        "special_tokens": _special_metadata(root, config, generation),
    }
    if compatibility_path is not None:
        compatibility = _json(Path(compatibility_path))
        result["compatibility"] = {
            "strata_support": False,
            "guard": "known-qwen-only",
            "requested": compatibility,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--compatibility", type=Path, help="optional JSON compatibility request")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    try:
        result = inspect_checkpoint(args.checkpoint, args.compatibility)
    except (CheckpointError, OSError) as exc:
        parser.error(str(exc))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
