import json
import struct
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from inspect_checkpoint import CheckpointError, inspect_checkpoint


def write_shard(root, name, tensors):
    header = {}
    offset = 0
    for tensor_name, dtype, shape, size in tensors:
        header[tensor_name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    (root / name).write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset)


def fixture(tensors=None, weight_map=None):
    root = Path(tempfile.mkdtemp())
    (root / "config.json").write_text(json.dumps({
        "architectures": ["LingForCausalLM"], "model_type": "ling",
        "max_position_embeddings": 131072, "eos_token_id": 2,
    }))
    tensors = tensors or [("model.experts.0.weight", "BF16", [2, 2], 8), ("model.embed.weight", "F32", [2], 8)]
    write_shard(root, "model-00001-of-00001.safetensors", tensors)
    weight_map = weight_map or {name: "model-00001-of-00001.safetensors" for name, *_ in tensors}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return root


class InspectorTests(unittest.TestCase):
    def test_valid_report_and_expert_split(self):
        root = fixture()
        (root / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))
        report = inspect_checkpoint(root)
        self.assertEqual(report["context_length"], 131072)
        self.assertEqual(report["parameter_count"], 6)
        self.assertEqual(report["routed_expert"]["payload_bytes"], 8)
        self.assertEqual(report["other"]["payload_bytes"], 8)
        self.assertEqual(len(report["files"]), 3)  # config, index, and shard header
        self.assertEqual(report["tensors"]["model.experts.0.weight"]["shape"], [2, 2])
        self.assertEqual(report["tensors"]["model.embed.weight"]["data_offsets"], [8, 16])
        self.assertEqual(report["tensors"]["model.embed.weight"]["header_bytes"], report["files"][2]["header_bytes"])

    def test_missing_shard_rejected(self):
        root = fixture(weight_map={"x": "not-present.safetensors"})
        with self.assertRaisesRegex(CheckpointError, "required regular file missing"):
            inspect_checkpoint(root)

    def test_index_shard_mismatch_rejected(self):
        root = fixture(tensors=[("x", "F32", [1], 4)], weight_map={"x": "model-00001-of-00001.safetensors"})
        write_shard(root, "model-00002-of-00002.safetensors", [("y", "F32", [1], 4)])
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"x": "model-00002-of-00002.safetensors", "y": "model-00001-of-00001.safetensors"}
        }))
        with self.assertRaisesRegex(CheckpointError, "index shard mismatch"):
            inspect_checkpoint(root)

    def test_index_missing_tensor_rejected(self):
        root = fixture(weight_map={"not-in-header": "model-00001-of-00001.safetensors"})
        with self.assertRaisesRegex(CheckpointError, "index tensor missing"):
            inspect_checkpoint(root)

    def test_header_extra_tensor_rejected(self):
        root = fixture(weight_map={"model.embed.weight": "model-00001-of-00001.safetensors"})
        with self.assertRaisesRegex(CheckpointError, "header tensor absent"):
            inspect_checkpoint(root)

    def test_bad_shape_rejected(self):
        root = fixture(tensors=[("x", "F32", ["bad"], 4)])
        with self.assertRaisesRegex(CheckpointError, "invalid shape"):
            inspect_checkpoint(root)

    def test_boolean_shape_value_rejected(self):
        root = fixture(tensors=[("x", "F32", [1], 4)])
        shard = root / "model-00001-of-00001.safetensors"
        raw = shard.read_bytes()
        header_len = struct.unpack("<Q", raw[:8])[0]
        header = b'{"x":{"dtype":"F32","shape":[true],"data_offsets":[0,4]}}'
        shard.write_bytes(struct.pack("<Q", len(header)) + header + raw[8 + header_len:])
        with self.assertRaisesRegex(CheckpointError, "invalid shape"):
            inspect_checkpoint(root)

    def test_unknown_dtype_rejected(self):
        root = fixture(tensors=[("x", "NOT_A_DTYPE", [1], 1)])
        with self.assertRaisesRegex(CheckpointError, "unsupported dtype"):
            inspect_checkpoint(root)

    def test_dtype_byte_mismatch_rejected(self):
        root = fixture(tensors=[("x", "F32", [2], 4)])
        with self.assertRaisesRegex(CheckpointError, "byte length mismatch"):
            inspect_checkpoint(root)

    def test_path_traversal_rejected(self):
        root = fixture(weight_map={"x": "../escape.safetensors"})
        with self.assertRaisesRegex(CheckpointError, "unsafe shard path"):
            inspect_checkpoint(root)

    def test_duplicate_index_key_rejected(self):
        root = fixture()
        (root / "model.safetensors.index.json").write_text('{"weight_map":{"x":"model-00001-of-00001.safetensors","x":"model-00001-of-00001.safetensors"}}')
        with self.assertRaisesRegex(CheckpointError, "duplicate JSON key"):
            inspect_checkpoint(root)

    def test_duplicate_header_key_rejected(self):
        root = fixture(tensors=[("x", "F32", [1], 4)])
        shard = root / "model-00001-of-00001.safetensors"
        raw = shard.read_bytes()
        header_len = struct.unpack("<Q", raw[:8])[0]
        payload = raw[8 + header_len:]
        header = b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        shard.write_bytes(struct.pack("<Q", len(header)) + header + payload)
        with self.assertRaisesRegex(CheckpointError, "duplicate JSON key"):
            inspect_checkpoint(root)

    def test_payload_gap_rejected(self):
        root = fixture(tensors=[("x", "F32", [1], 4)])
        shard = root / "model-00001-of-00001.safetensors"
        shard.write_bytes(shard.read_bytes() + b"\0" * 4)
        with self.assertRaisesRegex(CheckpointError, "unindexed payload|overlap or payload gap"):
            inspect_checkpoint(root)

    def test_payload_overlap_rejected(self):
        root = fixture(tensors=[("x", "F32", [1], 4), ("y", "F32", [1], 4)])
        shard = root / "model-00001-of-00001.safetensors"
        raw = shard.read_bytes()
        header_len = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + header_len])
        header["y"]["data_offsets"] = [2, 6]
        encoded = json.dumps(header, separators=(",", ":")).encode()
        shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 8)
        with self.assertRaisesRegex(CheckpointError, "overlap or payload gap"):
            inspect_checkpoint(root)

    def test_bad_offset_rejected(self):
        root = fixture(tensors=[("x", "F32", [2], 4)])
        shard = root / "model-00001-of-00001.safetensors"
        raw = shard.read_bytes()
        header_len = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + header_len])
        header["x"]["data_offsets"] = [0, 999]
        encoded = json.dumps(header, separators=(",", ":")).encode()
        shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 4)
        with self.assertRaisesRegex(CheckpointError, "offset out of bounds"):
            inspect_checkpoint(root)

    def test_duplicate_tensor_across_shards_rejected(self):
        root = fixture(tensors=[("x", "F32", [1], 4)])
        write_shard(root, "model-00002-of-00002.safetensors", [("x", "F32", [1], 4)])
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model-00001-of-00001.safetensors", "y": "model-00002-of-00002.safetensors"}}))
        with self.assertRaisesRegex(CheckpointError, "duplicate tensor"):
            inspect_checkpoint(root)


if __name__ == "__main__":
    unittest.main()
