import hashlib
import json
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from inspect_checkpoint import inspect_checkpoint
from export_expert_fixture import ExportError, export_fixture


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "config.json").write_text(json.dumps({"model_type": "ling", "architectures": ["LingForCausalLM"]}))
        shapes = {"gate_proj": [512, 1536], "up_proj": [512, 1536], "down_proj": [1536, 512]}
        header, payload, offset = {}, bytearray(), 0
        for name, shape in shapes.items():
            size = shape[0] * shape[1] * 2
            header[f"model.layers.1.mlp.experts.0.{name}.weight"] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
            payload.extend(bytes([len(name)]) * size)
            offset += size
        encoded = json.dumps(header, separators=(",", ":")).encode()
        (self.root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in header}}))
        self.report = self.root / "report.json"
        self.report.write_text(json.dumps(inspect_checkpoint(self.root)))

    def test_exact_export_manifest_and_bytes(self):
        output = self.root / "out"
        manifest = export_fixture(self.root, self.report, output, "ling")
        self.assertTrue(manifest["sample_exact_bytes_only"])
        self.assertFalse(manifest["quantized"])
        self.assertEqual(manifest["total_bytes"], 3 * 512 * 1536 * 2)
        for short, entry in manifest["tensors"].items():
            data = (output / entry["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
            self.assertEqual(len(data), entry["bytes"])

    def test_bad_shape_rejected(self):
        report = json.loads(self.report.read_text())
        report["tensors"]["model.layers.1.mlp.experts.0.gate_proj.weight"]["shape"] = [1, 1]
        bad = self.root / "bad.json"
        bad.write_text(json.dumps(report))
        with self.assertRaisesRegex(ExportError, "unexpected"):
            export_fixture(self.root, bad, self.root / "out", "ling")

    def test_header_drift_rejected(self):
        shard = self.root / "model.safetensors"
        data = bytearray(shard.read_bytes())
        header_len = struct.unpack("<Q", data[:8])[0]
        pos = data.find(b'"dtype":"BF16"', 8, 8 + header_len)
        data[pos + len('"dtype":"') : pos + len('"dtype":"') + 4] = b"F32x"
        shard.write_bytes(data)
        with self.assertRaisesRegex(ExportError, "header changed"):
            export_fixture(self.root, self.report, self.root / "out", "ling")

    def test_report_offset_substitution_rejected(self):
        report = json.loads(self.report.read_text())
        entry = report["tensors"]["model.layers.1.mlp.experts.0.gate_proj.weight"]
        entry["data_offsets"] = [2, entry["data_offsets"][1] + 2]
        bad = self.root / "offset.json"
        bad.write_text(json.dumps(report))
        with self.assertRaisesRegex(ExportError, "header differs"):
            export_fixture(self.root, bad, self.root / "out", "ling")

    def test_unsafe_shard_path_rejected(self):
        report = json.loads(self.report.read_text())
        report["tensors"]["model.layers.1.mlp.experts.0.gate_proj.weight"]["shard"] = "../escape.safetensors"
        bad = self.root / "unsafe.json"
        bad.write_text(json.dumps(report))
        with self.assertRaisesRegex(ExportError, "unsafe shard path"):
            export_fixture(self.root, bad, self.root / "out", "ling")

    def test_f_nocache_failure_is_fail_closed(self):
        with patch("export_expert_fixture.sys.platform", "darwin"), patch("fcntl.fcntl", side_effect=OSError("denied")):
            with self.assertRaisesRegex(ExportError, "F_NOCACHE"):
                export_fixture(self.root, self.report, self.root / "out", "ling")

    def test_existing_output_refused(self):
        output = self.root / "out"
        output.mkdir()
        with self.assertRaisesRegex(ExportError, "existing output"):
            export_fixture(self.root, self.report, output, "ling")


if __name__ == "__main__":
    unittest.main()
