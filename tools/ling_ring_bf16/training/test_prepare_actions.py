import json, subprocess, tempfile, unittest, sys
from pathlib import Path

ROOT=Path(__file__).parents[3]
CASES=Path("/work/AI/experiments/strata-corrected-action-20261021-r2/cases.json")
TOK=Path("/work/AI/checkpoints")
SCRIPT=ROOT/"tools/ling_ring_bf16/training/prepare_actions.py"

class PrepareTests(unittest.TestCase):
  def test_freeze_counts_boundaries_and_hidden_labels(self):
    with tempfile.TemporaryDirectory() as d:
      out=Path(d)/"prepared.json"; ret=Path(d)/"retention.json"
      subprocess.run([sys.executable,str(SCRIPT),"--cases",str(CASES),"--output",str(out),"--retention-output",str(ret),"--tokenizer-dir",str(TOK)],check=True)
      data=json.loads(out.read_text()); rec=data["records"]
      self.assertEqual(len(rec),80); self.assertTrue(all(r["record"]["target_boundary_verified"] for r in rec))
      self.assertTrue(all("family" not in r["record"]["prompt"] and "split" not in r["record"]["prompt"] for r in rec))
      self.assertTrue(all(r["record"]["target_ids"][-1] == data["tokenizer_pins"][r["model"]]["eos_id"] for r in rec))
      self.assertEqual(len(json.loads(ret.read_text())["records"]),8)
  def test_prompt_prefix_is_identical_and_explicit(self):
    with tempfile.TemporaryDirectory() as d:
      out=Path(d)/"p.json"; ret=Path(d)/"r.json"
      subprocess.run([sys.executable,str(SCRIPT),"--cases",str(CASES),"--output",str(out),"--retention-output",str(ret),"--tokenizer-dir",str(TOK)],check=True)
      data=json.loads(out.read_text()); self.assertTrue(all(r["record"]["controller_prefix"]==("</think>\n" if r["model"]=="ring" else "") for r in data["records"]))
      self.assertTrue(all("<think>" in r["record"]["prompt"] for r in data["records"]))

  def test_all_forty_oracles_match_independent_rules(self):
    cases=json.loads(CASES.read_text())["cases"]
    expected={c["id"]:(f"setcwd path={c['state']['target']}" if c["family"]=="wrongcwd" else f"inspect path={c['state']['parent']}" if c["family"]=="missingpath" else f"read path={c['state']['target']}") for c in cases}
    self.assertEqual(len(expected),40)
    for c in cases:
      self.assertTrue(expected[c["id"]].startswith(("setcwd path=/workspace/","inspect path=/workspace/","read path=/workspace/")))

if __name__=="__main__": unittest.main()
