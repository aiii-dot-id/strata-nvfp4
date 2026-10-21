import json, os, signal, tempfile, threading, time, unittest
from pathlib import Path
from unittest.mock import patch

import native

class FakeProc:
    pid=999991
    returncode=None
    def poll(self): return self.returncode
    def wait(self, timeout=None): self.returncode=-15; return self.returncode

class NativeTests(unittest.TestCase):
    def test_generation_receipt_rejects_bad_terminal_semantics(self):
        good={"schema":"strata-ling-generation-v1","prompt_token_ids":[1],"generated_token_ids":[2,9],"eos_token_ids":[9],"max_new_tokens":2,"finish_reason":"eos"}
        native._validate_generation(good,[1],2,[9],[1,2])
        for key,value in (("prompt_token_ids",[8]),("eos_token_ids",[7]),("finish_reason","length")):
            bad=dict(good); bad[key]=value
            with self.assertRaises(RuntimeError): native._validate_generation(bad,[1],2,[9],[1,2])

    def test_adapter_metadata_requires_model_and_manifest_digest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.lora"; p.write_bytes(b"adapter")
            meta={"model":"ling","hidden":1536,"checkpoint_sha256":"a"*64}
            Path(str(p)+".json").write_text(json.dumps(meta))
            native.MODELS["ling"]["manifest"] = str(Path(d)/"manifest")
            Path(native.MODELS["ling"]["manifest"]).write_bytes(b"manifest")
            digest=native.sha(native.MODELS["ling"]["manifest"])
            meta["checkpoint_sha256"]=digest; Path(str(p)+".json").write_text(json.dumps(meta))
            self.assertEqual(native._validate_adapter_meta(p,"ling",digest)["model"],"ling")
            meta["model"]="ring"; Path(str(p)+".json").write_text(json.dumps(meta))
            with self.assertRaises(RuntimeError): native._validate_adapter_meta(p,"ling",digest)

    def test_checkpoint_binding_rejects_wrong_digest(self):
        with tempfile.TemporaryDirectory() as d:
            manifest=Path(d)/"manifest"; manifest.write_text("manifest")
            old=native.MODELS["ling"]; native.MODELS["ling"]={**old,"manifest":str(manifest)}
            try:
                with self.assertRaises(ValueError):
                    native.run("ling",[1],Path(d)/"out",checkpoint_binding="0"*64)
            finally: native.MODELS["ling"]=old

    def test_sigterm_cleans_owned_mock_process_and_writes_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); exe=root/"fake"; exe.write_text("x"); manifest=root/"manifest"; manifest.write_text("m")
            old=native.MODELS["ling"]; native.MODELS["ling"]={**old,"manifest":str(manifest)}
            out=root/"out"; proc=FakeProc()
            timer=threading.Timer(.15,lambda: os.kill(os.getpid(),signal.SIGTERM)); timer.start()
            try:
                with patch.object(native,"_source_closure",return_value={}), patch.object(native,"_git_state",return_value={"commit":"x","dirty":False,"status":[]}), patch.object(native.subprocess,"Popen",return_value=proc), patch.object(native.os,"killpg") as killpg:
                    with self.assertRaises(RuntimeError) as cm:
                        native.run("ling",[1],out,executable=exe,timeout=5)
                self.assertIn("SIGTERM",str(cm.exception)); killpg.assert_called_once_with(proc.pid,signal.SIGTERM)
                self.assertTrue((root/"out.FAILED.json").is_file())
            finally:
                timer.cancel(); native.MODELS["ling"]=old

if __name__ == "__main__": unittest.main()
