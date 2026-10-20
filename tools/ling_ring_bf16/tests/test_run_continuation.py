# AI Identity Incorported <james@aiii.id>
import zlib, hashlib, importlib, json, pathlib, shutil, struct, sys, tempfile, types, unittest
from unittest.mock import patch

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
run_continuation = importlib.import_module("run_continuation")

TOKENS = [11, 22, 33, 44]
BINDING_INPUTS = {"fake-input": {"sha256": "a" * 64, "size": 1}}

def u64(n):
    return struct.pack("<Q", n)

def checksum(data):
    return zlib.crc32(data)

def write_state(path, binding, prefix, *, bad_checksum=False, bad_prefix=None, missing_array=False):
    payload = bytearray()
    def text(value):
        raw = value.encode("ascii")
        payload.extend(u64(len(raw))); payload.extend(raw)
    def vector(values):
        payload.extend(u64(len(values)))
        for value in values: payload.extend(struct.pack("<f", value))
    text("STRATA-CONT-1"); text(binding); payload.extend(u64(len(prefix))); payload.extend(u64(len(prefix)))
    for token in (bad_prefix if bad_prefix is not None else prefix): payload.extend(u64(token))
    vector([0.0] * run_continuation.VOCAB if prefix else [])
    payload.extend(u64(1 if missing_array else 0))
    crc = checksum(payload)
    if bad_checksum: crc ^= 1
    payload.extend(u64(crc))
    pathlib.Path(path).write_bytes(payload)

def binding_for(run, model="ling"):
    return {"schema":"strata-continuation-binding-v1","model":model,"inputs":BINDING_INPUTS,"profile":run.profile(model, 1024),"activation_dtype":"FP32","quantized":False,"eos_token_ids":[],"adapter":"none"}

class ContinuationControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="continuation-controls-")
        self.root = pathlib.Path(self.tmp.name)
        self.checkpoint = self.root / "checkpoint"; self.checkpoint.mkdir()
        (self.checkpoint / "tokenizer.json").write_text("fixture")
        self.manifest = self.root / "manifest.tsv"; self.manifest.write_text("fixture-manifest\n")
        self.exe = self.root / "ling.exe"; self.exe.write_bytes(b"fixture-exe"); self.exe.chmod(0o755)
        self.original = {
            "prepare": run_continuation.ling.prepare,
            "inputs": run_continuation.input_bindings,
            "validate_outputs": run_continuation.validate_outputs,
            "run_owned": run_continuation.ling._run_owned,
            "tokens": run_continuation.ling._tokens,
            "ling_all_inputs": run_continuation.ling._all_inputs,
            "ring_all_inputs": run_continuation.ring._all_inputs,
        }
        run_continuation.ling.prepare = lambda cp, out: (pathlib.Path(out).mkdir(parents=True), pathlib.Path(out, "manifest.tsv").write_bytes(self.manifest.read_bytes()))
        run_continuation.input_bindings = lambda *args: dict(BINDING_INPUTS)
        run_continuation.ling._tokens = lambda value: TOKENS
        fake_front = types.SimpleNamespace(_binding_paths=lambda cp: [cp / "tokenizer.json"], _eos_ids=lambda cp, ids: ([], []), _tokenizer=lambda cp: None)
        self.front_patch = patch.object(run_continuation, "frontend_module", return_value=fake_front)
        self.front_patch.start()
        run_continuation.ling._all_inputs = lambda *args, **kwargs: dict(BINDING_INPUTS)
        run_continuation.ring._all_inputs = lambda *args, **kwargs: dict(BINDING_INPUTS)
        run_continuation.ling._run_owned = self.fake_native

    def tearDown(self):
        run_continuation.ling.prepare = self.original["prepare"]
        run_continuation.input_bindings = self.original["inputs"]
        run_continuation.validate_outputs = self.original["validate_outputs"]
        run_continuation.ling._run_owned = self.original["run_owned"]
        run_continuation.ling._tokens = self.original["tokens"]
        run_continuation.ling._all_inputs = self.original["ling_all_inputs"]
        run_continuation.ring._all_inputs = self.original["ring_all_inputs"]
        self.front_patch.stop(); self.tmp.cleanup()

    def fake_native(self, argv, timeout):
        out = pathlib.Path(argv[4]); out.mkdir()
        all_tokens = [int(x) for x in argv[3].split(",")]
        start = 0
        if "--state-in" in argv:
            state = run_continuation.state_info(pathlib.Path(argv[argv.index("--state-in") + 1]))
            start = state["position"]
        end = int(argv[argv.index("--stop-after") + 1]) if "--stop-after" in argv else len(all_tokens)
        binding = argv[argv.index("--state-binding") + 1]
        for i in range(24): (out / f"layer-{i}.f32").write_bytes(b"\0" * ((end-start) * 1536 * 4))
        (out / "final-hidden.f32").write_bytes(b"\0" * ((end-start) * 1536 * 4))
        (out / "logits.f32").write_bytes(b"\0" * ((end-start) * 157184 * 4))
        with (out / "routes.tsv").open("w") as f:
            for pos in range(start, end):
                for layer in range(1, 24): f.write(f"{pos}\t{layer}\t" + "\t".join(f"{i}:0.125" for i in range(8)) + "\n")
        execution = {"status":"executed","model":"Ling-3.0-tiny","activation_dtype":"FP32","quantized":False,"layers":24,"checkpoint_id":run_continuation.sha(self.manifest),"rope_frequency":"native","rms_mode":"torch-cuda-order","matmul_backend":"fp32-cublas","pointwise_mode":"torch-cuda","lm_head_residency":"cuda","cache_budget_bytes":1024*1024*1024,"tokens":end,"token_ids":all_tokens[:end],"logits_mode":"all","logits_tokens":end-start,"lm_head_resident_uploads":1,"lm_head_resident_apply_calls":end-start}
        (out / "execution.json").write_text(json.dumps(execution))
        cont={"start_position":start,"processed_position":end,"processed_tokens":end-start,"paused":end < len(all_tokens)}
        (out / "continuation.json").write_text(json.dumps(cont))
        if "--state-out" in argv: write_state(out / "native-state.bin", binding, all_tokens[:end])
        return 0, "", ""

    def run_partial(self):
        out = self.root / "partial"
        return run_continuation.run("ling", self.checkpoint, self.manifest, self.exe, out, tokens=",".join(map(str, TOKENS)), stop_after=2, max_new_tokens=0)

    def test_success_partial_resume_and_receipts(self):
        partial = self.run_partial(); snapshot = self.root / "partial" / "snapshot"
        self.assertEqual(partial["continuation"]["processed_position"], 2)
        self.assertEqual(run_continuation.load_snapshot(snapshot)["state"]["position"], 2)
        resumed = run_continuation.run("ling", self.checkpoint, self.manifest, self.exe, self.root / "resumed", tokens=",".join(map(str, TOKENS)), state_in=snapshot, max_new_tokens=0)
        self.assertEqual(resumed["continuation"]["processed_position"], 4)
        self.assertEqual(resumed["execution"]["token_ids"], TOKENS)
        self.assertEqual(resumed["status"], "verified")

    def test_wrong_binding_and_mutated_snapshot(self):
        self.run_partial(); snap = self.root / "partial" / "snapshot"
        meta = run_continuation.read_json(snap / "manifest.json"); meta["binding_id"] = "0" * 64; run_continuation.save_json(snap / "manifest.json", meta)
        with self.assertRaises(run_continuation.StateError): run_continuation.load_snapshot(snap)
        shutil.rmtree(self.root / "partial")
        self.run_partial(); snap = self.root / "partial" / "snapshot"; data = bytearray((snap / "state.bin").read_bytes()); data[20] ^= 1; (snap / "state.bin").write_bytes(data)
        with self.assertRaises(run_continuation.StateError): run_continuation.load_snapshot(snap)

    def test_wrong_prefix_and_fork_independence(self):
        self.run_partial(); snap = self.root / "partial" / "snapshot"; meta = run_continuation.read_json(snap / "manifest.json"); info = run_continuation.state_info(snap / "state.bin");
        bad = self.root / "bad"; bad.mkdir(); write_state(bad / "state.bin", info["binding_id"], info["prefix"], bad_prefix=[99, 22]); run_continuation.save_json(bad / "manifest.json", dict(meta, state=run_continuation.state_info(bad / "state.bin")))
        with self.assertRaises(run_continuation.StateError):
            run_continuation.run("ling", self.checkpoint, self.manifest, self.exe, self.root / "bad-resume", tokens=",".join(map(str, TOKENS)), state_in=bad)
        fork = self.root / "fork"; run_continuation.fork_snapshot(snap, fork); self.assertNotEqual((snap / "state.bin").stat().st_ino, (fork / "state.bin").stat().st_ino); (fork / "state.bin").write_bytes((fork / "state.bin").read_bytes() + b"x"); self.assertNotEqual((snap / "state.bin").stat().st_size, (fork / "state.bin").stat().st_size)

    def test_output_exists_and_native_failure(self):
        out = self.root / "existing"; out.mkdir()
        with self.assertRaises(run_continuation.StateError): run_continuation.run("ling", self.checkpoint, self.manifest, self.exe, out, tokens=",".join(map(str, TOKENS)))
        run_continuation.ling._run_owned = lambda argv, timeout: (2, "", "failed")
        with self.assertRaises(run_continuation.StateError): run_continuation.run("ling", self.checkpoint, self.manifest, self.exe, self.root / "failed", tokens=",".join(map(str, TOKENS)))

    def test_raw_state_missing_array_and_nonfinite_file(self):
        path = self.root / "state.bin"; write_state(path, "a" * 64, [1, 2]); raw = bytearray(path.read_bytes()); path.write_bytes(raw[:-9])
        with self.assertRaises(run_continuation.StateError): run_continuation.state_info(path)
        missing = self.root / "missing-array.bin"; write_state(missing, "a" * 64, [1, 2], missing_array=True)
        with self.assertRaises(run_continuation.StateError): run_continuation.state_info(missing)
        bad = self.root / "nan.f32"; bad.write_bytes(struct.pack("<f", float("nan")))
        with self.assertRaises(run_continuation.StateError): run_continuation.finite_file(bad, 4)
        snap = self.root / "missing"; snap.mkdir(); (snap / "manifest.json").write_text("{}")
        with self.assertRaises(run_continuation.StateError): run_continuation.load_snapshot(snap)

    def test_bad_native_checksum_not_sealed(self):
        path=self.root / "bad-crc.bin";write_state(path,"a"*64,[1,2],bad_checksum=True)
        with self.assertRaisesRegex(run_continuation.StateError,"CRC32"):run_continuation.state_info(path)

    def test_wrong_route_ordering_control(self):
        out = self.root / "route"; out.mkdir(); count=1
        for i in range(24): (out / f"layer-{i}.f32").write_bytes(b"\0" * (count * 1536 * 4))
        (out / "final-hidden.f32").write_bytes(b"\0" * (count * 1536 * 4)); (out / "logits.f32").write_bytes(b"\0" * (count * 157184 * 4))
        with (out / "routes.tsv").open("w") as f:
            for layer in range(1,24): f.write(f"0\t{layer+1}\t" + "\t".join(f"{i}:0.125" for i in range(8)) + "\n")
        with self.assertRaises(run_continuation.StateError): run_continuation.validate_outputs(out,"ling",0,1)

if __name__ == "__main__": unittest.main(verbosity=2)
