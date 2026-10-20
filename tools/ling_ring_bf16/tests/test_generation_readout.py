import hashlib,json,pathlib,struct,tempfile,unittest,sys
TOOLS=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TOOLS)); import generation_readout as g

class Controls(unittest.TestCase):
 def setUp(self): self.t=tempfile.TemporaryDirectory()
 def tearDown(self): self.t.cleanup()
 def make(self, **kw):
  d=pathlib.Path(self.t.name)/str(len(list(pathlib.Path(self.t.name).glob("*")))); d.mkdir()
  vocab=kw.get("vocab",16); P=kw.get("P",[1,2]); G=kw.get("G",[3,4,9]); eos=kw.get("eos",[9]); rows=len(P)+len(G)-1
  vals=[]
  winners=[0,3,4,9][:rows]
  for r in range(rows):
   x=[0.0]*vocab; winner=winners[r]
   if kw.get("shift") and r==1: winner=5
   x[winner]=10.0
   if kw.get("tie") and r==1: x[5]=10.0
   vals.extend(x)
  if kw.get("nonfinite"): vals[0]=float("nan")
  lp=d/"logits.f32"; lp.write_bytes(struct.pack("<"+"f"*len(vals),*vals))
  e={"logits_tokens":kw.get("rows",rows),"tokens":kw.get("tokens",rows),"token_ids":kw.get("token_ids",P+G[:-1]),"model":"fixture-engine"}
  md={"sha256":g.sha(lp),"size":lp.stat().st_size}
  if kw.get("stale"): md["sha256"]="0"*64
  p={"status":"verified","model":"fixture","execution":e,"continuation":{"start_position":kw.get("start",0)},"generation":{"prompt_token_ids":P,"generated_token_ids":G,"eos_token_ids":eos,"max_new_tokens":kw.get("max_new",len(G)),"finish_reason":kw.get("reason","eos")},"native_outputs":{"logits.f32":md}}
  (d/"execution.json").write_text(json.dumps(e)); (d/"provenance.json").write_text(json.dumps(p)); return d
 def test_positive_tie_and_contract(self):
  r=g.readout(self.make(tie=True),vocab_domain=16); self.assertEqual(r["rows"][0]["top2_ids"][0],3)
  with self.assertRaises(ValueError): g.readout(self.make(vocab=16),vocab_domain=None)
  d=self.make(); p=json.loads((d/"provenance.json").read_text()); p.pop("model"); (d/"provenance.json").write_text(json.dumps(p)); self.assertEqual(g.readout(d,vocab_domain=16)["model"],"fixture-engine")
  with self.assertRaises(ValueError): g.readout(self.make(tokens=99),vocab_domain=16)
 def test_shift_offbyone_and_token_binding(self):
  with self.assertRaises(ValueError): g.readout(self.make(shift=True),vocab_domain=16)
  with self.assertRaises(ValueError): g.readout(self.make(token_ids=[1,2,3]),vocab_domain=16)
 def test_hash_nonfinite_expected_and_truncated(self):
  with self.assertRaises(ValueError): g.readout(self.make(nonfinite=True),vocab_domain=16)
  d=self.make(); (d/"logits.f32").write_bytes((d/"logits.f32").read_bytes()[:-4])
  with self.assertRaises(ValueError): g.readout(d,vocab_domain=16)
  with self.assertRaises(ValueError): g.readout(self.make(stale=True),vocab_domain=16)
  with self.assertRaises(ValueError): g.readout(self.make(),vocab_domain=16,expected_provenance_sha256="0"*64)
 def test_resume_budget_interior_eos_and_bad_rows(self):
  with self.assertRaises(ValueError): g.readout(self.make(start=1),vocab_domain=16)
  with self.assertRaises(ValueError): g.readout(self.make(G=[9,4,9]),vocab_domain=16)
  with self.assertRaises(ValueError): g.readout(self.make(max_new=0),vocab_domain=16)
  with self.assertRaises(ValueError): g.readout(self.make(rows=3),vocab_domain=16)

 def test_cli_domain_and_output_protection(self):
  d=self.make(); out=pathlib.Path(self.t.name)/"report.json"
  old=sys.argv; sys.argv=["generation_readout.py",str(d),"--output",str(out)]
  try:
   with self.assertRaises(ValueError): g.main()
  finally: sys.argv=old
  self.assertFalse(out.exists())
  inside=d/"report.json"; old=sys.argv; sys.argv=["generation_readout.py",str(d),"--output",str(inside)]
  try:
   with self.assertRaises(ValueError): g.main()
  finally: sys.argv=old
  old=sys.argv; sys.argv=["generation_readout.py",str(d),"--output",str(out)]
  saved=g.readout; g.readout=lambda *a,**k: {"ok":True}
  try:
   out.write_text("existing")
   with self.assertRaises(FileExistsError): g.main()
  finally: g.readout=saved; sys.argv=old

 def test_mutation_after_sampling_is_rejected(self):
  d=self.make(); original=g.top2; calls=[0]
  def mutate(row):
   calls[0]+=1
   result=original(row)
   if calls[0]==1: (d/"execution.json").write_text((d/"execution.json").read_text()+" ")
   return result
  g.top2=mutate
  try:
   with self.assertRaises(ValueError): g.readout(d,vocab_domain=16)
  finally: g.top2=original

if __name__=="__main__": unittest.main(verbosity=2)
