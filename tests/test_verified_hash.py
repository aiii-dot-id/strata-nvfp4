import hashlib,os
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
import importlib.util
spec=importlib.util.spec_from_file_location("verified_hash",REPO/"tools/ling_ring_bf16/verified_hash.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def test_serial_parallel_order_and_bytes(tmp_path):
 files=[]
 for n,b in (("z.bin",b"z"*17),("a.bin",b"abc"),("m.bin",b"middle")):
  p=tmp_path/n;p.write_bytes(b);files.append(p)
 want=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
 assert m.sha256_files(files,1)==want and m.sha256_files(files,4)==want
def test_missing_and_directory_fail_closed(tmp_path):
 p=tmp_path/"p";p.write_bytes(b"x")
 for bad in (tmp_path/"missing",tmp_path/"dir"):
  if bad.name=="dir":bad.mkdir()
  for workers in (1,4):
   try:m.sha256_files([p,bad],workers)
   except (FileNotFoundError,IsADirectoryError):pass
   else:raise AssertionError("unexpected digest success")
def test_worker_bounds():
 for x in (False,1.0,0,-1,5,None):
  try:m.require_hash_workers(x)
  except ValueError:pass
  else:raise AssertionError(x)

def test_hash_failure_propagates_without_partial_success(tmp_path, monkeypatch):
 files=[tmp_path/"a",tmp_path/"b"]
 for f in files:f.write_text("test")
 original=m.sha256_file
 def fail(path):
  if path.name=="b":raise RuntimeError("injected hash failure")
  return original(path)
 monkeypatch.setattr(m,"sha256_file",fail)
 import pytest
 for workers in (1,4):
  with pytest.raises(RuntimeError,match="injected hash failure"):
   m.sha256_files(files,workers)
