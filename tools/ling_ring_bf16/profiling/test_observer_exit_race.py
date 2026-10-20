"""AI Identity Incorported <james@aiii.id>. Exited-process observer controls."""
import errno, importlib.util, pathlib, tempfile, unittest
ROOT=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('observer',ROOT/'wrapper_observer.py')
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)

class Patch:
 def __init__(self,obj,name,value): self.obj,self.name,self.value=obj,name,value
 def __enter__(self): self.old=getattr(self.obj,self.name);setattr(self.obj,self.name,self.value);return self
 def __exit__(self,*_): setattr(self.obj,self.name,self.old)

class ObserverExitRace(unittest.TestCase):
 def test_stat_esrch_is_benign(self):
  with Patch(pathlib.Path,'read_text',lambda _: (_ for _ in ()).throw(ProcessLookupError())):
   self.assertIsNone(o._stat(123))
 def test_rss_esrch_and_missing_are_benign(self):
  for exc in (ProcessLookupError,FileNotFoundError):
   with Patch(pathlib.Path,'read_text',lambda _: (_ for _ in ()).throw(exc())):
    self.assertIsNone(o._rss_kib(123))
 def test_stat_permission_and_eio_propagate(self):
  for exc in (PermissionError('denied'), OSError(errno.EIO,'io')):
   with Patch(pathlib.Path,'read_text',lambda _, e=exc: (_ for _ in ()).throw(e)):
    with self.assertRaises(type(exc)): o._stat(123)
 def test_rss_permission_and_eio_propagate(self):
  for exc in (PermissionError('denied'), OSError(errno.EIO,'io')):
   with Patch(pathlib.Path,'read_text',lambda _, e=exc: (_ for _ in ()).throw(e)):
    with self.assertRaises(type(exc)): o._rss_kib(123)
 def test_lineage_exit_race_and_pid_reuse_are_not_owned(self):
  with Patch(o,'_all_stats',lambda: {}): self.assertEqual(o._lineage(40,10),{})
  with Patch(o,'_all_stats',lambda: {40:(1,40,11),41:(40,40,12)}): self.assertEqual(o._lineage(40,10),{})
 def test_sample_survives_lineage_exit_race(self):
  with tempfile.TemporaryDirectory() as td:
   w=o.WrapperObserver(pathlib.Path(td)/'profile.json',gpu_telemetry=False)
   w._wrapper=40;w._wrapper_start_ticks=10;w._native='/native';w._start=o.time.monotonic()
   with Patch(o,'_lineage',lambda *_: {}), Patch(o,'_stat',lambda *_:None):
    w._sample()
   self.assertEqual(w._samples[0]['owned_lineage_processes'],[])

if __name__=='__main__': unittest.main()
