import pathlib, subprocess, sys, tempfile, time, unittest
from wrapper_observer import WrapperObserver

class WrapperObserverTests(unittest.TestCase):
 def test_detects_real_descendant_exact_executable(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d); native=pathlib.Path('/bin/sleep').resolve()
   code="import subprocess,sys,time; subprocess.run(['/bin/sleep','1.2'])"
   wrapper=subprocess.Popen([sys.executable,'-c',code])
   obs=WrapperObserver(root/'result.json',poll_seconds=1,gpu_telemetry=False); obs.start(wrapper.pid,native)
   self.assertEqual(wrapper.wait(),0); result=obs.finish()
   self.assertEqual(result['envelopes']['native_polling_envelope']['status'],'observed')
   self.assertTrue(any(row['exact_native_pids'] for row in result['samples']))
 def test_missing_native_is_unavailable_not_zero(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d); wrapper=subprocess.Popen([sys.executable,'-c','import time; time.sleep(.1)'])
   obs=WrapperObserver(root/'result.json',poll_seconds=1,gpu_telemetry=False); obs.start(wrapper.pid,pathlib.Path('/bin/sleep'))
   wrapper.wait(); result=obs.finish()
   self.assertEqual(result['envelopes']['native_polling_envelope']['status'],'unavailable_exact_executable_never_observed')
   self.assertNotIn('first_observed_monotonic_seconds',result['envelopes']['native_polling_envelope'])
if __name__=='__main__': unittest.main()
