import importlib.util,tempfile,pathlib,unittest
spec=importlib.util.spec_from_file_location('control',pathlib.Path(__file__).with_name('camera_session_control.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class Tests(unittest.TestCase):
 def run_worker(self,body):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=pathlib.Path(tmp.name)
  worker=root/'worker.py';worker.write_text("import argparse,json,time,sys\nfrom pathlib import Path\np=argparse.ArgumentParser()\nfor k in ['output','stop-file','ready-file','max-seconds']:p.add_argument('--'+k)\na=p.parse_args()\nout=Path(a.output);out.mkdir()\n"+body)
  s=mod.CameraSession(root/'dataset',worker_path=worker);self.addCleanup(s._terminate);return s
 def test_start_failure(self):
  s=self.run_worker('sys.exit(7)\n');r=s.prepare(timeout=2);self.assertFalse(r['ready']);self.assertIn('code 7',r['error'])
 def test_stop_failure(self):
  s=self.run_worker("Path(a.ready_file).write_text(json.dumps({'host_monotonic_s':time.monotonic(),'state':'capturing'}))\nwhile not Path(a.stop_file).exists():time.sleep(.01)\n(out/'result.json').write_text(json.dumps({'success':False,'error':'encoding failed'}))\nsys.exit(4)\n")
  self.assertTrue(s.prepare(timeout=2)['ready']);self.assertTrue(s.check_running()['running']);r=s.stop(timeout=2);self.assertFalse(r['success']);self.assertIn('encoding failed',r['error'])
 def test_stop_timeout(self):
  s=self.run_worker("Path(a.ready_file).write_text(json.dumps({'host_monotonic_s':time.monotonic(),'state':'capturing'}))\ntime.sleep(20)\n")
  self.assertTrue(s.prepare(timeout=2)['ready']);r=s.stop(timeout=.1);self.assertFalse(r['success']);self.assertIn('timed out',r['error']);self.assertIsNotNone(s.process.poll())
 def test_stopped_stale_and_malformed(self):
  s=self.run_worker("Path(a.ready_file).write_text(json.dumps({'host_monotonic_s':time.monotonic(),'state':'capturing'}))\ntime.sleep(20)\n")
  self.assertTrue(s.prepare(timeout=2)['ready'])
  import json,time
  for status in [{'host_monotonic_s':time.monotonic(),'state':'stopped'}, {'host_monotonic_s':time.monotonic()-4,'state':'capturing'}, {'host_monotonic_s':True,'state':'capturing'}, []]:
   s.ready_file.write_text(json.dumps(status));self.assertFalse(s.is_running())
  s.ready_file.write_text('{');self.assertFalse(s.is_running())
 def test_success(self):
  s=self.run_worker("Path(a.ready_file).write_text(json.dumps({'host_monotonic_s':time.monotonic(),'state':'capturing'}))\nwhile not Path(a.stop_file).exists():time.sleep(.01)\n(out/'result.json').write_text(json.dumps({'success':True,'frames':3}))\n")
  self.assertTrue(s.prepare(timeout=2)['ready']);r=s.stop(timeout=2);self.assertTrue(r['success']);self.assertEqual(r['frames'],3)
if __name__=='__main__':unittest.main()
