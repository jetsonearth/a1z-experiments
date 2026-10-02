import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from teach import finalize_episode

class FinalizationTests(unittest.TestCase):
    def run_case(self, failure=None):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'episode.json'
            samples=[{'q':[0]*6,'t_monotonic_s':1.0}]
            class Camera:
                output=Path(temp)/'dataset'
                def stop(self,timeout):
                    saved=json.loads(path.read_text())
                    assert saved['samples']==samples
                    assert saved['metadata']['finalization_complete'] is False
                    if failure is not None: raise failure
                    return {'frames':3,'error':None,'success':True}
            meta={'status':'motion_complete','camera_recorded':False}
            with contextlib.redirect_stdout(io.StringIO()):finalize_episode(path,meta,samples,Camera())
            result=json.loads(path.read_text())
            self.assertEqual(result['samples'],samples)
            self.assertTrue(result['metadata']['finalization_complete'])
            self.assertEqual(result['metadata']['status'],'motion_complete')
            self.assertEqual(result['metadata']['camera_recorded'],failure is None)
            self.assertFalse(path.with_suffix('.incoming').exists())
            return result
    def test_joints_durable_before_encoding(self):self.run_case()
    def test_keyboard_interrupt_keeps_joints(self):self.assertIn('camera_error',self.run_case(KeyboardInterrupt())['metadata'])
    def test_encoder_failure_keeps_joints(self):self.assertIn('camera_error',self.run_case(RuntimeError('encoder failed'))['metadata'])

if __name__=='__main__':unittest.main()
