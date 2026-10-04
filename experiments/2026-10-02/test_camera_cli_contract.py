"""Run collector camera construction through the real worker CLI, without hardware."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import camera_session
from camera_session_control import CameraSession
import collect_task

class CameraCLIContractTests(unittest.TestCase):
    def test_collector_arguments_are_accepted_by_real_camera_worker(self):
        checked=[]
        class ValidateOnlyCamera(CameraSession):
            def prepare(self):
                # Same command as real prepare(), parsed by actual worker parser.
                checked.append(camera_session.parse_args(self.command()[2:]))
                return {'ready':False,'error':'TEST_STOP_BEFORE_HARDWARE'}
        with tempfile.TemporaryDirectory() as tmp,patch.object(collect_task,'__file__',str(Path(tmp)/'collect_task.py')),patch.object(collect_task,'verify_revision',return_value='test'),patch.object(collect_task,'CameraSession',ValidateOnlyCamera),patch.object(collect_task,'build_robot') as robot,patch('builtins.input',return_value='READY'),contextlib.redirect_stdout(io.StringIO()) as output:
            collect_task.main()
            self.assertEqual(len(checked),1,'Collector passed camera arguments rejected by the actual worker CLI')
            robot.assert_not_called()
            self.assertIn('stage=camera_startup controller_started=False',output.getvalue())
            self.assertIn('TEST_STOP_BEFORE_HARDWARE',output.getvalue())
            self.assertIn('TASK_SESSION_COMPLETE 0',output.getvalue())

    def test_negative_optional_duration_is_rejected_before_hardware_imports(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stderr(io.StringIO()):
            camera=CameraSession(Path(tmp)/'capture',max_seconds=-1)
            with self.assertRaises(SystemExit) as raised:
                camera_session.parse_args(camera.command()[2:])
            self.assertEqual(raised.exception.code,2)

if __name__=='__main__':unittest.main()
