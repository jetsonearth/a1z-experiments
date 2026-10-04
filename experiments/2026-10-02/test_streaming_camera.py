"""Real FFmpeg tests matching short task and longer preparation/support capture."""
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from camera_session import StreamingVideo,probe_video,parse_args
from camera_session_control import CameraSession

@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
class StreamingTests(unittest.TestCase):
    def test_14_second_and_42_second_streams_finalize_without_png_batch(self):
        for count in [420,1258]:
            with self.subTest(frames=count),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);writer=StreamingVideo(root/'videos'/'wrist.mp4',width=64,height=48)
                for n in range(count):writer.write(bytes([n%256,70,120])*(64*48))
                start=time.monotonic();writer.close();elapsed=time.monotonic()-start
                check=probe_video(writer.path,root,count)
                self.assertEqual(int(check['decoded']['streams'][0]['nb_read_packets']),count)
                self.assertLess(elapsed,5,'Stopping must only flush encoded packets')
                self.assertEqual(list(root.rglob('*.png')),[])

    def test_default_is_unlimited_and_optional_long_duration_is_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            default=CameraSession(Path(temp)/'default')
            self.assertIsNone(parse_args(default.command()[2:]).max_seconds)
            bounded=CameraSession(Path(temp)/'bounded',max_seconds=3600)
            self.assertEqual(parse_args(bounded.command()[2:]).max_seconds,3600)

if __name__=='__main__':unittest.main()
