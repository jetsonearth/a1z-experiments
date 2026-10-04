"""No hardware: multi-episode recording and manufacturer gripper message routing."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import collect_task
from teach import FeedbackBus

class Values(list):
    def tolist(self):return list(self)

class BufferTests(unittest.TestCase):
    def test_episode_boundaries_exclude_reset_and_do_not_share_storage(self):
        b=collect_task.EpisodeBuffer()
        self.assertIsNone(b.append({'reset':True}))
        b.begin();b.append({'episode':1})
        first=b.finish()
        self.assertIsNone(b.append({'reset':True}))
        b.begin();b.append({'episode':2});second=b.finish()
        self.assertEqual(first,[{'episode':1}]);self.assertEqual(second,[{'episode':2}])
        self.assertIsNone(b.finish())

    def test_repeated_begin_preserves_current_data(self):
        b=collect_task.EpisodeBuffer();b.begin();b.append(1)
        with self.assertRaises(ValueError):b.begin()
        self.assertEqual(b.finish(),[1])

class GripperRoutingTests(unittest.TestCase):
    def frame(self,mid=7,first=0x17,rx=True,extended=False):
        return SimpleNamespace(arbitration_id=mid,data=bytes([first])+b'\x80'*7,is_rx=rx,is_extended_id=extended,is_error_frame=False)

    def test_enabled_id7_reaches_sdk_without_touching_arm_freshness(self):
        msg=self.frame();raw=Mock();raw.recv.side_effect=[msg,None]
        bus=FeedbackBus(raw,clock=lambda:10,gripper_enabled=True)
        self.assertIs(bus.recv(),msg)
        self.assertEqual(bus.gripper_last,10)
        self.assertEqual(bus.last,[None]*6);self.assertEqual(bus.count,[0]*6)
        with self.assertRaises(RuntimeError):bus.require_fresh()

    def test_disabled_gripper_and_malformed_reports_are_filtered(self):
        for enabled,frames in [(False,[self.frame()]),(True,[self.frame(first=0x16),self.frame(rx=False),self.frame(extended=True)])]:
            raw=Mock();raw.recv.side_effect=frames+[None]
            bus=FeedbackBus(raw,gripper_enabled=enabled)
            self.assertIsNone(bus.recv());self.assertIsNone(bus.gripper_last)

    def test_missing_gripper_feedback_not_fabricated_from_command(self):
        g=SimpleNamespace(_motor=SimpleNamespace(last_feedback=None),get_pos=lambda:0.0,get_feedback_norm=lambda:0.0)
        result=collect_task.gripper_observation(SimpleNamespace(gripper=g))
        self.assertIsNone(result['gripper_position']);self.assertFalse(result['gripper_feedback_valid'])
        self.assertEqual(result['gripper_target'],0)

class ContinuousCaptureTests(unittest.TestCase):
    def test_two_episodes_and_gripper_commands_share_one_controller_camera(self):
        saved=[];events=[]
        robot=Mock();robot.is_running=True;robot.gravity_comp_factor=1.0
        robot._vel_limit=Values([1]*6);robot._torque_clip=Values([1]*6)
        target=[1.0]
        robot.gripper=SimpleNamespace(_motor=SimpleNamespace(last_feedback=object()),get_pos=lambda:target[0],get_feedback_norm=lambda:target[0])
        def command(v):target[0]=v;events.append(('gripper',v))
        robot.command_gripper.side_effect=command
        robot.get_joint_state.side_effect=lambda:{'pos':Values([0]*6),'vel':Values([0]*6),'eff':Values([0]*6)}
        robot.start.side_effect=lambda:events.append(('start',))
        robot.stop.side_effect=lambda:events.append(('stop',))
        bus=Mock();bus.gripper_last=time.monotonic();bus.ages.return_value=[0]*6
        camera=Mock();camera.prepare.return_value={'ready':True};camera.is_running.return_value=True
        camera.stop.return_value={'success':True,'video_checks':[{'file':'videos/wrist.mp4'}]}
        def persist(path,meta,rows):saved.append((str(path),copy.deepcopy(meta),copy.deepcopy(rows)))
        def commands():
            yield 'BEGIN\n';time.sleep(.09);yield 'END\n'
            self.assertEqual(robot.stop.call_count,0);self.assertEqual(camera.stop.call_count,0)
            self.assertEqual(len(saved),1);self.assertFalse(saved[0][1]['finalization_complete'])
            yield 'OPEN\n';yield 'CLOSE\n'
            time.sleep(.03) # unrecorded reset/preparation interval
            yield 'BEGIN\n';time.sleep(.09);yield 'END\n'
            self.assertEqual(robot.stop.call_count,0);self.assertEqual(camera.stop.call_count,0)
            yield 'DISABLE\n'
        with tempfile.TemporaryDirectory() as tmp,patch.object(collect_task,'__file__',str(Path(tmp)/'collect_task.py')),patch.object(collect_task,'verify_revision',return_value='sdk-test'),patch.object(collect_task,'build_robot',return_value=(robot,bus)) as build,patch.object(collect_task,'preflight'),patch.object(collect_task,'persist_episode',side_effect=persist),patch.object(collect_task,'CameraSession',return_value=camera),patch('builtins.input',return_value='READY'),patch('sys.stdin',commands()),contextlib.redirect_stdout(io.StringIO()) as output:
            camera.output=Path(tmp)/'shared-camera'
            collect_task.main()
        build.assert_called_once_with(with_gripper=True)
        robot.start.assert_called_once();robot.stop.assert_called_once();camera.prepare.assert_called_once();camera.stop.assert_called_once()
        self.assertEqual(events,[('start',),('gripper',1.0),('gripper',0.0),('stop',)])
        self.assertEqual(len(saved),4)
        first,second=saved[:2]
        self.assertGreaterEqual(len(first[2]),2);self.assertGreaterEqual(len(second[2]),2)
        self.assertTrue(all(r['gripper_target']==1 for r in first[2]))
        self.assertTrue(all(r['gripper_target']==0 for r in second[2]))
        self.assertEqual(first[1]['session_id'],second[1]['session_id'])
        self.assertNotEqual(first[0],second[0])
        self.assertTrue(all(item[1]['finalization_complete'] for item in saved[2:]))
        self.assertTrue(all(item[1]['camera_recorded'] for item in saved[2:]))
        self.assertIn('TASK_SESSION_COMPLETE 2',output.getvalue())

if __name__=='__main__':unittest.main()
