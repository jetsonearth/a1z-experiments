"""Offline protocol checks; never opens hardware or enables motors."""
import unittest
from unittest.mock import patch, Mock
import server

class RecorderLifecycleTests(unittest.TestCase):
    def recorder(self):
        r=server.Recorder();r.active=True;r.master=123;return r

    @patch.object(server.os,'write')
    def test_ready_phase_requires_fresh_button(self,write):
        r=self.recorder();r.observe_line('TASK_PHASE ready')
        write.assert_not_called()
        r.ready();write.assert_called_once_with(123,b'READY\r')
        with self.assertRaises(RuntimeError):r.ready()

    @patch.object(server.os,'write')
    def test_task_end_never_interrupts_or_disables(self,write):
        r=self.recorder();r.observe_line('TASK_PHASE positioning');r.begin()
        r.observe_line('TASK_PHASE recording');r.stop();r.stop()
        r.observe_line('TASK_PHASE support');r.stop()
        self.assertEqual([c.args[1] for c in write.call_args_list],[b'BEGIN\r',b'END\r'])
        self.assertEqual(r.state,'support');self.assertTrue(r.active)
        r.disable();self.assertEqual(write.call_args.args,(123,b'DISABLE\r'))
        self.assertEqual(r.state,'finalizing')
        with self.assertRaises(RuntimeError):r.disable()

    @patch.object(server.os,'write')
    def test_out_of_phase_commands_do_not_write(self,write):
        r=self.recorder();r.state='starting'
        for action in [r.ready,r.begin,r.stop,r.disable]:
            with self.assertRaises(RuntimeError):action()
        write.assert_not_called()

    def test_done_marker_before_child_exit_does_not_interrupt(self):
        r=self.recorder();r.master=None;r.state='starting'
        child=Mock();child.poll.return_value=None
        child.wait.side_effect=lambda **kw:setattr(child.poll,'return_value',0)
        output=b'TASK_PHASE ready\nTASK_PHASE finalizing\n__A1Z_TEACH_FINISHED__ 0\n'
        with patch.object(server.pty,'openpty',return_value=(123,124)),patch.object(server.subprocess,'Popen',return_value=child) as spawn,patch.object(server.os,'close'),patch.object(server.os,'read',return_value=output),patch.object(server.os,'write') as write,patch.object(server.select,'select',return_value=([123],[],[])):
            r._run()
        write.assert_not_called();child.terminate.assert_not_called()
        self.assertEqual(spawn.call_args.kwargs['env']['A1Z_MODE'],'task')
        self.assertFalse(r.active)

    def test_ui_read_failure_preserves_live_process_without_interrupt(self):
        r=self.recorder();r.master=None;r.state='starting'
        child=Mock();child.poll.return_value=None
        with patch.object(server.pty,'openpty',return_value=(123,124)),patch.object(server.subprocess,'Popen',return_value=child),patch.object(server.os,'close') as close,patch.object(server.os,'read',side_effect=OSError('read failed')),patch.object(server.os,'write') as write,patch.object(server.select,'select',return_value=([123],[],[])):
            r._run()
        write.assert_not_called();child.terminate.assert_not_called()
        self.assertTrue(r.active);self.assertEqual(r.master,123)
        self.assertNotIn((123,),[c.args for c in close.call_args_list])

if __name__=='__main__':unittest.main()

class FailureMessageTests(unittest.TestCase):
    def test_camera_startup_is_not_robot_protection(self):
        message=server.failure_message({'stage':'camera_startup','controller_started':False,'fault':'invalid camera argument'})
        self.assertIn('相机启动失败',message)
        self.assertIn('尚未启用',message)
        self.assertNotIn('保护',message)

    def test_unknown_failure_does_not_claim_motors_were_never_enabled(self):
        message=server.failure_message({'status':'failed','fault':'unclassified'})
        self.assertNotIn('尚未启用',message)
        self.assertNotIn('保护',message)

class ContinuousSessionTests(unittest.TestCase):
    @patch.object(server.os, 'write')
    def test_multiple_episodes_stay_active_until_explicit_disable(self, write):
        r=server.Recorder();r.active=True;r.master=123
        r.observe_line('TASK_PHASE positioning');r.begin()
        r.observe_line('TASK_PHASE recording');r.stop()
        r.observe_line('TASK_EPISODE /remote/one-teaching.json 50')
        r.observe_line('TASK_PHASE support');r.begin()
        r.observe_line('TASK_PHASE recording');r.gripper(True)
        r.observe_line('TASK_GRIPPER open');r.gripper(False)
        r.observe_line('TASK_GRIPPER closed');r.stop()
        r.observe_line('TASK_EPISODE /remote/two-teaching.json 70')
        r.observe_line('TASK_PHASE support')
        self.assertTrue(r.active);self.assertEqual(len(r.session_episodes),2)
        self.assertEqual([c.args[1] for c in write.call_args_list], [b'BEGIN\r',b'END\r',b'BEGIN\r',b'OPEN\r',b'CLOSE\r',b'END\r'])
        self.assertEqual(r.gripper_state,'closed')
        r.disable();self.assertEqual(write.call_args.args[1],b'DISABLE\r')

    def test_session_episode_records_usb_insertion_per_take(self):
        r=server.Recorder()
        r.observe_line("USB_WATCH {'targets': None, 'mode': 'new_since_startup'}");self.assertIs(r.usb_connected,False)
        r.observe_line('TASK_PHASE recording');r.observe_line('USB_CONNECTED');r.observe_line('USB_REMOVED')
        r.observe_line('TASK_EPISODE /remote/a-teaching.json 120')  # inserted mid-take, pulled before end
        r.observe_line('TASK_PHASE support');r.observe_line('TASK_PHASE recording')
        r.observe_line('TASK_EPISODE /remote/b-teaching.json 90')  # never inserted
        r.observe_line('USB_CONNECTED');r.observe_line('TASK_PHASE recording')
        r.observe_line('TASK_EPISODE /remote/c-teaching.json 80')  # already connected at the end
        self.assertEqual([e.get('usb_inserted') for e in r.session_episodes],[True,False,True])
        r2=server.Recorder();r2.observe_line('TASK_PHASE recording');r2.observe_line('TASK_EPISODE /remote/d-teaching.json 10')
        self.assertNotIn('usb_inserted',r2.session_episodes[0])  # no watcher: unknown, not False

    def test_temperature_qc_and_discard_lines_are_parsed_even_when_merged(self):
        r=server.Recorder()
        r.observe_line('TASK_PHASE positioning');self.assertIsNotNone(r.snapshot()['enabled_s'])
        r.observe_line('TASK_TEMP {"mos":[40.0,74.5,41,42,43,44],"rotor":[35,60,36,37,38,39]}Motor temperature warning: joint2 MOS=74.5°C')
        snap=r.snapshot()['motor_temps'];self.assertEqual(snap['mos'][1],74.5);self.assertEqual(snap['rotor'][1],60);self.assertLess(snap['age_s'],1)
        self.assertEqual(r.motor_temp['joint'],2)  # legacy warning parsing still works on the merged line
        r.observe_line('TASK_GRIPPER openMotor temperature warning: joint3 MOS=71.0°C')
        self.assertEqual(r.gripper_state,'open');self.assertEqual(r.motor_temp['joint'],3)
        r.observe_line('TASK_PHASE recording');r.observe_line('TASK_EPISODE /r/a-teaching.json 40')
        r.observe_line('TASK_QC {"path": "/r/a-teaching.json", "frames": 40, "duration_s": 0.8, "usb_inserted": true, "gripper_changes": 2, "j4_max": 1.2, "j4_limit": 1.7, "flags": ["过短", "录制中开合夹爪"]}TASK_PHASE support')
        e=r.session_episodes[0];self.assertEqual(e['qc']['flags'],['过短','录制中开合夹爪']);self.assertIs(e['usb_inserted'],True);self.assertNotIn('path',e['qc'])
        self.assertEqual(r.state,'support')  # phase marker glued after the JSON still applies
        r.observe_line('TASK_QC {"path": "/r/b-teaching.json", "flags": []}')  # QC before its episode: attached later
        r.observe_line('TASK_PHASE recording');r.observe_line('TASK_EPISODE /r/b-teaching.json 300');self.assertEqual(r.session_episodes[1]['qc']['flags'],[])
        r.discard_pending=True;r.observe_line('TASK_DISCARDED /r/b-teaching.jsonMotor temperature warning')
        self.assertTrue(r.session_episodes[1]['excluded']);self.assertNotIn('excluded',r.session_episodes[0]);self.assertFalse(r.discard_pending)
        r.observe_line('TASK_QC {broken');r.observe_line('TASK_TEMP not-json')  # malformed payloads are ignored

    @patch.object(server.os, 'write')
    def test_discard_only_in_support_with_an_unexcluded_episode(self, write):
        r=server.Recorder();r.master=123
        for state in ['positioning','recording','ready','finalizing']:
            r.state=state;r.session_episodes=[{'path':'/r/a','frames':10}]
            with self.assertRaises(RuntimeError):r.discard()
        r.state='support';r.session_episodes=[]
        with self.assertRaises(RuntimeError):r.discard()
        r.session_episodes=[{'path':'/r/a','frames':10,'excluded':True}]
        with self.assertRaises(RuntimeError):r.discard()
        write.assert_not_called()
        r.session_episodes=[{'path':'/r/a','frames':10}];r.discard()
        self.assertEqual(write.call_args.args[1],b'DISCARD\r');self.assertTrue(r.discard_pending);self.assertEqual(r.state,'support')
        with self.assertRaises(RuntimeError):r.discard()  # no double send while awaiting confirmation
        r.observe_line('TASK_IGNORED DISCARD positioning');self.assertFalse(r.discard_pending)

    @patch.object(server.os, 'write')
    def test_acknowledgments_are_remote_only_and_gripper_is_phase_gated(self, write):
        r=server.Recorder();r.master=123;r.state='positioning'
        r.gripper(True);self.assertEqual(r.ack_events,[])
        r.observe_line('TASK_GRIPPER open');self.assertEqual(r.ack_events[-1]['text'],'夹爪已张开')
        r.begin();self.assertEqual(len(r.ack_events),1)
        r.observe_line('TASK_PHASE recording');self.assertEqual(r.ack_events[-1]['text'],'开始录制')
        for state in ['ready','starting','stopping','finalizing','syncing','saved','error']:
            r.state=state
            with self.assertRaises(RuntimeError):r.gripper(False)

    def test_all_saved_episodes_sync_once_after_finished_marker(self):
        r=server.Recorder();child=Mock();child.poll.return_value=None
        child.wait.side_effect=lambda **kw:setattr(child.poll,'return_value',0)
        output=b'SAVED /one\nSAVED /one\nSAVED /two\n__A1Z_TEACH_FINISHED__ 0\n'
        with patch.object(server.pty,'openpty',return_value=(123,124)),patch.object(server.subprocess,'Popen',return_value=child),patch.object(server.os,'close'),patch.object(server.os,'read',return_value=output),patch.object(server.select,'select',return_value=([123],[],[])),patch.object(server,'settle_shell'),patch.object(server,'remote_episode',side_effect=[('one',50,'motion_complete'),('two',70,'motion_complete')]) as sync,patch.object(server,'local_media',return_value=None),patch.object(server.Path,'read_text',return_value='{"metadata":{}}'):
            r._run()
        self.assertEqual([call.args[0] for call in sync.call_args_list],['/one','/two'])
        self.assertEqual(r.state,'saved');self.assertIn('2 条，共 120 帧',r.message)

class SharedMediaCacheTests(unittest.TestCase):
    def test_second_episode_reuses_verified_shared_media(self):
        import hashlib, json, tempfile
        from pathlib import Path
        payload=b'shared session video';digest=hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache=root/'.media-cache';cache.mkdir()
            (cache/digest).write_bytes(payload)
            with patch.object(server,'EPISODES',root),patch.object(server,'serial_rpc',return_value=json.dumps({'bytes':len(payload),'sha256':digest})) as rpc:
                server.fetch_remote_file('/shared/video.mp4',root/'episode-two.mp4')
            self.assertEqual((root/'episode-two.mp4').read_bytes(),payload)
            self.assertEqual(rpc.call_count,1)  # hash query only, no video blocks

class RevisedCaptureTests(unittest.TestCase):
    @patch.object(server.os, 'write')
    def test_toggle_uses_backend_target_and_waits_for_remote_ack(self, write):
        r=server.Recorder();r.master=123;r.state='positioning'
        with self.assertRaises(RuntimeError):r.toggle_gripper()
        r.observe_line('TASK_GRIPPER open');r.toggle_gripper()
        self.assertEqual(write.call_args.args[1],b'CLOSE\r')
        with self.assertRaises(RuntimeError):r.toggle_gripper()
        self.assertEqual(write.call_count,1)
        r.observe_line('TASK_GRIPPER closed');r.toggle_gripper()
        self.assertEqual(write.call_args.args[1],b'OPEN\r')

    def test_json_is_announced_before_media_sync_finishes(self):
        import json,hashlib,zlib,base64,tempfile
        from pathlib import Path
        raw=json.dumps({'metadata':{},'samples':[{'q':[0]*6}]}).encode()
        output='SHA '+hashlib.sha256(raw).hexdigest()+'\nDATA '+base64.b64encode(zlib.compress(raw)).decode()
        r=server.Recorder();r.state='syncing'
        def media(name,metadata,progress):
            self.assertEqual(r.last_episode,name)
            self.assertEqual(r.frames,1)
            raise server.MediaTransferPaused('paused')
        with tempfile.TemporaryDirectory() as temp,patch.object(server,'EPISODES',Path(temp)),patch.object(server,'serial_rpc',return_value=output),patch.object(server,'sync_media',side_effect=media):
            with self.assertRaises(server.MediaTransferPaused):server.remote_episode(server.REMOTE_DIR+'demo-teaching.json',on_json_saved=r.json_saved)
            self.assertTrue((Path(temp)/'demo-teaching.json').exists())

class ResumableTransferTests(unittest.TestCase):
    def tearDown(self):server.MEDIA_PAUSE.clear()

    def test_pause_leaves_prefix_and_resume_downloads_only_remainder(self):
        import hashlib,json,base64,tempfile
        from pathlib import Path
        payload=b'a'*65536+b'b'*400;checksum=hashlib.sha256(payload).hexdigest()
        info=json.dumps({'bytes':len(payload),'sha256':checksum})
        def rpc(source,timeout):
            if 'p.stat()' in source:return info
            if 'f.seek(0)' in source:
                server.MEDIA_PAUSE.set();return base64.b64encode(payload[:65536]).decode()
            self.assertIn('f.seek(65536)',source);return base64.b64encode(payload[65536:]).decode()
        with tempfile.TemporaryDirectory() as temp,patch.object(server,'EPISODES',Path(temp)),patch.object(server,'serial_rpc',side_effect=rpc) as mock:
            dest=Path(temp)/'wrist.mp4'
            with self.assertRaises(server.MediaTransferPaused):server.fetch_remote_file('/video.mp4',dest)
            self.assertFalse(dest.exists());self.assertEqual(dest.with_suffix('.mp4.incoming').stat().st_size,65536)
            server.MEDIA_PAUSE.clear();server.fetch_remote_file('/video.mp4',dest)
            self.assertEqual(dest.read_bytes(),payload)
            self.assertEqual(mock.call_count,4)

    def test_legacy_prefix_without_sidecar_is_sha_verified(self):
        import hashlib,json,base64,tempfile
        from pathlib import Path
        payload=b'prefix-tail';info=json.dumps({'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
        with tempfile.TemporaryDirectory() as temp,patch.object(server,'EPISODES',Path(temp)),patch.object(server,'serial_rpc',side_effect=[info,base64.b64encode(b'-tail').decode()]) as rpc:
            dest=Path(temp)/'wrist.mp4';dest.with_suffix('.mp4.incoming').write_bytes(b'prefix')
            server.fetch_remote_file('/video.mp4',dest)
            self.assertIn('f.seek(6)',rpc.call_args.args[0]);self.assertEqual(dest.read_bytes(),payload)


class CollectionTests(unittest.TestCase):
    def test_takes_persist_across_rounds_and_restarts(self):
        import tempfile; from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp, patch.object(server,'HARDWARE_ENABLED',True), patch.object(server,'COLLECTION',Path(tmp)/'collection.json'):
            r=server.Recorder()
            r.observe_line('TASK_EPISODE /e/a.json 100')
            r.observe_line('TASK_QC {"path":"/e/a.json","usb_inserted":true,"flags":[]}')
            r.session_episodes=[]  # a new round clears the per-round list
            r.observe_line('TASK_EPISODE /e/b.json 80')
            r.observe_line('TASK_DISCARDED /e/b.json')
            again=server.Recorder()  # server restart reloads from disk
            got=again.snapshot()['collection']
            self.assertEqual([e['path'] for e in got],['/e/a.json','/e/b.json'])
            self.assertTrue(got[0]['usb_inserted']);self.assertTrue(got[1]['excluded'])
