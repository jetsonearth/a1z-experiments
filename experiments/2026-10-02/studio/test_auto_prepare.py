"""No hardware access: startup sequencing and CAN repair failure boundaries."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import server
import startup_check

class PreparationTests(unittest.TestCase):
    @patch.object(server.glob, 'glob', return_value=['/dev/mock'])
    @patch.object(server, 'serial_rpc')
    @patch.object(server.os, 'write')
    def test_preparation_never_confirms_motor_readiness(self, write, rpc, ports):
        r = server.Recorder(); r.active = True
        rpc.return_value = json.dumps({'ok': True, 'host': 'jetson01', 'checks': []})
        with patch.object(r, '_run', side_effect=lambda:r.observe_line('TASK_PHASE ready')) as run:
            r._prepare()
        run.assert_called_once(); write.assert_not_called()
        self.assertEqual(r.state, 'ready'); self.assertTrue(r.active)

    @patch.object(server.glob, 'glob', return_value=['/dev/mock'])
    @patch.object(server, 'serial_rpc', return_value='{"ok":false,"error":"相机未连接","checks":[]}')
    def test_failed_preflight_does_not_launch_recorder(self, rpc, ports):
        r = server.Recorder(); r.active = True
        with patch.object(r, '_run') as run: r._prepare()
        run.assert_not_called(); self.assertEqual(r.state,'error'); self.assertFalse(r.active)
        self.assertEqual(r.message,'相机未连接')

    @patch.object(server.threading, 'Thread')
    def test_duplicate_start_is_idempotent(self, thread):
        r = server.Recorder(); r.start(); r.start(); r.connect()
        self.assertEqual(thread.call_count,1)
        self.assertEqual(r.state,'checking')

    @patch.object(server.os, 'write')
    def test_cancel_only_before_motor_enable(self, write):
        r=server.Recorder();r.master=123;r.state='positioning'
        with self.assertRaises(RuntimeError): r.cancel_prepare()
        write.assert_not_called()
        r.state='ready';r.cancel_prepare()
        write.assert_called_once_with(123,b'CANCEL\r')

class HardwareChecksTests(unittest.TestCase):
    def run_case(self, state='up', camera=True):
        calls=[]
        def run(args, **kwargs):
            calls.append(args)
            if args[0]=='lsusb':out='8086:0b5b D405' if camera else ''
            elif args[0]=='sudo':out=''
            else:
                repaired=any(c[0]=='sudo' for c in calls)
                out=json.dumps([{'flags':['UP'] if state=='up' or repaired else [],'linkinfo':{'info_data':{'state':'ERROR-ACTIVE','bittiming':{'bitrate':1000000}}}}])
            return Mock(returncode=0,stdout=out,stderr='')
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(startup_check.Path,'is_file',return_value=True),patch.object(startup_check.shutil,'which',return_value='/usr/bin/ffmpeg'),patch.object(startup_check.subprocess,'run',side_effect=run):
                result=startup_check.run_checks(Path(directory))
        return result,calls

    def test_down_can_restored_with_vendor_bitrate(self):
        result,calls=self.run_case('down')
        self.assertTrue(result['ok'],result)
        self.assertEqual([c for c in calls if c[0]=='sudo'],[['sudo','-n','ip','link','set','can1','up','type','can','bitrate','1000000']])

    def test_healthy_can_not_reset(self):
        result,calls=self.run_case()
        self.assertTrue(result['ok']);self.assertFalse(any(c[0]=='sudo' for c in calls))

    def test_missing_camera_blocks_before_can_mutation(self):
        result,calls=self.run_case('down',False)
        self.assertFalse(result['ok']);self.assertFalse(any(c[0]=='sudo' for c in calls))

if __name__=='__main__':unittest.main()
