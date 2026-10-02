"""No hardware: regression checks for acquisition/finalization boundaries."""
import unittest
from unittest.mock import patch, Mock
import server


class RecorderLifecycleTests(unittest.TestCase):
    def recorder(self):
        r = server.Recorder()
        r.active = True
        r.state = 'recording'
        r.master = 123
        return r

    @patch.object(server.os, 'write')
    def test_repeated_stop_interrupts_only_once(self, write):
        r = self.recorder()
        r.stop()
        r.stop()
        r._interrupt_acquisition()
        write.assert_called_once_with(123, b'\x03')
        self.assertEqual(r.state, 'stopping')

    @patch.object(server.os, 'write')
    def test_both_finalizing_markers_prohibit_interrupt(self, write):
        for line in ['FINALIZING: saving joint data', '连续静止 2 秒，保存并停用。']:
            r = self.recorder()
            self.assertTrue(r.observe_finalization(line))
            r.stop()
            r._interrupt_acquisition()
            self.assertEqual(r.state, 'finalizing')
            self.assertTrue(r.active)
        write.assert_not_called()

    @patch.object(server.os, 'write')
    def test_error_after_finalizing_does_not_allow_cleanup_interrupt(self, write):
        r = self.recorder()
        r.observe_finalization('FINALIZING: video encoding')
        r.update('error', 'timeout')
        r._interrupt_acquisition()
        r.stop()
        write.assert_not_called()
        for action in [r.start, r.connect, r.sync]:
            with self.assertRaises(RuntimeError): action()

    @patch.object(server.os, 'write')
    def test_stop_before_master_exists_is_deferred_once(self, write):
        r = self.recorder()
        r.state = 'starting'
        r.master = None
        r.stop()
        write.assert_not_called()
        r.master = 123
        r._interrupt_acquisition()
        r._interrupt_acquisition()
        write.assert_called_once_with(123, b'\x03')

    def test_done_marker_before_child_exit_does_not_send_interrupt(self):
        r = self.recorder()
        r.state = 'starting'
        r.master = None
        child = Mock()
        child.poll.return_value = None
        child.wait.side_effect = lambda **kw: setattr(child.poll, 'return_value', 0)
        output = b'FINALIZING: encoding\n__A1Z_TEACH_FINISHED__ 0\n'
        with patch.object(server.pty, 'openpty', return_value=(123,124)), \
             patch.object(server.subprocess, 'Popen', return_value=child), \
             patch.object(server.os, 'close'), \
             patch.object(server.os, 'read', return_value=output), \
             patch.object(server.os, 'write') as write, \
             patch.object(server.select, 'select', return_value=([123],[],[])):
            r._run()
        write.assert_not_called()
        child.terminate.assert_not_called()
        self.assertFalse(r.active)

    @patch.object(server.os, 'write')
    def test_late_stop_after_save_is_noop(self, write):
        for state in ['syncing', 'saved']:
            r = self.recorder()
            r.state = state
            r.stop()
        write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
