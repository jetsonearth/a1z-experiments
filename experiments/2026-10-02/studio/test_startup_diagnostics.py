"""Startup regression checks; no serial access or robot commands."""
import importlib.util
from pathlib import Path
import unittest
import server

spec = importlib.util.spec_from_file_location('mac_console', Path(__file__).resolve().parents[1] / 'mac_console.py')
console = importlib.util.module_from_spec(spec)
spec.loader.exec_module(console)

class StartupDiagnosticsTests(unittest.TestCase):
    def test_bash_and_zsh_idle_prompts(self):
        for prompt in ('dimos@jetson01 ~ $ ', 'dimos@jetson01 ~/a1z % ', '\r\ndimos@jetson01 ~ $ '):
            self.assertTrue(console.idle_shell_prompt(prompt), repr(prompt))

    def test_non_idle_output_is_rejected(self):
        for output in ('jetson01 login: ', 'Password: ', '输入 READY: ', 'dimos@jetson01 ~ $ python3 collect_task.py', 'dimos@jetson01 ~ $ \nTASK_PHASE ready\n'):
            self.assertFalse(console.idle_shell_prompt(output), repr(output))

    def test_prompt_failure_explains_no_command_sent(self):
        message = server.console_failure('Traceback\nRuntimeError: Jetson idle shell not confirmed; no recording command sent')
        self.assertIn('未发送采集启动命令', message)

    def test_specific_exception_survives(self):
        message = server.console_failure('\x1b[31mOSError: Serial disconnected\x1b[0m\r\n')
        self.assertIn('OSError: Serial disconnected', message)

    def test_missing_marker_stays_explicit(self):
        self.assertIn('未收到进程完成标记', server.console_failure('unrecognized output'))

if __name__ == '__main__':
    unittest.main()

class SharedShellHandshakeTests(unittest.TestCase):
    def handshake(self, chunks):
        import serial_session
        from unittest.mock import patch
        from itertools import count
        writes = []
        def read(*args):
            return next(chunks, b'unknown foreground prompt')
        with patch.object(serial_session.os, 'write', side_effect=lambda fd,data:writes.append(data)), patch.object(serial_session.os, 'read', side_effect=read), patch.object(serial_session.select, 'select', return_value=([123],[],[])), patch.object(serial_session.time, 'sleep'), patch.object(serial_session.time, 'monotonic', side_effect=count(0,.1)):
            try:
                serial_session.ensure_shell(123)
                error = None
            except RuntimeError as e:
                error = str(e)
        return writes,error

    def test_normalized_shell_is_reused_without_commands(self):
        writes,error=self.handshake(iter([b'dimos@jetson01 A1Z $ ']))
        self.assertIsNone(error);self.assertEqual(writes,[b'\x0c'])

    def test_zsh_bootstrap_is_paced_and_does_not_exec_login_shell(self):
        writes,error=self.handshake(iter([b'dimos@jetson01 ~ % ',b'\r\nbash-5.1$ ',b'\r\ndimos@jetson01 A1Z $ ']))
        self.assertIsNone(error)
        self.assertTrue(all(len(chunk)==1 for chunk in writes))
        self.assertEqual(b''.join(writes),b"\x0cbash --noprofile --norc\rPS1='dimos@jetson01 A1Z $ '\r")

    def test_unknown_foreground_never_receives_enter_or_interrupt(self):
        writes,error=self.handshake(iter([b'TASK_PHASE ready']))
        self.assertIsNotNone(error);self.assertEqual(writes,[b'\x0c'])
