import struct
import unittest
from unittest.mock import patch, Mock
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from spacemouse_teleop import MouseState, Clutch, differential_target
import spacemouse_teleop as teleop


def packet(kind, *axes):
    return bytes([kind])+struct.pack('<hhh', *axes)


class MouseTests(unittest.TestCase):
    def test_axes_and_buttons(self):
        m = MouseState()
        m.feed(packet(1, 350, -350, 175), 1.)
        m.feed(packet(2, -175, 70, -70), 1.)
        m.feed(b'\x03\x03\x00', 1.)
        self.assertEqual(m.axes(1.01, True), [1., 1., -.5, -.5, -.2, .2])
        self.assertEqual(m.axes(1.01)[3:], [0., 0., 0.])
        self.assertEqual(m.buttons, 3)

    def test_stale_axes_stop_independently(self):
        m = MouseState()
        m.feed(packet(1, 350, 0, 0), 1.)
        m.feed(packet(2, 0, 350, 0), 1.2)
        self.assertEqual(m.axes(1.21, True), [0., 0., 0., 0., -1., 0.])
        self.assertTrue(m.stale_motion(1.21, False))

    def test_clutch_requires_release_after_start_and_stale_motion(self):
        m, c = MouseState(), Clutch()
        m.feed(b'\x03\x01', 1.)
        self.assertFalse(c.update(m, 1., False))
        m.feed(b'\x03\x00', 1.01)
        self.assertFalse(c.update(m, 1.01, False))
        m.feed(b'\x03\x01', 1.02)
        m.feed(packet(1, 40, 0, 0), 1.02)
        self.assertTrue(c.update(m, 1.02, False))
        self.assertFalse(c.update(m, 1.3, False))
        m.feed(packet(1, 40, 0, 0), 1.31)
        self.assertFalse(c.update(m, 1.31, False))

    def test_right_button_latches_pause(self):
        m, c = MouseState(), Clutch()
        m.feed(b'\x03\x00', 1.)
        c.update(m, 1., False)
        m.feed(b'\x03\x01', 1.)
        self.assertTrue(c.update(m, 1., False))
        m.feed(b'\x03\x03', 1.)
        self.assertFalse(c.update(m, 1., False))
        m.feed(b'\x03\x01', 1.)
        self.assertFalse(c.update(m, 1., False))
        self.assertTrue(c.paused)

    def test_bad_reports_fail(self):
        for bad in (b'\x01\x00', b'\x02\x00', b'\x03'):
            with self.assertRaises(ValueError):
                MouseState().feed(bad, 0.)
        with self.assertRaises(OSError):
            MouseState().feed(b'', 0.)


class KinematicsTests(unittest.TestCase):
    def setUp(self):
        pin = SimpleNamespace(LOCAL_WORLD_ALIGNED=0,
            computeFrameJacobian=lambda *args: np.eye(6), integrate=lambda model,q,dq: q+dq)
        self.kin = SimpleNamespace(pin=pin, k=SimpleNamespace(_model=None,_data=None,_frame_id=0))
        self.bounds = np.array([[-1.,1.]]*6)

    def step(self, q=None, axes=None, dt=.02, velocity=2.):
        return differential_target(self.kin, np.zeros(6) if q is None else q,
            [1,0,0,0,0,0] if axes is None else axes, dt, .025, .25, self.bounds, velocity)

    def test_differential_motion_and_no_input(self):
        self.assertAlmostEqual(self.step()[0], .0005/1.0001)
        np.testing.assert_equal(self.step(axes=[0]*6), np.zeros(6))

    def test_vendor_limits_and_delayed_tick_reject(self):
        for kwargs in ({'q': np.ones(6)}, {'velocity': .001}, {'dt': .2}, {'axes': [float('nan')]*6}):
            with self.assertRaises(ValueError):
                self.step(**kwargs)


class ReadyGateTest(unittest.TestCase):
    def test_cancel_before_ready_never_opens_can(self):
        factory, preflight = Mock(), Mock()
        fake_teach = SimpleNamespace(build_robot=factory, preflight=preflight, verify_revision=lambda:'test')
        fake_mouse = Mock(state=MouseState())
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(teleop, 'ROOT', Path(folder)), \
                patch.object(teleop, 'STATE_FILE', Path(folder)/'state.json'), \
                patch.object(teleop, 'Mouse', return_value=fake_mouse), \
                patch.dict('sys.modules', {'teach':fake_teach, 'insertion':SimpleNamespace(Kin=lambda:None)}), \
                patch('sys.argv', ['spacemouse_teleop.py','--execute']), \
                patch('builtins.input', return_value='no'):
            teleop.main()
        factory.assert_not_called()
        preflight.assert_not_called()
        fake_mouse.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
