"""Fault-injection checks without importing the SDK or touching hardware."""
import unittest
from types import SimpleNamespace
from teach import FeedbackBus, MotionGate


def frame(mid, first, rx=True, error=False):
    return SimpleNamespace(arbitration_id=mid, data=bytes([first]) + b'\x80' * 7,
                           is_rx=rx, is_error_frame=error, is_extended_id=False)


class FakeBus:
    def __init__(self, frames=()):
        self.frames = iter(frames)
        self.timeout = None
    def recv(self, timeout=0):
        return next(self.frames, None)
    def send(self, msg, timeout=None):
        self.timeout = timeout


class GuardTests(unittest.TestCase):
    def test_missing_joint_is_not_hidden_by_other_traffic(self):
        guard = FeedbackBus(FakeBus([frame(i, 0x20 if i <= 3 else 0x10 | i)
                                    for i in range(1, 6)]), clock=lambda: 1.0)
        while guard.recv() is not None:
            pass
        with self.assertRaisesRegex(RuntimeError, 'Missing/stale'):
            guard.require_fresh()

    def test_echo_wrong_report_and_wrong_embedded_id_do_not_refresh(self):
        guard = FeedbackBus(FakeBus([frame(1, 0x20, rx=False), frame(1, 0), frame(4, 0x15)]))
        self.assertIsNone(guard.recv())
        self.assertEqual(guard.count, [0] * 6)

    def test_individual_stale_feedback_halts(self):
        now = [1.0]
        guard = FeedbackBus(FakeBus([frame(i, 0x20 if i <= 3 else 0x10 | i)
                                    for i in range(1, 7)]), clock=lambda: now[0])
        while guard.recv() is not None:
            pass
        guard.require_fresh()
        now[0] = 1.101
        with self.assertRaises(RuntimeError):
            guard.require_fresh()

    def test_successive_commands_are_paced_without_delaying_spaced_commands(self):
        now = [1.0]
        sleeps = []
        def sleep(delay):
            sleeps.append(delay)
            now[0] += delay
        guard = FeedbackBus(FakeBus(), clock=lambda: now[0], sleeper=sleep)
        guard.send(frame(1, 0x20))
        guard.send(frame(2, 0x20))
        self.assertAlmostEqual(sleeps[0], 0.0003)
        now[0] += 0.01
        guard.send(frame(3, 0x20))
        self.assertEqual(len(sleeps), 1)

    def test_can_error_and_send_deadline(self):
        raw = FakeBus([frame(1, 0x20, error=True)])
        guard = FeedbackBus(raw)
        with self.assertRaisesRegex(RuntimeError, 'CAN error'):
            guard.recv()
        guard.send(frame(1, 0x20), timeout=None)
        self.assertEqual(raw.timeout, 0.02)


class MotionGateTests(unittest.TestCase):
    def test_stationary_wait_does_not_consume_recording_window(self):
        gate = MotionGate([0] * 6, now=100, duration=20, wait_limit=120)
        for now in range(101, 160):
            gate.observe([0] * 6, now)
        self.assertIsNone(gate.motion_started)
        self.assertEqual(gate.deadline, 220)
        for i in range(3):
            gate.observe([0.02] + [0] * 5, 160 + 0.004 * i)
        self.assertEqual(gate.motion_started, 160)
        self.assertEqual(gate.deadline, 180)

    def test_noise_and_single_spike_do_not_trigger(self):
        gate = MotionGate([0] * 6, now=0)
        gate.observe([0.001] * 6, 1)
        gate.observe([0.02] * 6, 2)
        gate.observe([0] * 6, 3)
        self.assertIsNone(gate.motion_started)
        self.assertEqual(gate.hits, 0)

    def test_stillness_ends_and_resumed_motion_resets_idle_timer(self):
        gate = MotionGate([0] * 6, now=0)
        for t in [5, 5.004, 5.008]:
            gate.observe([0.02] * 6, t)
        self.assertFalse(gate.finished(6.9))
        gate.observe([0.03] * 6, 6.9)
        self.assertFalse(gate.finished(8.8))
        self.assertTrue(gate.finished(8.91))

    def test_recording_deadline_does_not_extend_when_still(self):
        gate = MotionGate([0] * 6, now=0, duration=20)
        for t in [10, 10.004, 10.008]:
            gate.observe([0.03] * 6, t)
        gate.observe([0] * 6, 20)
        self.assertEqual(gate.deadline, 30)


if __name__ == '__main__':
    unittest.main()
