"""Offline regressions using installed LeRobot queue/transport methods, no hardware."""
import pickle
from queue import Queue
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import TimedAction, TimedObservation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.async_inference.robot_client import RobotClient

from async_policy import FreshObservationServerMixin, TimedClientMixin, Timeline


def obs(step, must_go=False):
    # Identical proprioception, potentially different images: it must be processed.
    return TimedObservation(timestamp=100+step/30, timestep=step,
                            observation={'joint': 0.0}, must_go=must_go)


def actions(start, value=1.0, count=30):
    return [TimedAction(timestamp=100+i/30, timestep=i,
                        action=torch.full((7,), value)) for i in range(start, start+count)]


class Server(FreshObservationServerMixin, PolicyServer):
    pass


class OfflineRobotClient(RobotClient):
    """Use upstream aggregation and consumption without constructing a robot or socket."""
    def __init__(self):
        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = 30
        self.action_queue_size = []
        self.must_go = threading.Event()
        self.config = types.SimpleNamespace(aggregate_fn=lambda old, new: .3*old+.7*new)
        self.robot = types.SimpleNamespace(action_features=[f'j{i}' for i in range(7)],
                                          send_action=lambda action: action)


class Client(TimedClientMixin, OfflineRobotClient):
    pass


class AsyncTests(unittest.TestCase):
    def setUp(self):
        self.server = Server(PolicyServerConfig())
        self.client = Client()
        self.client.reset_stream()

    def test_identical_state_can_trigger_early_inference_and_overlap(self):
        c = self.client
        start = c.latest_action
        c._aggregate_action_queues(actions(start), c.config.aggregate_fn)
        for _ in range(9):
            c.control_loop_action()
        self.assertLessEqual(c.action_queue.qsize()/30, .7)
        self.assertTrue(self.server._enqueue_observation(obs(start, True)))
        self.server.observation_queue.get_nowait()
        self.assertTrue(self.server._enqueue_observation(obs(c.latest_action)))
        # Next chunk arrives while old actions still remain, with unchanged state.
        c._aggregate_action_queues(actions(c.latest_action, 2.0), c.config.aggregate_fn)
        overlap = next(r for r in reversed(c.async_trace.rows) if r['event']=='chunk_received')
        self.assertGreater(overlap['overlap'], 0)
        result = c.control_loop_action()
        self.assertAlmostEqual(result['j0'], 1.7, places=5)

    def test_duplicate_forced_and_out_of_order_observations_are_rejected(self):
        s = self.server
        self.assertTrue(s._enqueue_observation(obs(10)))
        self.assertFalse(s._enqueue_observation(obs(10, True)))
        self.assertFalse(s._enqueue_observation(obs(9, True)))
        self.assertEqual(s.observation_queue.qsize(), 1)

    def test_slow_inference_retains_only_latest_pending_observation(self):
        s = self.server
        s._enqueue_observation(obs(1))
        entered, release = threading.Event(), threading.Event()
        result = []
        def predict(o):
            entered.set()
            self.assertTrue(release.wait(2))
            return actions(o.get_timestep())
        s._predict_action_chunk = predict
        ctx = types.SimpleNamespace(peer=lambda:'offline')
        worker = threading.Thread(target=lambda:result.append(s.GetActions(None, ctx)))
        worker.start()
        self.assertTrue(entered.wait(2))
        for step in range(2, 20):
            self.assertTrue(s._enqueue_observation(obs(step)))
        self.assertEqual(s.observation_queue.qsize(), 1)
        self.assertEqual(s.observation_queue.queue[0].get_timestep(), 19)
        self.assertFalse(s._enqueue_observation(obs(1, True)))
        release.set(); worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(pickle.loads(result[0].data)[0].get_timestep(), 1)

    def test_new_connection_resets_deduplication(self):
        s = self.server
        s._enqueue_observation(obs(100))
        s._reset_server()
        self.assertEqual(s.observation_queue.qsize(), 0)
        self.assertTrue(s._enqueue_observation(obs(1)))

    def test_stop_and_handback_reject_old_response_without_erasing_new_queue(self):
        c = self.client
        old = actions(c.latest_action)
        c.pause_stream('takeover')
        c._aggregate_action_queues(old, c.config.aggregate_fn)
        self.assertTrue(c.action_queue.empty())
        c.reset_stream()
        c._aggregate_action_queues(actions(c.latest_action, 2.0), c.config.aggregate_fn)
        depth = c.action_queue.qsize()
        c._aggregate_action_queues(old, c.config.aggregate_fn)
        self.assertEqual(c.action_queue.qsize(), depth)
        self.assertEqual(c.control_loop_action()['j0'], 2.0)

    def test_empty_intervals_distinguish_startup_from_running_starvation(self):
        c = self.client
        c.observe_queue()
        c._aggregate_action_queues(actions(c.latest_action, count=2), c.config.aggregate_fn)
        c.observe_queue()
        c.control_loop_action()
        c.observe_queue()
        c.pause_stream('stop')
        starts = [r for r in c.async_trace.rows if r['event']=='queue_empty_start']
        self.assertEqual([r['startup'] for r in starts], [True, False])
        ends = [r for r in c.async_trace.rows if r['event']=='queue_empty_end']
        self.assertEqual([r['reason'] for r in ends], ['refilled', 'stop'])

    def test_concurrent_arrival_cannot_resurrect_queue_after_pause(self):
        c = self.client
        # Hold the stream lock so a receiver is pending while the main thread stops.
        with c._stream_lock:
            worker = threading.Thread(target=c._aggregate_action_queues,
                                      args=(actions(c.latest_action), c.config.aggregate_fn))
            worker.start()
            c.pause_stream('stop')
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(c.action_queue.empty())

    def test_trace_flush_appends_and_is_valid_json(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'trace.jsonl'
            trace = Timeline(p)
            trace.record('start', step=1); trace.flush()
            trace.record('end', duration_s=.1); trace.flush()
            self.assertEqual([json.loads(l)['event'] for l in p.read_text().splitlines()],
                             ['start', 'end'])
            self.assertFalse(trace.rows)

    def test_prediction_trace_connects_observation_and_inference(self):
        with patch.object(PolicyServer, '_predict_action_chunk', return_value=actions(10)):
            result = self.server._predict_action_chunk(obs(10))
        self.assertEqual(len(result), 30)
        rows = self.server.async_trace.rows
        self.assertEqual([r['event'] for r in rows], ['inference_start', 'inference_end'])
        self.assertEqual(rows[0]['observation_wall'], obs(10).get_timestamp())
        self.assertEqual(rows[1]['step'], 10)

    def test_observation_send_trace_includes_frame_age_and_queue_depth(self):
        c = self.client
        c.robot.frame_age = lambda: .02
        with patch.object(RobotClient, 'send_observation', return_value=True):
            self.assertTrue(c.send_observation(obs(c.latest_action)))
        rows = c.async_trace.rows[-2:]
        self.assertEqual([r['event'] for r in rows], ['observation_send', 'observation_sent'])
        self.assertEqual(rows[0]['frame_age_s'], .02)
        self.assertEqual(rows[0]['queue'], 0)


if __name__ == '__main__':
    unittest.main()
