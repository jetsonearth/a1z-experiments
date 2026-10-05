"""A1Z-local adapters for LeRobot async inference; no hardware imports.

Keep LeRobot's single pending observation and overlapping-action fusion, but do
not reject fresh observations just because raw joint angles changed little.
"""
import json
from pathlib import Path
from queue import Empty, Queue
import threading
import time


class Timeline:
    """Buffer client events off the real-time disk path; flush at trial boundaries."""
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.rows = []
        self.lock = threading.Lock()

    def record(self, event, **fields):
        with self.lock:
            self.rows.append(dict(t=time.monotonic(), wall=time.time(), event=event, **fields))

    def flush(self):
        if self.path is None:
            return
        with self.lock:
            if not self.rows:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a') as out:
                for row in self.rows:
                    out.write(json.dumps(row, allow_nan=False) + '\n')
            self.rows.clear()


class FreshObservationServerMixin:
    def __init__(self, *args, **kwargs):
        self._observation_lock = threading.Lock()
        self._last_accepted_step = -1
        self.async_trace = Timeline()
        super().__init__(*args, **kwargs)

    def _reset_server(self):
        with self._observation_lock:
            super()._reset_server()
            self._last_accepted_step = -1
            self.last_processed_obs = None

    def _enqueue_observation(self, obs):
        step = obs.get_timestep()
        with self._observation_lock:
            # Includes pending and in-flight requests, even if must_go is set.
            # The client advances its step range on each trial/handback.
            if step <= self._last_accepted_step:
                self.async_trace.record('observation_rejected', step=step,
                                        reason='duplicate_or_out_of_order')
                return False
            try:
                replaced = self.observation_queue.get_nowait().get_timestep()
            except Empty:
                replaced = None
            self.observation_queue.put_nowait(obs)
            self._last_accepted_step = step
            self.async_trace.record('observation_accepted', step=step,
                                    observation_wall=obs.get_timestamp(),
                                    must_go=obs.must_go, replaced_step=replaced)
            return True

    def _predict_action_chunk(self, obs):
        started = time.monotonic()
        self.async_trace.record('inference_start', step=obs.get_timestep(),
                                observation_wall=obs.get_timestamp())
        try:
            actions = super()._predict_action_chunk(obs)
            self.async_trace.record('inference_end', step=obs.get_timestep(),
                                    duration_s=time.monotonic()-started, actions=len(actions))
            return actions
        except Exception as exc:
            self.async_trace.record('inference_error', step=obs.get_timestep(),
                                    error=type(exc).__name__)
            raise
        finally:
            self.async_trace.flush()


class TimedClientMixin:
    """Trace queue timing, and serialize reset/stop against arriving chunks.

    The extra lock prevents LeRobot's queue snapshot/replace from racing with
    action consumption or reset. It is never held across network calls.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stream_lock = threading.RLock()
        self._stream_active = False
        self._stream = 0
        self._last_action_time = None
        self._empty_since = None
        self.async_trace = Timeline()

    def _record(self, event, **fields):
        self.async_trace.record(event, stream=self._stream, **fields)

    def reset_stream(self):
        with self._stream_lock:
            with self.action_queue_lock:
                self.action_queue = Queue()
            with self.latest_action_lock:
                self.latest_action += max(1000, self.action_chunk_size + 1)
            self._stream += 1
            self._stream_active = True
            self._last_action_time = None
            self._empty_since = None
            self.must_go.set()
            self._record('stream_start', step_floor=self.latest_action)

    def pause_stream(self, reason):
        with self._stream_lock:
            self._stream_active = False
            with self.action_queue_lock:
                self.action_queue = Queue()
            if self._empty_since is not None:
                self._record('queue_empty_end', duration_s=time.monotonic()-self._empty_since,
                             reason=reason)
                self._empty_since = None
            self._record('stream_pause', reason=reason)

    def observe_queue(self):
        with self._stream_lock:
            if not self._stream_active:
                return
            depth = self.action_queue.qsize()
            if depth == 0 and self._empty_since is None:
                self._empty_since = time.monotonic()
                self._record('queue_empty_start', startup=self._last_action_time is None)
            elif depth and self._empty_since is not None:
                self._record('queue_empty_end', duration_s=time.monotonic()-self._empty_since,
                             reason='refilled', queue=depth)
                self._empty_since = None

    def send_observation(self, obs):
        started = time.monotonic()
        self._record('observation_send', step=obs.get_timestep(),
                     observation_wall=obs.get_timestamp(), must_go=obs.must_go,
                     queue=self.action_queue.qsize(), frame_age_s=self.robot.frame_age())
        result = super().send_observation(obs)
        self._record('observation_sent', step=obs.get_timestep(), ok=result,
                     duration_s=time.monotonic()-started)
        return result

    def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
        with self._stream_lock:
            if not self._stream_active:
                self._record('chunk_discarded', reason='paused', actions=len(incoming_actions))
                return
            with self.latest_action_lock:
                latest = self.latest_action
            with self.action_queue_lock:
                old = {a.get_timestep(): a.get_action() for a in self.action_queue.queue}
            future = [a for a in incoming_actions if a.get_timestep() > latest]
            overlap = [a for a in future if a.get_timestep() in old]
            # Compare raw joint predictions for the same future time (radians).
            revision = max((float((a.get_action()[:6]-old[a.get_timestep()][:6]).abs().max())
                            for a in overlap), default=0.0)
            self._record('chunk_received', first_step=(incoming_actions[0].get_timestep()
                         if incoming_actions else None), actions=len(incoming_actions),
                         future=len(future), stale=len(incoming_actions)-len(future),
                         overlap=len(overlap), queue_before=len(old),
                         max_overlap_revision_rad=revision,
                         first_action_wall=(incoming_actions[0].get_timestamp()
                         if incoming_actions else None))
            # A fully stale result must not clear an already valid new queue.
            if future:
                super()._aggregate_action_queues(future, aggregate_fn)

    def control_loop_action(self, *args, **kwargs):
        with self._stream_lock:
            if not self._stream_active:
                return None
            started = time.monotonic()
            result = super().control_loop_action(*args, **kwargs)
            self._record('action', step=self.latest_action, queue=self.action_queue.qsize(),
                         interval_s=(started-self._last_action_time
                                     if self._last_action_time is not None else None))
            self._last_action_time = started
            return result
