#!/usr/bin/env python3
"""Bounded, arm-only teaching check for the pinned Galaxea gripper SDK.
No replay, position stiffness, gripper commands, homing, or zero calibration.
The operator must support the arm throughout; stopping disables the motors.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/home/dimos/GALAXEA-A1Z')
REV = 'e931ecd0e25ad35df251097ba42921b3d2fa7224'
sys.path.insert(0, str(ROOT))


class FeedbackBus:
    """Track real, correctly formatted feedback separately for each joint."""
    def __init__(self, bus, clock=time.monotonic, sleeper=time.sleep):
        self.bus, self.clock, self.sleeper = bus, clock, sleeper
        self.last_send = None
        self.send_gap_s = 0.0003
        self.last = [None] * 6
        self.count = [0] * 6
        self.tx = collections.deque(maxlen=64)

    def send(self, msg, timeout=None):
        # This gs_usb/CAN box loses joint-6 replies with tightly packed 1..6
        # bursts. Hardware A/B check: 0/123 replies unpaced, 123/123 paced.
        # Keep the freshness guard; pace transport rather than masking loss.
        if self.last_send is not None:
            delay = self.send_gap_s - (self.clock() - self.last_send)
            if delay > 0:
                self.sleeper(delay)
        self.tx.append((msg.arbitration_id, bytes(msg.data)))
        result = self.bus.send(msg, timeout=0.02)
        self.last_send = self.clock()
        return result

    def recv(self, timeout=0.0):
        # The SDK uses nonblocking reads. Bound filtering even on a noisy bus.
        for _ in range(32):
            msg = self.bus.recv(timeout=timeout)
            if msg is None:
                return None
            mid, data = msg.arbitration_id, bytes(msg.data)
            if msg.is_error_frame:
                raise RuntimeError('CAN error frame')
            if not msg.is_rx or msg.is_extended_id or mid not in range(1, 7) or len(data) != 8:
                continue
            if (mid, data) in self.tx:
                continue
            if mid <= 3 and data[0] >> 5 != 1:
                continue
            if mid >= 4 and data[0] & 15 != mid:
                continue
            self.last[mid - 1] = self.clock()
            self.count[mid - 1] += 1
            return msg
        raise RuntimeError('Too many unrelated CAN frames')

    def ages(self):
        now = self.clock()
        return [None if t is None else now - t for t in self.last]

    def require_fresh(self, max_age=0.1):
        ages = self.ages()
        if any(a is None or a > max_age for a in ages):
            raise RuntimeError(f'Missing/stale joint feedback: {ages}')
        return ages

    def shutdown(self):
        self.bus.shutdown()


class MotionGate:
    """Recording time starts with sustained measured motion, not Enter."""
    def __init__(self, q0, now, duration=120.0, wait_limit=120.0):
        self.q0 = list(q0)
        self.wait_started = now
        self.duration = duration
        self.wait_limit = wait_limit
        self.motion_started = None
        self.candidate = None
        self.hits = 0
        self.last_motion = None
        self.motion_anchor = list(q0)

    def finished(self, now):
        return self.motion_started is not None and now - self.last_motion >= 2.0

    @property
    def deadline(self):
        if self.motion_started is None:
            return self.wait_started + self.wait_limit
        return self.motion_started + self.duration

    def observe(self, q, now):
        if self.motion_started is not None:
            if max(abs(a - b) for a, b in zip(q, self.motion_anchor)) >= 0.003:
                self.motion_anchor = list(q)
                self.last_motion = now
            return
        if max(abs(a - b) for a, b in zip(q, self.q0)) >= 0.015:
            self.hits += 1
            if self.candidate is None:
                self.candidate = now
            if self.hits >= 3:
                self.motion_started = self.candidate
                self.last_motion = now
                self.motion_anchor = list(q)
        else:
            self.hits = 0
            self.candidate = None


def build_robot():
    import numpy as np
    from a1z.robots.get_robot import get_a1z_robot
    robot = get_a1z_robot(can_channel='can1', with_gripper=False,
                          zero_gravity_mode=True, gravity_comp_factor=0.0,
                          control_freq_hz=250, min_freq_hz=80,
                          urdf_path=str(ROOT / 'a1z/robot_models/a1z/A1Z_G1Z.urdf'))
    bus = FeedbackBus(robot._bus)
    robot._bus = bus
    for motor in robot._motor_chain._motor_a_list + robot._motor_chain._motor_b_list:
        motor.bus = bus
    # Keep the pinned official SDK velocity limits; no recording-specific override.
    robot._torque_clip = np.array([4., 4., 4., 2., 0.5, 0.5])
    return robot, bus


def preflight(robot, bus):
    import numpy as np
    chain = robot._motor_chain
    # Do not enable motors during preflight. MotorB returns state on disable;
    # MotorA returns it after the same zero-stiffness probe verified on this arm.
    chain.disable_all()
    for motor in chain._motor_a_list:
        motor.send_mit_command(pos=0, vel=0, kp=0, kd=0.05, torque=0)
    time.sleep(0.03)
    until = time.monotonic() + 0.15
    while time.monotonic() < until:
        robot._read_state()
        if all(t is not None and time.monotonic() - t < 0.1 for t in bus.last):
            break
        time.sleep(0.001)
    bus.require_fresh(0.1)
    robot._check_motor_errors()
    robot._check_motor_temps()
    q = robot._state.pos.copy()
    robot._check_joint_limits(q)
    tau = robot._gravity_model.compute_gravity_torque(q)
    if not np.all(np.isfinite(tau)) or tau.shape != (6,):
        raise RuntimeError('Invalid gravity model output')
    return {'q_rad': q.tolist(), 'gravity_model_nm': tau.tolist(),
            'feedback_counts': list(bus.count), 'error_codes': robot._state.error_codes.tolist(),
            'model_nq': robot._gravity_model.nq}


def verify_revision():
    rev = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if rev != REV:
        raise RuntimeError(f'SDK changed: expected {REV}, got {rev}; review before running')
    paths = ['a1z/motor_drivers', 'a1z/robots', 'a1z/dynamics', 'a1z/robot_models']
    if subprocess.check_output(['git', '-C', str(ROOT), 'diff', 'HEAD', '--', *paths]):
        raise RuntimeError('SDK has tracked control/model changes; review before running')
    return rev


def persist_episode(path, metadata, samples):
    temporary = path.with_suffix('.incoming')
    with temporary.open('w') as stream:
        json.dump({'metadata': metadata, 'samples': samples}, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def finalize_episode(path, metadata, samples, camera):
    """Arm is already stopped. Persist joints before waiting on any camera work."""
    metadata['finalization_complete'] = False
    if camera is not None:
        metadata['camera'] = {'dataset_dir': str(camera.output), 'finalization_pending': True}
    persist_episode(path, metadata, samples)
    print('SAVED ' + str(path), flush=True)
    if camera is not None:
        try:
            metadata['camera'] = camera.stop(timeout=60)
        except BaseException as exc:
            metadata['camera'] = {'dataset_dir': str(camera.output),
                                  'error': 'Camera finalization interrupted: ' + repr(exc)}
        metadata['camera_recorded'] = not metadata['camera'].get('error') and bool(metadata['camera'].get('frames'))
        if not metadata['camera_recorded']:
            metadata['camera_error'] = metadata['camera'].get('error') or 'No camera frames'
            # Keep the independent joint acquisition outcome intact.
    metadata['finalization_complete'] = True
    persist_episode(path, metadata, samples)
    print('SAVED ' + str(path), flush=True)


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--camera', action='store_true', help='Record D405 in a separate LeRobot process; no camera work in control loop')
    parser.add_argument('--seconds', type=float, default=120.0, help='Maximum recording duration; normal end is 2 seconds still')
    args = parser.parse_args()
    if not 5 <= args.seconds <= 120:
        parser.error('Maximum recording duration must be between 5 and 120 seconds')
    outdir = Path(__file__).resolve().parent / 'episodes'
    outdir.mkdir(exist_ok=True)
    lock = (outdir.parent / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rev = verify_revision()
    # Avoid coexistence with a previously launched SDK example/controller.
    processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    for line in processes.splitlines():
        if any(x in line for x in ['examples/gravity_comp.py', 'examples/teach_and_play.py',
                                   'examples/position_hold.py', 'tools/motor_diag.py',
                                   'a1z-controller.py', 'a1z-native.py']):
            raise RuntimeError('Existing possible controller: ' + line)
    robot = bus = None
    camera = None
    samples = []
    pre_roll = collections.deque(maxlen=60)
    gate = None
    meta = {'sdk_revision': rev, 'source': 'kinesthetic_hardware_check',
            'units': {'q': 'rad', 'dq': 'rad/s', 'effort': 'Nm'},
            'gripper_controlled': False, 'camera_recorded': False,
            'model': 'A1Z_G1Z.urdf; wrist camera mass not separately calibrated',
            'max_gravity_factor': 0.3, 'kp': [0] * 6,
            'can_min_send_gap_s': 0.0003,
            'prepare_seconds': 0, 'stop_after_still_seconds': 2.0, 'wait_for_motion_seconds': 120,
            'recording_trigger': 'measured joint displacement >=0.015 rad for 3 updates; may include passive drift',
            'max_joint_displacement_rad': None, 'velocity_limit_source': 'official_sdk_default',
            'feedforward_caps_nm': [4, 4, 4, 2, 0.5, 0.5],
            'experiment_date': '2026-10-02',
            'jetson_wall_clock_utc_unverified': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    status, fault = 'preflight_only', None
    stop_errors = []
    signal.signal(signal.SIGTERM, lambda *unused: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGHUP, lambda *unused: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if args.camera and not args.preflight:
            from camera_session_control import CameraSession
            camera_path = outdir.parent / 'datasets' / ('20261002-' + str(time.monotonic_ns()))
            camera = CameraSession(camera_path, max_seconds=300)
            ready = camera.prepare(timeout=30)
            if not ready.get('ready'):
                raise RuntimeError('Camera not ready: ' + str(ready.get('error')))
            meta['camera_ready'] = ready
            print('CAMERA_READY：D405 已开始采集；视频编码将在机械臂停用后完成。', flush=True)
        robot, bus = build_robot()
        meta['max_joint_velocity_rad_s'] = robot._vel_limit.tolist()
        meta['preflight'] = preflight(robot, bus)
        robot.stop()
        print('PREFLIGHT_OK ' + json.dumps(meta['preflight']), flush=True)
        if args.preflight:
            return
        print('\n底座须固定，夹爪空载。双手托住臂体，避开关节缝隙。', flush=True)
        print('本轮为低比例辅助托重，不会悬停。全程托住，只缓慢移动几厘米。', flush=True)
        print('按回车后立即待录，无倒计时；你开始移动才录，连续静止 2 秒后自动保存。', flush=True)
        print('待录和单次录制各最多 2 分钟。结束/异常会停用电机，不回零。', flush=True)
        print('按开始时先让机械臂靠稳垫子；拖动时持续托住，结束前放回支撑。', flush=True)
        input('按回车开始监听；Ctrl+C 可中止。\n')
        # Fresh state after the operator's wait. Preload benign cached commands
        # before SDK enable_all() to avoid restoring an old position target.
        if camera is not None and not camera.is_running():
            raise RuntimeError('Camera exited before motion listening started')
        meta['preflight_at_start'] = preflight(robot, bus)
        q0 = robot._state.pos.copy()
        zero = np.zeros(6)
        robot._motor_chain.send_commands(zero, zero, zero, np.full(6, 0.05), zero)
        time.sleep(0.02)
        robot._read_state()
        start = time.monotonic()
        gate = MotionGate(q0.tolist(), start, duration=args.seconds)
        original_read = robot._read_state
        original_update = robot._update
        original_send = robot._motor_chain.send_commands
        last_command = {}
        last_sample = [0.0]
        first_read = [True]
        control_fault = [None]

        def send_recorded(*a, **kw):
            values = dict(zip(['pos', 'vel', 'kp', 'kd', 'torque'], a))
            values.update(kw)
            last_command.clear()
            last_command.update({k: np.asarray(values[k]).tolist()
                                 for k in ['pos', 'vel', 'kp', 'kd', 'torque']})
            return original_send(*a, **kw)

        def checked_read():
            if first_read[0]:
                until = time.monotonic() + 0.03
                while time.monotonic() < until:
                    original_read()
                    if all(t is not None and time.monotonic() - t < 0.1 for t in bus.last) and np.all(robot._state.error_codes[3:] == 1):
                        break
                first_read[0] = False
            else:
                original_read()
            bus.require_fresh(0.1)
            now = time.monotonic()
            if now >= gate.deadline:
                raise RuntimeError('Session deadline reached')
            q = robot._state.pos
            if not np.all(np.isfinite(q)):
                raise RuntimeError('Nonfinite joint position')
            limits = np.asarray(robot._joint_limits)
            if np.any(q < limits[:, 0] - 0.08) or np.any(q > limits[:, 1] + 0.08):
                raise RuntimeError('Joint outside first-test range')
            errs = robot._state.error_codes
            if np.any(errs[:3] != 0) or np.any(errs[3:] != 1):
                raise RuntimeError(f'Unexpected active motor status: {errs}')
            gate.observe(q.tolist(), now)
            # Ramp in from zero; this is deliberately not full gravity compensation.
            robot.gravity_comp_factor = 0.3 * min(1.0, max(0.0, (now - start) / 2.0))

        def recorded_update():
            try:
                original_update()
            except Exception as exc:
                control_fault[0] = repr(exc)
                st = robot._state
                meta['fault_state'] = {'q': st.pos.tolist(), 'dq': st.vel.tolist(), 'feedback_age_s': bus.ages()}
                raise
            now = time.monotonic()
            if now - last_sample[0] >= 0.02:
                s = robot._state
                row = {'t_monotonic_s': now, 't_session_s': now - start,
                                'q': s.pos.tolist(), 'dq': s.vel.tolist(),
                                'effort': s.eff.tolist(), 'errors': s.error_codes.tolist(),
                                'feedback_age_s': bus.ages(), 'rx_counts': list(bus.count),
                                'gravity_factor': robot.gravity_comp_factor,
                                'motor_command': dict(last_command)}
                if gate.motion_started is None:
                    pre_roll.append(row)
                else:
                    if pre_roll:
                        samples.extend(pre_roll)
                        pre_roll.clear()
                    samples.append(row)
                last_sample[0] = now

        robot._motor_chain.send_commands = send_recorded
        robot._read_state = checked_read
        robot._update = recorded_update
        robot.start(initial_kp=zero, initial_kd=np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1]))
        print('ARMED：正在待录。现在可以缓慢拖动几厘米，检测到移动才开始录制。', flush=True)
        status = 'waiting_for_motion'
        last_notice = None
        while robot.is_running:
            if camera is not None and not camera.is_running():
                raise RuntimeError('Camera process stopped; preserving partial episode')
            remaining = gate.deadline - time.monotonic()
            recording = gate.motion_started is not None
            if recording and status != 'recording':
                status = 'recording'
                last_notice = None
                print('RECORDING：已检测到移动，正式录制开始；请持续托住。', flush=True)
            if recording and gate.finished(time.monotonic()):
                status = 'motion_complete'
                print('连续静止 2 秒，录制结束，正在保存并停用。', flush=True)
                break
            if remaining <= 0.15:
                status = 'recording_time_limit' if recording else 'no_motion_timeout'
                break
            whole = int(remaining)
            notice = whole if recording else whole // 10
            if notice != last_notice:
                last_notice = notice
                if recording:
                    print(f'录制中 / 已录 {len(samples)} 帧；停止移动 2 秒即保存。' +
                          (' — 即将达到 2 分钟上限，请放回支撑。' if remaining <= 10 else ''), flush=True)
                else:
                    print(f'待录中，尚未检测到明显移动；最多再等 {whole + 1} 秒。' +
                          (' 请放回垫子，准备停用。' if remaining <= 10 else ''), flush=True)
            time.sleep(0.04)
        if status in ['recording', 'waiting_for_motion']:
            status = 'controller_stopped_early'
            fault = control_fault[0] or 'See session terminal log for control-loop exception'
    except KeyboardInterrupt:
        status = 'operator_stopped'
    except Exception as exc:
        status, fault = 'failed', repr(exc)
        print('CHECK_FAILED ' + fault, flush=True)
    finally:
        # Repeated Stop must not interrupt motor shutdown or durable saving.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if robot is not None:
            try:
                robot.stop()
            except Exception as exc:
                stop_errors.append(repr(exc))
            thread = robot._thread
            if thread is not None and thread.is_alive():
                stop_errors.append('Controller thread still alive after stop')
            # SDK disable_all intentionally suppresses send errors. Send once more
            # individually and report any errors instead of claiming confirmed off.
            for motor in robot._motor_chain._motor_a_list + robot._motor_chain._motor_b_list:
                try:
                    motor.disable()
                except Exception as exc:
                    stop_errors.append(f'motor {motor.motor_id}: {exc!r}')
        if bus is not None:
            bus.shutdown()
        # From here no active arm control remains; preserve the data on disconnect.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        print('FINALIZING: 控制已结束，正在保存关节数据和视频，请等待。', flush=True)
        if gate is not None:
            meta['motion_detected'] = gate.motion_started is not None
            meta['waiting_duration_s'] = ((gate.motion_started or time.monotonic()) - gate.wait_started)
            if gate.motion_started is not None:
                for row in samples:
                    row['t_s'] = row['t_monotonic_s'] - gate.motion_started
        meta.update(status=status, fault=fault, stop_send_errors=stop_errors,
                    samples=len(samples), disable_commands_sent=not stop_errors)
        stamp = '20261002-' + str(time.monotonic_ns()) + f'-{os.getpid()}'
        path = outdir / (stamp + ('-preflight' if args.preflight else '-teaching') + '.json')
        finalize_episode(path, meta, samples, camera)
        print('STOP: 停用指令已发送；机械臂应由桌面或手支撑。' if not stop_errors
              else 'STOP ERROR: 请使用机械臂独立电源停止。' + repr(stop_errors), flush=True)
        # Prevent the SDK finalizer from sending after bus shutdown.
        if robot is not None:
            robot._motor_chain.disable_all = lambda: None
        fcntl.flock(lock, fcntl.LOCK_UN)
    if status in ['failed', 'controller_stopped_early'] or stop_errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
