#!/usr/bin/env python3
"""Offline by default. Explicit, supervised hold/replay; never home or move gripper."""
import argparse
import bisect
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import time

from teach import build_robot, preflight, verify_revision, persist_episode


def load_trajectory(path, speed=0.25):
    if not math.isfinite(speed) or not 0 < speed <= 0.25:
        raise ValueError('First replay speed must be in (0, 0.25]')
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    meta = data['metadata']
    if meta.get('status') != 'motion_complete' or meta.get('fault') or not meta.get('finalization_complete'):
        raise ValueError('Require a complete, fault-free source episode')
    if meta.get('units', {}).get('q') != 'rad':
        raise ValueError('Require explicit radian units')
    samples = data['samples']
    if len(samples) < 2:
        raise ValueError('Need at least two samples')
    times, positions = [], []
    for row in samples:
        t, q = float(row['t_monotonic_s']), row['q']
        if len(q) != 6 or not all(math.isfinite(v) for v in [t, *q]):
            raise ValueError('Nonfinite or malformed sample')
        if times and (t <= times[-1] or t-times[-1] > 0.1):
            raise ValueError('Nonmonotonic time or source gap >100 ms')
        times.append(t)
        positions.append(q)
    times = [(t-times[0])/speed for t in times]
    if times[-1] > 120:
        raise ValueError('First replay must finish within 120 s')
    return times, positions, hashlib.sha256(raw).hexdigest()


def interpolate(times, positions, elapsed):
    if elapsed <= 0:
        return list(positions[0])
    if elapsed >= times[-1]:
        return list(positions[-1])
    i = bisect.bisect_right(times, elapsed)-1
    u = (elapsed-times[i])/(times[i+1]-times[i])
    return [a+(b-a)*u for a, b in zip(positions[i], positions[i+1])]


def require_near_start(actual, target, tolerance=0.05):
    if len(actual) != 6 or not all(math.isfinite(x) for x in actual):
        raise ValueError('Invalid current position')
    error = max(abs(a-b) for a, b in zip(actual, target))
    if error > tolerance:
        raise ValueError(f'Not at replay start: max error {error:.4f} rad; no automatic approach')


def approach_duration(actual, target):
    require_near_start(actual, target, tolerance=0.35)
    # Quintic interpolation peaks at 1.875; commanded approach <=0.05 rad/s.
    return max(3., 1.875*(max(abs(a-b) for a,b in zip(actual,target))+.02)/.05)


def approach_target(actual, target, elapsed, duration):
    u = max(0., min(1., elapsed/duration))
    blend = 10*u**3-15*u**4+6*u**5
    return [a+(b-a)*blend for a,b in zip(actual,target)]


def effective_target(robot, requested):
    target = robot._clip_joint_pos(requested.copy())
    if target is None:
        raise ValueError("SDK rejected target outside position limits")
    return target


def run(args, times, positions, digest):
    import numpy as np
    root = Path(__file__).resolve().parent
    if (root/'REPLAY_BLOCKED').exists():
        raise RuntimeError('Hardware replay paused after abrupt stop; review stopping behavior before re-enabling')
    lock = (root/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    robot = bus = None
    rows = []
    metadata = {'source': str(args.episode), 'source_sha256': digest,
                'mode': args.execute, 'speed_factor': args.speed,
                'status': 'not_started', 'fault': None, 'hardware_executed': False,
                'stop_behavior': 'disable; not guaranteed pose hold',
                'gravity_factor': 0.3, 'velocity_limit_source': 'official_sdk_default'}
    errors = []
    def interrupted(*unused):
        raise KeyboardInterrupt()
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, interrupted)
    try:
        metadata['sdk_revision'] = verify_revision()
        processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
        for line in processes.splitlines():
            if any(name in line for name in ['examples/gravity_comp.py', 'examples/teach_and_play.py',
                    'examples/position_hold.py', 'tools/motor_diag.py', 'a1z-controller.py', 'a1z-native.py']):
                raise RuntimeError('Another possible controller: '+line)
        print('现场测试：清空运动路径，配置能承接失能臂体的支撑。停止会失能，不保证悬停。')
        print('保持测试不走轨迹；仅显式 --approach 时缓慢接近起点。结束前有 5 秒终点保持，然后失能。')
        if input('现场准备好后输入 READY；其他输入退出：').strip() != 'READY':
            metadata['status'] = 'cancelled'
            return
        robot, bus = build_robot()
        metadata['preflight'] = preflight(robot, bus)
        q0 = robot._state.pos.copy()
        approach = 0.
        if args.execute == 'replay':
            if args.approach:
                approach = approach_duration(q0, positions[0])
            else:
                require_near_start(q0, positions[0])
            applied = [effective_target(robot, np.array(q)).tolist() for q in positions]
            metadata['max_source_clip_rad'] = max(abs(a-b) for q, v in zip(positions, applied) for a, b in zip(q, v))
            metadata['target_policy'] = 'official SDK tolerance clipping; raw source unchanged'
        metadata['approach_duration_s'] = approach
        metadata['approach_target_rad'] = positions[0] if approach else None
        metadata['kp'] = robot._default_kp.tolist()
        metadata['kd'] = robot._default_kd.tolist()
        metadata['max_joint_velocity_rad_s'] = robot._vel_limit.tolist()
        # Clear the motor driver's cached target before SDK enable_all.
        zero = np.zeros(6)
        robot._motor_chain.send_commands(zero, zero, zero, np.full(6, .05), zero)
        original_read, original_update = robot._read_state, robot._update
        state = {'start': None, 'last': None, 'fault': None, 'last_sample': 0., 'target': q0.copy()}
        ramp, settle, tail = 3., 1., 5.
        motion_duration = times[-1] if args.execute == 'replay' else 0.
        total = ramp+approach+settle+motion_duration+tail
        def read():
            original_read()
            bus.require_fresh(.1)
            q = robot._state.pos
            if not np.all(np.isfinite(q)):
                raise RuntimeError('Nonfinite feedback')
            robot._check_joint_limits(q)
        def update():
            now = time.monotonic()
            if state['start'] is None:
                # SDK start has read the fresh position but not yet sent a PD target.
                if args.execute == 'replay':
                    if approach:
                        require_near_start(robot._state.pos, q0, tolerance=.02)
                    else:
                        require_near_start(robot._state.pos, positions[0])
                state['start'] = now
                state['target'] = robot._state.pos.copy()
                q0[:] = state['target']
            try:
                if state['last'] is not None and now-state['last'] > .1:
                    raise RuntimeError('Control deadline missed >100 ms')
                state['last'] = now
                elapsed = now-state['start']
                if elapsed > total+.2:
                    raise RuntimeError('Run deadline expired')
                q = q0.copy()
                if args.execute == 'replay':
                    if approach and elapsed < ramp:
                        q = q0.copy()
                    elif approach and elapsed < ramp+approach:
                        q = np.array(approach_target(q0, positions[0], elapsed-ramp, approach))
                    elif elapsed < ramp+approach+settle:
                        if approach:
                            q = np.array(positions[0])
                        else:
                            q = np.array(approach_target(q0, positions[0], elapsed, ramp))
                    else:
                        if not state.get('start_verified'):
                            require_near_start(robot._state.pos, effective_target(robot, np.array(positions[0])))
                            state['start_verified'] = True
                        q = np.array(interpolate(times, positions, elapsed-ramp-approach-settle))
                if np.max(np.abs(robot._state.pos-state['target'])) > .15:
                    raise RuntimeError('Tracking deviation >0.15 rad; stop test')
                gain = min(1., elapsed/ramp)
                robot.gravity_comp_factor = .3*gain
                requested_q = q.copy()
                q = effective_target(robot, q)
                robot.command_joint_state({'pos': q, 'vel': zero,
                    'kp': robot._default_kp*gain, 'kd': robot._default_kd})
                state['target'] = q
                original_update()
                if now-state['last_sample'] >= .02:
                    rows.append({'t_s': elapsed, 'q': robot._state.pos.tolist(),
                        'requested_q': requested_q.tolist(), 'target_q': q.tolist(), 'dq': robot._state.vel.tolist(),
                        'feedback_age_s': bus.ages(), 'kp_scale': gain})
                    state['last_sample'] = now
                if elapsed >= total:
                    robot._stop_event.set()
            except Exception as exc:
                state['fault'] = repr(exc)
                raise
        robot._read_state, robot._update = read, update
        metadata['hardware_executed'] = True
        robot.start(initial_kp=zero, initial_kd=robot._default_kd)
        print(f'测试开始，最长 {total:.1f} 秒；Ctrl+C 立即停止并失能。', flush=True)
        last_notice = None
        while robot.is_running:
            elapsed = time.monotonic()-(state['start'] or time.monotonic())
            if elapsed >= total-tail:
                left = max(0, math.ceil(total-elapsed))
                if left != last_notice:
                    print(f'终点保持；{left} 秒后失能，支撑须就位。', flush=True)
                    last_notice = left
            time.sleep(.03)
        if state['fault']:
            raise RuntimeError(state['fault'])
        if not rows or rows[-1]['t_s'] < total-.1:
            raise RuntimeError('Controller stopped before completion')
        metadata['status'] = 'completed'
    except KeyboardInterrupt:
        metadata['status'] = 'operator_stopped'
    except Exception as exc:
        metadata.update(status='failed', fault=repr(exc))
    finally:
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
            signal.signal(sig, signal.SIG_IGN)
        if robot is not None:
            try:
                robot.stop()
            except Exception as exc:
                errors.append(repr(exc))
            if robot._thread is not None and robot._thread.is_alive():
                errors.append('Control thread did not stop')
            for motor in robot._motor_chain._motor_a_list+robot._motor_chain._motor_b_list:
                try:
                    motor.disable()
                except Exception as exc:
                    errors.append(repr(exc))
        if bus is not None:
            try:
                bus.shutdown()
            except Exception as exc:
                errors.append(repr(exc))
        if robot is not None:
            robot._motor_chain.disable_all = lambda: None
        metadata['stop_send_errors'] = errors
        out = root/'replay-results'
        out.mkdir(exist_ok=True)
        path = out/f'{time.monotonic_ns()}-{args.execute}.json'
        persist_episode(path, metadata, rows)
        print('RESULT', path, metadata['status'], metadata['fault'], errors, flush=True)
        lock.close()
    if metadata['status'] != 'completed' or errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--speed', type=float, default=.25)
    parser.add_argument('--execute', choices=['hold', 'replay'], help='Explicit supervised hardware test; omitted means offline only')
    parser.add_argument('--approach', action='store_true', help='Explicit short joint-space approach, <=0.35 rad per joint; no collision detection')
    args = parser.parse_args()
    if args.approach and args.execute != 'replay':
        parser.error('--approach requires --execute replay')
    times, positions, digest = load_trajectory(args.episode, args.speed)
    print(json.dumps({'offline_check': 'passed', 'source_sha256': digest,
          'frames': len(times), 'duration_s': times[-1], 'start_q': positions[0],
          'end_q': positions[-1], 'will_connect_hardware': bool(args.execute)}, indent=2))
    if args.execute:
        run(args, times, positions, digest)

if __name__ == '__main__':
    main()
