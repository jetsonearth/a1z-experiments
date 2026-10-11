#!/usr/bin/env python3
"""Attended SpaceMouse Compact teleop using vendor PD/gravity/limits.

Without --execute: read the mouse only. With --execute: wait for typed READY
before touching CAN. Left button is a clutch; release holds measured pose.
Right button or s pauses; g resumes. r toggles rotation, initially disabled.
q twice exits after the operator supports the arm. No gripper or homing command.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import struct
import sys
import termios
import time
import tty

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / 'spacemouse-state.json'
STALE_S = .15


class MouseState:
    def __init__(self):
        self.raw = [0] * 6
        self.updated = [float('-inf')] * 2
        self.buttons = 0
        self.button_seen = False
        self.packets = 0

    def feed(self, data, now):
        if not data:
            raise OSError('SpaceMouse disconnected')
        if data[0] in (1, 2):
            if len(data) != 7:
                raise ValueError('Unexpected Compact axis report length')
            group = data[0] - 1
            self.raw[group*3:group*3+3] = struct.unpack('<hhh', data[1:7])
            self.updated[group] = now
        elif data[0] == 3:
            if len(data) < 2:
                raise ValueError('Short button report')
            self.buttons = data[1] & 3
            self.button_seen = True
        else:
            return
        self.packets += 1

    def axes(self, now, rotation=False):
        # Right-handed base-frame mapping: raw x -> +X, raw y -> -Y,
        # raw z -> -Z. Apply the same proper rotation to angular axes.
        out = []
        for i, raw in enumerate(self.raw):
            value = max(-1., min(1., raw/350.))
            if now-self.updated[i//3] > STALE_S or abs(value) < .05:
                value = 0.
            if i >= 3 and not rotation:
                value = 0.
            out.append(value * (1 if i % 3 == 0 else -1))
        return out

    def stale_motion(self, now, rotation):
        return any(now-self.updated[g] > STALE_S and any(self.raw[g*3:g*3+3])
                   for g in range(2 if rotation else 1))


class Clutch:
    def __init__(self):
        self.released = False
        self.active = False
        self.paused = False

    def reset(self, paused=None):
        self.released = False
        self.active = False
        if paused is not None:
            self.paused = paused

    def update(self, mouse, now, rotation):
        left, right = bool(mouse.buttons & 1), bool(mouse.buttons & 2)
        if right:
            self.reset(paused=True)
        if mouse.button_seen and not left and not right:
            self.released = True
        if mouse.stale_motion(now, rotation):
            self.reset()
        self.active = bool(left and not right and self.released and not self.paused)
        return self.active


class Mouse:
    def __init__(self):
        paths = []
        for item in Path('/sys/class/hidraw').glob('hidraw*'):
            if 'HID_ID=0003:0000256F:0000C635' in (item/'device/uevent').read_text():
                paths.append(Path('/dev')/item.name)
        if len(paths) != 1:
            raise RuntimeError(f'Expected one SpaceMouse Compact, found {paths}')
        self.path = paths[0]
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        self.state = MouseState()

    def poll(self):
        # Drain old reports before acting. A backlog never becomes delayed motion.
        for _ in range(128):
            if not select.select([self.fd], [], [], 0)[0]:
                return self.state
            self.state.feed(os.read(self.fd, 64), time.monotonic())
        raise RuntimeError('SpaceMouse report backlog')

    def close(self):
        os.close(self.fd)


def differential_target(kin, q, axes, dt, linear, angular, limits, vendor_vel):
    import numpy as np
    pin, k = kin.pin, kin.k
    if not 0 < dt <= .1:
        raise ValueError('Control tick delayed; release clutch')
    q = np.asarray(q, dtype=float)
    twist = np.asarray(axes, dtype=float) * np.array([linear]*3+[angular]*3)
    if q.shape != (6,) or twist.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(twist).all():
        raise ValueError('Nonfinite or malformed motion input')
    J = pin.computeFrameJacobian(k._model, k._data, q, k._frame_id, pin.LOCAL_WORLD_ALIGNED)
    # Damped differential IK; no global pose solve or branch-changing reset.
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4*np.eye(6), twist)
    result = pin.integrate(k._model, q, dq*dt)
    bounds = np.asarray(limits)
    if (not np.isfinite(result).all() or np.any(np.abs(dq) > vendor_vel)
            or np.any(result < bounds[:, 0]) or np.any(result > bounds[:, 1])):
        raise ValueError('Requested step exceeds vendor joint/velocity limits; release clutch')
    return result


def save_state(**data):
    data['updated_at'] = time.time()
    temp = STATE_FILE.with_suffix('.incoming')
    temp.write_text(json.dumps(data, ensure_ascii=False))
    temp.replace(STATE_FILE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--seconds', type=float, default=0, help='Input-monitor duration; 0 = until stopped')
    parser.add_argument('--linear', type=float, default=.025, help='Full-deflection sensitivity, m/s')
    parser.add_argument('--angular', type=float, default=.25, help='Full-deflection sensitivity, rad/s')
    args = parser.parse_args()
    import math
    if not all(math.isfinite(x) and x > 0 for x in (args.linear, args.angular)):
        parser.error('Sensitivities must be finite and positive')
    lock = (ROOT/'spacemouse.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mouse = Mouse()
    arm = bus = controller_lock = old_tty = flight = None
    clutch = Clutch()
    rotation = False
    start = last = time.monotonic()
    last_print = quit_at = 0.
    q_cmd = None
    error = None
    try:
        print('SpaceMouse 遥操 · 厂家控制参数 · 不归零 · 本版不控制夹爪', flush=True)
        if args.execute:
            import numpy as np
            from teach import build_robot, preflight, verify_revision
            from insertion import Kin
            revision = verify_revision()
            kin = Kin()
            controller_lock = (ROOT/'controller.lock').open('a')
            fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print('左键按住才移动，松开保持；右键/S 暂停，G 恢复；R 开关旋转；先支撑臂体再按两次 Q 失能退出。', flush=True)
            print('先确认底座固定、臂体有支撑、活动范围清空；首轮只测空中平移。', flush=True)
            save_state(state='await_ready', packets=mouse.state.packets, rotation=False)
            if input('现场就绪，输入 READY：').strip() != 'READY':
                return
            print('SPACEMOUSE_STARTING', flush=True)
            # No CAN construction, probe, enable or gripper command before READY.
            arm, bus = build_robot(with_gripper=False)
            preflight(arm, bus)
            arm.zero_gravity_mode = False
            arm.start()
            q_cmd = arm.get_joint_state()['pos'].copy()
            folder = ROOT/'teleop-runs'/time.strftime('%Y%m%d-%H%M%S')
            folder.mkdir(parents=True)
            flight = (folder/'flight.jsonl').open('a', buffering=1)
            (folder/'metadata.json').write_text(json.dumps({'sdk_revision': revision,
                'controller': 'vendor PD and gravity defaults', 'gripper_controlled': False,
                'linear_sensitivity_m_s': args.linear, 'angular_sensitivity_rad_s': args.angular,
                'axis_mapping': 'raw x,-y,-z and rx,-ry,-rz -> base frame',
                'end_effector_frame': kin.k._model.frames[kin.k._frame_id].name}, indent=2))
            old_tty = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            print('\nSPACEMOUSE_HOLD — 先松开左右键，再按住左键推旋钮。', flush=True)
        else:
            print('INPUT_ONLY: only reading HID; no robot connection', flush=True)
        last = time.monotonic()
        was_active = False
        while True:
            now = time.monotonic()
            dt, last = now-last, now
            if not args.execute and args.seconds and now-start >= args.seconds:
                break
            try:
                state = mouse.poll()
            except (OSError, ValueError, RuntimeError) as exc:
                if arm is None:
                    raise
                clutch.reset(paused=True)
                error = str(exc)
                state = mouse.state
            if args.execute:
                if not arm.is_running:
                    raise RuntimeError('Vendor controller stopped; inspect its fault output')
                while select.select([sys.stdin], [], [], 0)[0]:
                    key = os.read(sys.stdin.fileno(), 1).decode(errors='ignore').lower()
                    if key in ('s', '\x1b', '\x03'):
                        clutch.reset(paused=True)
                    elif key == 'g' and error is None:
                        clutch.reset(paused=False)
                    elif key == 'r':
                        rotation = not rotation
                        clutch.reset()
                    elif key == 'q':
                        clutch.reset(paused=True)
                        if now-quit_at < 2:
                            return
                        quit_at = now
                        print('臂体放稳支撑后，再按 Q 失能退出。', flush=True)
                active = clutch.update(state, now, rotation)
                if error:
                    active = False
                measured = arm.get_joint_state()
                bus.require_fresh(.1)
                if dt > .1:
                    clutch.reset(paused=True)
                    active = False
                if active != was_active:
                    q_cmd = measured['pos'].copy()
                    arm.command_joint_state({'pos': q_cmd, 'vel': np.zeros(6)})
                if active:
                    try:
                        target = differential_target(kin, q_cmd, state.axes(now, rotation), dt,
                            args.linear, args.angular, arm._joint_limits, arm._vel_limit)
                        arm.command_joint_state({'pos': target, 'vel': np.zeros(6)})
                        q_cmd = target
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        clutch.reset(paused=True)
                        q_cmd = measured['pos'].copy()
                        arm.command_joint_state({'pos': q_cmd, 'vel': np.zeros(6)})
                        active = False
                        print('SPACEMOUSE_HOLD', str(exc), flush=True)
                # A pause event must hold even when reset() cleared clutch.active.
                if not active and was_active:
                    q_cmd = measured['pos'].copy()
                    arm.command_joint_state({'pos': q_cmd, 'vel': np.zeros(6)})
                was_active = active
                flight.write(json.dumps({'t': now, 'q': measured['pos'].tolist(),
                    'dq': measured['vel'].tolist(), 'target': q_cmd.tolist(),
                    'axes': state.axes(now, rotation), 'buttons': state.buttons, 'active': active})+'\n')
            else:
                active = False
            if now-last_print >= .5:
                last_print = now
                status = 'teleop_active' if active else ('teleop_hold' if args.execute else 'input_only')
                data = {'state': status, 'raw': state.raw, 'axes': state.axes(now, True),
                        'buttons': state.buttons, 'packets': state.packets, 'rotation': rotation,
                        'paused': clutch.paused, 'error': error}
                if arm is not None:
                    data.update(q=measured['pos'].tolist(), temp_rotor=measured['temp_rotor'].tolist(), tip_mm=(kin.fk(measured['pos'])[:3, 3]*1000).tolist())
                save_state(**data)
                print(('SPACEMOUSE_ACTIVE' if active else 'SPACEMOUSE_HOLD' if args.execute else 'INPUT'),
                      'raw=',state.raw,'buttons=',state.buttons,'rotation=',rotation,
                      'paused=',clutch.paused, 'error=',error or '-', flush=True)
            time.sleep(max(0., .02-(time.monotonic()-now)))
    finally:
        if old_tty is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
        if arm is not None:
            arm.stop()
        if bus is not None:
            bus.shutdown()
        if flight is not None:
            flight.close()
        mouse.close()
        if controller_lock is not None:
            controller_lock.close()
        lock.close()
        save_state(state='ended', error=error, packets=mouse.state.packets)
        print('SESSION_SAVED SpaceMouse session ended', flush=True)


if __name__ == '__main__':
    main()
