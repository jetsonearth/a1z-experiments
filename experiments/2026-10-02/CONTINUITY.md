# Continuous execution repair — 2026-10-05

Status: deployed to idle Jetson on 2026-10-05 at 18:19; physical validation pending. No evaluation session or motor was started.

Runtime backup: `/home/dimos/a1z-experiments/2026-10-02/deployment-backups/continuity-20261005-181901` (manifest records previous-file existence and deployed hashes).

## Changes

- `robot_proc.smooth_step` now analytically integrates acceleration, cruise and
  braking phases from the existing commanded position and velocity. It cannot
  reset a moving velocity merely because a target is crossed. Targets inside
  braking distance require braking past the target before returning. Limits
  remain the existing 0.5 rad/s and 3 rad/s² when using the default smooth preset.
  This bounds the base reference velocity/acceleration, not jerk, actual joint
  motion, or the integral-offset contribution to the final position command.
- The client keeps the next two queued arm points when a new chunk overlaps,
  preserving their discrete velocity, and fades toward the new prediction over
  the remaining overlap with a quintic envelope. Gripper aggregation is unchanged.
  With fewer than four overlap points it preserves remaining old arm points;
  with no overlap only the downstream bounded-acceleration tracker can bridge.
  This is not a claim of globally bounded acceleration in the policy queue.
- A single observation thread captures and sends fresh observations, with no
  backlog. Serialization and RPC do not hold the stream lock. Step tagging is
  atomic with capture; old responses are still fenced out across stop/handback.
  The sender activates only after robot engagement succeeds. Sender errors reach
  the main loop; shutdown closes transport before joining the sender.
- New metadata: `async_client_version=continuous_stream_v2` and
  `trajectory_version=bounded_accel_v2` (or `legacy_velocity_clip` with
  `--no-smooth`). Action timing now includes call duration; chunk_splice records
  overlap and transition length. No model, integral gain or vendor gain changes.

## Verification

21 unit regressions passed against installed Jetson dependencies, without
hardware: six trajectory-contract tests, 15 server/client tests. Coverage includes
6,000 randomized six-joint moving-target steps, abrupt reversal, near-target
braking, reduced speed limits, timestep partition invariance, nonconstant chunk
splicing, repeated replans converging to the new goal, slow RPC concurrent with
action/pause/reset, stale responses and observation errors. Existing fake-robot
HIL lifecycle and toy stick-slip tests passed. The previous toy test's exemption
for acceleration violations on arrival was removed.

Pure software replay of recorded policy targets from 14 valid trials in
`eval-runs/20261005-175100` used a 100 Hz zero-order-held target, identical initial
pose and soft-start setting for old/new smoothers. Per-step maximum-joint
acceleration violations (>3+1e-6 rad/s²): old 9,118, new 0. This is a reference
trajectory replay, NOT a robot dynamics or contact simulation. Per-trial P95
reference-to-policy target error increased by at most 0.342 degrees. It does not
measure real tracking error or the effect of the new chunk splice in closed loop.

Run in the project environment:

```
python -m unittest test_continuity test_async_policy
python test_hil_client.py
python test_smooth.py
```

## Limits and next check

Chunk transitions intentionally delay adoption of a revised prediction. Actual
insertion precision, speed, contact behavior, and gripper slip remain unverified.
No success detector or post-insertion hold was added. Compare a new batch using
the same checkpoint/start conditions; inspect actual motion as well as command
continuity, observation timing, queue depth and intervention labels.

All source edits remain uncommitted. Restoring local files alone does not roll
back Jetson; use the device deployment backup for runtime rollback while idle.
