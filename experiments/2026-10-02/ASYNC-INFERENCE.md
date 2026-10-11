# A1Z asynchronous inference: fresh-observation adapter

Current execution repair: see [CONTINUITY.md](CONTINUITY.md) for the continuous_stream_v2 sender, chunk splice and bounded_accel_v2 tracker. The sections below document the earlier fresh-observation adapter and its initial deployment.

Implementation date: 2026-10-05. Deployed to Jetson at 16:51; server startup and client READY gate verified at 16:53. Not physically validated.

## Problem and scope

In `eval-runs/20261005-161424`, policy target updates normally arrived every
33.5 ms, but recurring gaps were 134–167 ms, about 1.07 s apart. The later
`20261005-163237` batch exhibited the same pattern. A read of that later server
log contained 42 forced (`must_go`) requests out of 43 inference calls.

The installed LeRobot server filters observations using a raw state-distance
threshold of 1. A1Z joint positions are radians, so fine motions can be rejected
even when the image changed. When the action queue empties, a forced request
bypasses that check. This is strong evidence for queue starvation, not proof
that every acceleration peak has this cause. Startup overshoot and contact
forces still require separate diagnosis.

`eval.sh` now starts the project-local `a1z_policy_server.py`. Its adapter accepts
fresh action-timestep observations without the joint-similarity filter. It
rejects duplicates/out-of-order steps (including forced duplicates), keeps one
latest pending observation, and leaves upstream model inference and transport
intact. Installed LeRobot files are not modified.

The client retains 30-step chunks, the 0.7 refill threshold, and upstream
30% old / 70% new fusion. The adapter serializes consumption, incoming-queue
replacement, and stream reset; paused and previous-stream responses cannot
restore old actions after a stop or handback. Stream reset keeps the existing
1000-step jump, increasing it if needed to exceed the received chunk length.

No controller gains, integral code, velocity caps, camera settings, action
filters, or contact behavior are changed. This does not guarantee smooth
physical motion or successful insertion.

## Timing evidence for the next run

- Client session: `async-timing.jsonl`, buffered in memory during motion and
  appended at trial completion and shutdown. An abrupt process kill can lose
  buffered events, just as it can lose the current trial log.
- Server: `server-runs/<timestamp>-async.jsonl`, a unique file per server process,
  flushed after inference. Server startup prints the path. Old `/tmp/policy_server.log`
  still exists for human-readable messages, but is not the durable timing source.
- Every event has monotonic `t` and wall time `wall`. Client events include stream
  number, observation timestep/timestamp, image age, send duration, action
  interval, queue-empty intervals (startup distinguished from running starvation),
  incoming overlap count, and maximum same-timestep joint prediction revision.
- Join server/client events by observation timestamp and step. Both currently
  run on one Jetson; wall-time latency comparisons on separate machines would
  require clock synchronization. Image age measures host receipt, not exposure.
- Trial metadata marks `async_client_version: timed_stream_v1`; this is a client
  marker, not proof that the server adapter is running. Verify the server trace's
  `observation_filter: fresh_timestep_v1` as well.

Acceptance: fewer running queue-empty intervals and periodic target-update gaps,
actual overlapping responses, bounded latest-only pending work, and no stale
actions across stop/takeover/handback. Then compare target discontinuities and
measured speed peaks; an improvement in scheduling is not physical validation.

## Offline checks and eventual use

Run `python test_async_policy.py` with the project's LeRobot environment. It uses
real upstream queue/transport methods with fake actions, no model or hardware.
Also run `python test_hil_client.py` for the existing fake-robot takeover flow.
Verified on 2026-10-05: all 10 regressions passed using the installed Jetson
LeRobot dependencies with in-memory test modules; the fake-robot HIL test also
passed. No model, server listener, or physical controller was started. The Mac
venv could not import the transport because grpcio 1.71.0 is older than its
generated stubs require (1.73.1); no dependencies were changed. Python syntax,
shell syntax, and Git whitespace checks passed.

The tests cover identical-state early requests, fusion, slow inference with a
single latest pending observation, duplicate/out-of-order requests, reconnect,
late responses after stop/reset, concurrent stop/arrival, and timing records.

Deployment verified on 2026-10-05: `async_policy.py`, `a1z_policy_server.py`,
`a1z_client.py`, `eval.sh`, and `test_async_policy.py` installed in
`/home/dimos/a1z-experiments/2026-10-02`; launcher updated at
`/home/dimos/a1z-infer/eval.sh`. All 10 tests passed again from deployed files.
No active eval session existed before deployment. Previous files, launcher,
server log, and deployed hashes are preserved in
`/home/dimos/a1z-experiments/2026-10-02/deployment-backups/async-20261005-165109`.
Restoring local Git changes alone does not roll back the deployed Jetson files.

Started tmux `eval` with ACT `060000`, stiffness 1.5, ki 2.0, integral maximum
3 degrees. Verified listener on 127.0.0.1:8080 and server trace
`server-runs/20261005-165248-667053-async.jsonl` with
`observation_filter: fresh_timestep_v1`. Client is waiting at the onsite READY
prompt, before robot.connect(). No READY or motion command was sent. Model
loading and physical behavior are not yet verified in this deployed session.

Keep results with the previous runner as a separate baseline. Do not silently
combine success rates across the change. New files are untracked until added;
include them when stashing this change.
