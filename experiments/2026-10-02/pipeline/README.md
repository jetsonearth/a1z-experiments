# First collect → ACT train → physical eval loop

## Verified on 2026-10-02

- Operator confirmed manufacturer-default position hold was normal, then exited with the arm supported. This does not establish reset, replay, or USB task success.
- Dashboard now uses collect_task.py. Explicit task recording boundaries exclude manual reset; ending recording preserves gravity assistance until supported disable.
- Existing diagnostic episode exported to 101 LeRobot image/state/action rows, reopened successfully. ACT completed one CPU optimizer update, saved weights, and reloaded for finite six-joint inference. This is smoke-only diagnostic data, not a trained USB policy.

## Action convention

30 Hz observations: wrist RGB plus six joint angles in radians; new gripper-enabled demonstrations add measured normalized gripper opening as the seventh state coordinate. The seventh action is the commanded gripper opening (0 closed / 1 open). Missing gripper feedback is rejected rather than replaced with the commanded position. The action label is the next aligned demonstrated joint position, an absolute joint target. It is reconstructed from kinesthetic motion, not the zero-stiffness teaching motor command. Raw data is preserved. Physical tracking and task success require separate evaluation.

## Remaining

1. Collect one real task episode in the dashboard; check operation-area visibility and complete capture. Then collect a first small batch of consistent demonstrations.
2. Export explicitly selected task episodes with export_training.py. Earlier hardware checks are rejected unless --smoke is specified.
3. Train ACT on the user-provided Vast machine using run-cloud.sh, then retrieve the checkpoint. No public Hub upload.
4. Run supervised physical policy evaluation using current camera/state observations and manufacturer action APIs; record all trials, failures and manual resets. The evaluation entry evaluate.py is implemented; physical operation and inference timing remain unverified.

## Commands

Use the project .venv-pipeline locally. Cloud environment: pinned LeRobot 0.4.4 in a CUDA PyTorch image.

```sh
python pipeline/export_training.py <task-episode.json> [more-task-episodes.json] --output <new-dataset-directory>
bash pipeline/run-cloud.sh <absolute-dataset-directory> <new-run-directory>
```

The initial training budget is 2,000 optimizer steps, configurable by STEPS. This is a first-loop engineering budget, not a convergence claim. GPU availability, real task data and physical evaluation remain necessary.

## Evaluation entry

Default invocation is offline and never imports robot/CAN code:

```sh
python pipeline/evaluate.py --checkpoint <run/checkpoint> --dataset <dataset>
```

On the Jetson, live evaluation explicitly requires `--execute`, an operator READY, and a real (not smoke-only) training report alongside the checkpoint. It uses the same RGB/state normalization as training, SDK position targets, an outcome label, JSON target/measurement/timing logs and an observation video. End of trial holds current position; supported disable is separate. Hardware faults still follow SDK shutdown. Video is nominal 30 fps; actual timings are in JSON.

```sh
python pipeline/evaluate.py --checkpoint <run/checkpoint> --execute --seconds 10
```

## Rapid collection (2026-10-02 update)

Start one gravity-assisted session. SDK startup opens the gripper. After operator READY, R begins/ends each demonstration; Space toggles the commanded gripper state. Held keys do not repeat. END keeps gravity assistance and camera capture active; reset manually, then R for the next demonstration. At the end of the batch, support the arm and disable. Audio acknowledges commands received by the remote collector, not measured completion of jaw travel.

Camera streaming was physically validated for 15 seconds: 452 frames, 30.001 Hz, zero skipped frames, 1.773 seconds to finalize. Continuous collector/hotkeys/gripper labels passed offline tests; physical gripper operation and user audio feedback still need the first on-site check. Bulk transfer to Mac uses slow serial and occurs after the batch.
