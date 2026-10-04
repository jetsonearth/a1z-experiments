# HG-DAgger for the USB insertion

Reference: Kelly, Sidrane, Driggs-Campbell, Kochenderfer, *HG-DAgger: Interactive Imitation Learning with Human Experts*, ICRA 2019 ([arXiv 1810.02890](https://arxiv.org/abs/1810.02890)). Infra added 2026-10-04; no HIL session has been run yet.

## What the paper prescribes, and how it maps here

| Paper (Algorithm 1) | Here |
|---|---|
| D initialised with the behavioral-cloning set D_BC; the BC policy is the first novice | D_BC = `datasets/act-v2` (45 demos, 9,087 frames); novice 1 = `runs/act-v2/060000` |
| Novice drives; the human is the gating function g and takes over when the state leaves the safe set | Policy drives; **Space** = take over (arm floats in vendor gravity assist and you drag it) |
| Human keeps uninterrupted, exclusive control until they manually hand back | Kinesthetic control in float mode; **Space** = hand back (queued chunks dropped, re-engaged at the current pose) |
| Only expert-controlled samples are labelled: D_i = {(o_t, π_H(x_t)) \| g(x_t)=1} | `export_hil.py` exports only the operator segments; policy segments are never labels |
| D ← D ∪ D_i, then train the next novice on D | `export_hil.py --aggregate` merges with the previous D; `pipeline/train_hil.sh` trains |
| Several epochs; each adds ~20% of the BC label count (10,000 BC, then 5 × 2,000) | Suggested: ~1,800 frames/epoch (about 60 s of operator control, e.g. 25–30 interventions); 3–5 epochs |
| Test time: the novice runs with no human | `eval.sh` (no `--hil`) with the 1/2/3 outcome labels |
| Doubt d_N(o) = ‖diag(Cov of an ensemble)‖₂ at each takeover; τ = mean of the last 25% of those | Not implemented (needs an ensemble of ACTs). The takeover observation is saved per segment (`takeover_camera_index`), so doubt and τ can be computed offline later. In the paper τ is only used for evaluation, never for gating. |

## Deliberate choices beyond the paper (each one reversible)

- **One episode per intervention.** ACT predicts 100-step chunks. A separate episode makes LeRobot pad and mask the chunk at the handback, so no label ever contains the policy's own motion after the handback.
- **Leading idle frames are trimmed** (`--idle-rad 0.003`, at most 1.5 s). In the paper the takeover is itself an action (turning the wheel). With kinesthetic takeover, the first frames are reaction time and would teach "stay still" in exactly the failure state. Pass `--idle-rad 0` to keep everything.
- **Discard key `d`** after a trial, for a fumbled correction. The paper keeps every expert label. Raw files are kept either way.
- **Fine-tune vs. retrain.** The paper only says "train on D". `train_hil.sh` defaults to fine-tuning from the previous novice (20k steps); `MODE=scratch` retrains with the act-v2 recipe (60k steps). Run both once in epoch 1 and compare them on the robot.
- **Data fidelity.** Labels follow the demonstrations exactly: same 30 Hz alignment (`unified_dataset.alignment_plan`), state = measured joints + measured gripper, action = next aligned joints + commanded gripper, same camera streams (color + depth on), same H.264 encoder settings, same task string.

## One epoch, step by step

1. **Jetson.** Run `ssh -t jetson-direct 'CKPT=<novice i>/pretrained_model ~/a1z-infer/eval.sh --hil --epoch <i>'`. Press `g`, let the policy drive, and press Space when it is going wrong. Move straight away (no pausing), and use the same recovery every time: lift 1–2 mm, slide, drop in, press to USB. Then either press Space to hand back, or finish it yourself; USB enumeration ends the trial. Press `x` to end a trial early, and `d` afterwards to discard it.
2. **Copy** `hil-runs/<session>` from the Jetson to `~/a1z-data/usb-insert-20261003/hil-runs/`.
3. **Mac.** Check what would be exported with `python export_hil.py <sessions> --output /tmp/x --dry-run`, then build the datasets:
   `python export_hil.py <sessions> --output datasets/hil-e<i> --aggregate <D_{i-1}> --aggregate-output datasets/agg-e<i>`
   For epoch 1, D_0 is `datasets/act-v2`.
4. **GPU.** Run `bash pipeline/train_hil.sh /workspace/data/agg-e<i> /workspace/runs/hil-e<i> /workspace/init/pretrained_model`.
5. **Jetson.** Copy the new checkpoint over and set `pretrained_backbone_weights` to null in `config.json`, as for act-v2. Evaluate 10–20 trials without `--hil`, then continue with epoch i+1.

**Progress metrics:** intervention rate (interventions per trial, operator seconds per trial), the autonomous two-level success (found port / inserted), and frames added per epoch.

## Tests

`test_export_hil.py` (export and labels on a synthetic session using the real encoder) and `test_hil_client.py` (key flow with a fake robot) both pass. On 2026-10-04, merging with act-v2 (45 → 47 episodes) and a 3-step `lerobot-train --policy.path` fine-tune from 060000 were also checked on the Mac. The normalizer stats are replaced by the aggregate's.
