# A1Z experiment conventions

- User instruction, 2026-10-02: follow the actual manufacturer SDK and examples for control parameters. Do not introduce custom velocity caps, start-angle gates, torque caps, gravity scaling (such as 0.3), or other speculative restrictions. Keep manufacturer protections intact. Any necessary deviation must have concrete evidence and an explanation to the user.
- Prioritize the collect → train → physical evaluation pipeline within the user’s four-hour work session. Do not make full historical trajectory replay, automatic reset, UI polish, or format generalization prerequisites.
- Use this repository as the source of truth. The my-vault experiment directory is a historical copy.
- Preserve raw episodes. Distinguish offline checks, trained checkpoints, and observed physical evaluation.
- The paced CAN transport has observed evidence (joint-6 response loss without pacing); it is a transport fix, not a motion-speed limit.
- Do not send READY or enable motors without the on-site operator’s explicit readiness.
