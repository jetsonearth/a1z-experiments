# A1Z Console

Design: Nothing design language (Space Grotesk / Space Mono / Doto from Google Fonts, system fallbacks offline; OLED-black dark mode, off-white light mode via prefers-color-scheme). The older serial-based collection studio (`studio/`, `Start Studio.command`) is unchanged.

Web front end for evaluation, HG-DAgger collection and control-layer tests. Runs on the Jetson, opened on the Mac.

**Start:** double-click `A1Z Console.command` in the repo root. It starts `console/server.py` on the Jetson if needed (inference venv, `127.0.0.1:8780` only), opens an SSH tunnel and the browser.

**Pages:** overview · run console (live wrist camera, start/READY/trial keys, labels, `i` reference pose, client output, temperatures) · runs (all sessions, per-trial error vs the hand-inserted reference, video, relabel / exclude) · report (model × stiffness × integral, rates with 95% Wilson intervals, median error) · models · data · hardware (gripper / joint temperatures).

**How it drives the robot:** it creates the same tmux session as `eval.sh` (`A1Z_NO_ATTACH=1`) and types keys into the client window; robot_proc.py's safety layer and all files are unchanged. `ssh -t jetson-direct tmux attach -t eval` still works. READY is only sent when the operator clicks it while the client is waiting for it. The gripper over-temperature lock (≥ 88 °C coil) is a page-side guard on the start buttons.

**Writes:** only `operator_label` / `excluded` edits in trial JSON, recorded in `console_edits` with `edited_via: console`.

Verified 2026-10-05 without motors: all pages against real session data, video playback with seeking, starting an eval session up to the READY prompt, cancel, close. Not yet verified with the arm enabled.
