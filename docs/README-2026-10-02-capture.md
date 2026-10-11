# A1Z 实验项目

A1Z 单臂拖拽示教、D405 视频采集、数据检查和受监督回放实验。2026-10-02 从 my-vault 独立整理；这是后续代码与实验记录的主目录。

**当前状态：关节与视频联合采集已验证；自动接近测试出现前倾后急速回落，实机回放已暂停。保留 `REPLAY_BLOCKED`，不要直接移除后试跑。停止会失能，不保证保持姿态。尚未进行 policy 训练或成功实机轨迹回放。**

## 目录

- `docs/experiment-plan.md`：实验计划与最新进度。
- `experiments/2026-10-02/`：本轮全部代码、测试、采集控制台及调研。
- 其下 `episodes/`：已取回 Mac 的关节记录、视频与检查报告。
- 其下 `replay-results/`：原位保持与失败接近测试日志。
- 其下 `diagnostics/`：相机测试、故障诊断及历史日志。
- 其下 `replay-preparation/`：回放准备文件与历史操作说明；以暂停状态为准。

## 启动本机控制台

```sh
bash ~/a1z-experiments/"Start Studio.command"
```

浏览器打开 http://127.0.0.1:8769/ 。保持终端打开。启动网页本身不会启动电机。

若旧控制台仍占用 8769，先在它的终端 Ctrl+C 退出（仅在没有录制、同步或控制任务时），再启动新目录入口；不要同时运行两套控制台。

## 硬件与数据位置

Mac 项目：`~/a1z-experiments`。Jetson 实验仍位于 `/home/dimos/a1z-experiments/2026-10-02/`，SDK 仍位于 `/home/dimos/GALAXEA-A1Z`，固定提交见 teach.py。原始相机 LeRobot 数据集仍在 Jetson 的 datasets/；本仓库包括已传回 Mac 的视频和时间戳，不是 Jetson 全盘备份。

my-vault 原实验目录保留为迁移时的历史副本，未删除，不再作为后续代码主目录。my-vault 中的计划是知识库副本；后续实验更新以本仓库为准。复制的计划仍含指向个人 vault 的研究背景双链，外部阅读时可能不可跳转。

## 离线检查

```sh
cd ~/a1z-experiments/experiments/2026-10-02
python3 -m unittest test_replay test_feedback_guard test_finalization
```

这些测试不启动硬件，不代表实机控制安全或跟踪质量已通过。Mac 控制台使用 Python 3 标准库；硬件侧依赖既有 Jetson 环境（SDK、NumPy、RealSense、LeRobot 等），当前未提供可重建的完整环境锁定文件。

现有小规模实验数据纳入私有 Git 仓库以便追溯。后续大规模视频数据需独立存储与校验，不适合无限追加到 Git。
