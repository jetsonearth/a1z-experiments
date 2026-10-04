# A1Z Capture

本地轻量采集台，深色 Nothing 风格，使用 Google Fonts 的 Space Grotesk / Space Mono（离线时系统字体回退）。直接读取实验目录现有 `episodes/*-teaching.json`，不复制数据集基础设施。

## 启动

```sh
python3 Robotics/Experiments/a1z-2026-10-02/studio/server.py
```

打开 http://127.0.0.1:8769 。默认仅回看：状态、历史记录、播放/暂停、拖动时间轴、0.25–2 倍速、六关节读数、单关节曲线、关节姿态示意和原始 JSON 下载。姿态来自记录的关节角与现有 kinematics.json，不是视频，不驱动机械臂。当前已导入示范没有相机视频。

## 硬件接入边界

`A1Z_STUDIO_HARDWARE=1` 才允许连接、开始、停止、远端同步四个 POST 操作。启用前需要由负责硬件的操作者检查现有 `mac_console.py` / `teach.py` 路径、串口设备、机械臂支撑和互斥状态。UI 本次验证没有运行这些操作，也没有验证物理录制链路；不要将页面可回看视为硬件采集已验证。

页面开始按钮复用旧 recorder 的动作触发机制，不添加倒计时。带相机元数据的示范已接入录后视频传输与时间轴回放，仍待首条真实联合示范验证。服务器只监听 127.0.0.1；POST 需要页面临时 token，并检查 Host / Origin。

## API

- `GET /api/status`：`state`、`connected`、`device`、`message`、`frames`、`elapsed`、`last_episode`、`camera`、`hardware_enabled`、`log`。
- `GET /api/episodes`：本地已保存示范索引。
- `GET /api/episode/{filename}`：已有完整轨迹 JSON。
- `POST /api/connect`, `/api/start`, `/api/stop`, `/api/sync`：请求头 `X-A1Z-Token` 取自页面 meta。默认禁用，返回 409。

没有自动访问串口、自动启动录制或电机回放操作。相机字段 `not_recorded` 表示未接入当前采集链路，不代表相机未连接。

## 本次验证（2026-10-02）

- Python / JavaScript 语法检查通过。
- 浏览器真实加载首条示范：551 帧、11.2 秒、49.2 Hz。
- 播放后时间前进；暂停后拖到 5.6 秒，读数与姿态更新；切换 J3 显示对应角度曲线。
- 硬件按钮默认禁用；后端拒绝未启用的硬件操作。

联合示范视频和完整物理采集链路尚未经过此 UI 的端到端验证。独立 D405 测试视频与旧轨迹没有对应关系。

## 后续接入更新

界面遵循 Nothing 设计规范，录制优先：大号状态（录制中为红色 + Doto 计时）、下一步按钮与 R／空格／X 快捷键；进度（目标默认 30，可改，存 localStorage）+ 分段进度条；电机温度（最热关节 + J1–J6 分段条，85°C 急停线，>3 s 无 TASK_TEMP 显示 STALE）与 ENABLED 时长；侧边「本轮录制」实时列表含每条 QC 标记（TASK_QC）、排除状态（X / `/api/discard` → Jetson `DISCARD`，仅 support 阶段）。本机历史与回看默认折叠。语音提示始终开启。

带 `metadata.camera` 的示范会在控制结束后传输视频。数据集路径限制在本实验远端 datasets 下，唯一 MP4 与 raw-camera-timestamps.json 每次 64 KB 分块传回，最大单文件 128 MB，SHA256 校验后原子发布。原始 teaching.json 不改动；伴随文件置于 `<stem>-media/`。传输失败保留远端文件与已保存关节记录，点击同步可以重试。

已有本地 MP4 和时间戳时，使用 30 fps 视频帧序号映射 `host_monotonic_s` 对齐关节采样；仅当时间区间重叠时启用，表示主机接收时间对齐，不等于相机曝光硬同步。没有可用时间戳时视频独立回放并明确提示。联合示范仍需第一次真实录制后验证。

串口 RPC 不再发送 Ctrl-C 或回车来确认空闲，只用 Ctrl-L 重绘并匹配 Jetson shell 提示符；无法确认则拒绝发送命令。后台任务在清理结束前持续互斥，中文提示使用增量 UTF-8 解码，Stop 与自动应答在同一锁内串行处理。媒体传输测试使用离线替身，没有访问硬件。

## 可选 LeRobot 回看

录制仍以原始记录为默认。页面不会启动 LeRobot 导出，也不表示已经选择训练或部署框架。仅当记录包含 `metadata.lerobot_dataset` 且 `success === true`、`validated === true` 时，才优先传输已验证导出的回放文件。未导出继续显示原始记录，不被标为失败。

可选文件为统一视频、frame-timestamps.json、由真实 Parquet 观测导出的 playback.json；meta/info.json 与 alignment-report.json 为辅助检查文件。它们与原始相机视频使用不同本地文件名，避免传输中途混用。观测帧数、六关节值、时间戳及 30 Hz 网格通过检查后，API 可用派生 samples 回放，显示 LeRobot 30 Hz 对齐观测；原始教学 JSON 字节和原始下载均保持不变。导出或传输失败保留原因，并尽可能回看原始相机视频。

无硬件测试：`python3 studio/test_optional_media.py` 与 `python3 studio/test_lifecycle.py`（从实验目录运行）。

## 当前采集流程（厂家默认，2026-10-02）

以上旧的动录静停说明已被本节替代。以仓库 `experiments/2026-10-02/studio/server.py` 启动，硬件权限仍由 `A1Z_STUDIO_HARDWARE=1` 显式启用。

现有 dashboard 通过 `mac_console.py` 的 `A1Z_MODE=task` 调用 `collect_task.py`：

1. 连接、准备采集：不启用电机。
2. 新出现的「已支撑，启用重力辅助」按钮：现场操作者确认后才发送 READY，进入厂家重力辅助模式。
3. 手动到任务起点，点击「开始录制」。
4. 完成任务，点击「结束录制」：只结束关节记录，重力辅助保持启用。
5. 手动复位并放稳支撑，点击「已支撑，失能并保存」；关闭控制、整理视频、传回数据。

没有动录静停、自动超时失能或浏览器关闭触发失能。没有添加运动速度限制、起点角度门槛或重力缩放。自动 reset 和实机 replay 尚未接入；下方播放按钮仍然仅屏幕回看。

新增 POST `/api/ready`、`/api/begin`、`/api/disable`；各操作检查当前流程状态。旧 `/api/stop` 现在发送录制结束回车，不发送 Ctrl-C。离线测试验证状态顺序、重复点击、录制结束不失能、进程异常不发送额外中断。物理采集仍需操作者完成首次验证。


## 2026-10-03 自动准备

运行根目录 `Start Studio.command` 后，控制台自动检查串口、采集程序、控制锁、D405 USB 枚举、可写存储、CAN 状态，再进入现场确认。无需先点连接、再点准备。重复打开启动器复用正在运行的服务，不替换正在采集的进程。

- CAN 只在接口 DOWN 且能独占 controller.lock 时恢复厂家 can1 / 1 Mbps；已 UP 的接口不重置。控制锁繁忙、配置不符、相机缺失时显示具体失败原因。
- 所有串口 RPC 和采集入口共用 serial_session 的提示符判断及短命令限速。只在看到确切空闲 shell 后初始化 Bash；不向未知前台程序发送回车/中断。
- 不自动发送 READY；现场确认仍由操作者点击。检查表中的相机“已连接”仅表示 USB 枚举，图像帧和关节反馈在启动环节进一步检查。
- 电源或 USB-C 未接入时等待最多三分钟。错误后可点“重新准备”。登录鉴权没有绕过，也不保存密码。
- 当前 Mac 仅枚举 CP2102N USB-UART，未枚举 Jetson USB 网卡；Jetson 以太网无 carrier、无活动 Wi-Fi。SSH 服务虽 active，但尚无可达链路，不能声称已实现冷启动免登录。需接通受支持的网络后才能配置 SSH 密钥。
- 离线验证：`A1Z_STUDIO_FETCH_VIDEO=1 python3 -B -m unittest discover -s studio -p 'test_*.py'`。该环境变量仅为视频功能测试，不改变正常启动默认；视频继续保留在 Jetson。
