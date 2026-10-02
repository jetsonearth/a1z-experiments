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

界面改为紧凑控制台：顶部控制与状态、主视频回放、侧边录制列表；关节姿态与曲线默认折叠。独立相机测试录像是另一个折叠入口，不会被当作旧关节示范的视频。当前不需要实时视频或网络连接。

带 `metadata.camera` 的示范会在控制结束后传输视频。数据集路径限制在本实验远端 datasets 下，唯一 MP4 与 raw-camera-timestamps.json 每次 64 KB 分块传回，最大单文件 128 MB，SHA256 校验后原子发布。原始 teaching.json 不改动；伴随文件置于 `<stem>-media/`。传输失败保留远端文件与已保存关节记录，点击同步可以重试。

已有本地 MP4 和时间戳时，使用 30 fps 视频帧序号映射 `host_monotonic_s` 对齐关节采样；仅当时间区间重叠时启用，表示主机接收时间对齐，不等于相机曝光硬同步。没有可用时间戳时视频独立回放并明确提示。联合示范仍需第一次真实录制后验证。

串口 RPC 不再发送 Ctrl-C 或回车来确认空闲，只用 Ctrl-L 重绘并匹配 Jetson shell 提示符；无法确认则拒绝发送命令。后台任务在清理结束前持续互斥，中文提示使用增量 UTF-8 解码，Stop 与自动应答在同一锁内串行处理。媒体传输测试使用离线替身，没有访问硬件。

## 可选 LeRobot 回看

录制仍以原始记录为默认。页面不会启动 LeRobot 导出，也不表示已经选择训练或部署框架。仅当记录包含 `metadata.lerobot_dataset` 且 `success === true`、`validated === true` 时，才优先传输已验证导出的回放文件。未导出继续显示原始记录，不被标为失败。

可选文件为统一视频、frame-timestamps.json、由真实 Parquet 观测导出的 playback.json；meta/info.json 与 alignment-report.json 为辅助检查文件。它们与原始相机视频使用不同本地文件名，避免传输中途混用。观测帧数、六关节值、时间戳及 30 Hz 网格通过检查后，API 可用派生 samples 回放，显示 LeRobot 30 Hz 对齐观测；原始教学 JSON 字节和原始下载均保持不变。导出或传输失败保留原因，并尽可能回看原始相机视频。

无硬件测试：`python3 studio/test_optional_media.py` 与 `python3 studio/test_lifecycle.py`（从实验目录运行）。
