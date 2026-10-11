# SpaceMouse 遥操准备

## 2026-10-08 后续准备：本地代码已实现，尚未部署

- 接上网线后一度实测 SSH `jetson-direct` 成功，Mac dashboard `online:true`。随后 Mac 的 en14 和 `/dev/cu.usbserial-10` 同时消失，当前无法部署；不要把这次成功连通视为现在仍在线。
- Jetson 上已给 dimos 增加 `/dev/hidraw0` 的只读 ACL；这是临时设备权限。已读取真实 HID 描述符：Compact 分别用 report 1/2 的三个 int16 表示平移/旋转，report 3 表示两个按钮。90 秒只读监视启动后连接中断，尚未收到可核对的操作者动作报告。
- 已向旧评测的 READY 提示输入 `no` + Enter 请求取消。随后连接中断，取消结果尚未复核；未发送 READY、未使能或控制机械臂。
- 本地新增 `experiments/2026-10-02/spacemouse_teleop.py` 和 8 项离线测试。读取 HID 与电机执行分开；`--execute` 等待 READY 后才打开 CAN。使用当前 SDK 的默认 PD/重力和已有厂家保护；不控制夹爪、不归零、不加积分。
- 交互：左键按住移动，松开保持实测姿态；右键或 S 暂停，G 恢复后须释放并重新按左键；R 开关旋转；先支撑手臂再连按两次 Q 退出失能。默认只有平移，满幅输入灵敏度 25 mm/s，旋转开启后 0.25 rad/s；灵敏度是输入映射，不是实测运动上限。坐标映射为 raw x,-y,-z 与 rx,-ry,-rz 对应机械臂基座轴；实际手感方向尚待逐轴核对。
- 输入过期、断连、循环延迟和逆运动学越过 SDK 限制时不再发送移动目标；断连后须重启本轮。正常持有默认位置刚度，退出/SDK 故障会失能，现场须准备支撑。
- 本地 console 已加入 SpaceMouse 模式、原位保持的 READY 说明、暂停/恢复/旋转开关和六轴显示。Jetson 后端尚未同步，不要据本地前端按钮认为功能已可用。
- 验证：8 项解析/按钮/过期输入/关节限制/READY 隔离测试通过，Python/JS 语法检查通过。Jetson 运动学依赖可导入；新算法尚未在真实 URDF 上离线验证、未做真机试动。

恢复连接后的顺序：核对旧会话已取消和控制锁 → 备份并部署新脚本及 console → 安装精确匹配 `256f:c635` 的只读 udev 规则 → 输入监视与真实模型离线检查 → 页面等待现场 READY。前沿阅读、训练模型和旧实验数据未改。


author: ai · generated_by: Codex · owner_endorsement: unconfirmed

核查时间：2026-10-08 20:55–20:57 Asia/Taipei。状态：设备已识别，遥操尚未实现或验证。

## 本次直接读取的设备状态

- Mac `/dev/cu.usbserial-10` 可访问 Jetson `jetson01` 的空闲 shell。SSH `jetson`、`jetson-direct` 均超时；Jetson 有线网络接口显示 DOWN。
- Jetson `lsusb`：`256f:c635 3Dconnexion SpaceMouse Compact`；HID 为 `/dev/hidraw0`，权限 `crw------- root root`。已确认枚举，尚未读取六轴或按键报告。
- USB-CAN `a8fa:8598`；`can1` UP、1 Mbps、CAN 状态 ERROR-ACTIVE（正常可工作的 CAN 错误状态名称，不表示本次发现总线故障）。没有查询电机反馈，不能由接口 UP 推断电机就绪。
- D405 `8086:0b5b` 已枚举，未启动相机流。
- Jetson SDK `/home/dimos/GALAXEA-A1Z` 为 `e931ecd`，含 `a1z/robots/kinematics.py`。
- tmux `eval:1` 的旧 `a1z_client.py --hil --epoch 3` 会话停在“现场就绪，输入 READY”，未见 `robot_proc` 进程；本次没有向该会话发送输入。
- 检索 Jetson 实验目录未发现以 space 或 teleop 命名的文件；当前没有确认可直接运行的 SpaceMouse 入口。

## 接下来

1. 给操作用户配置仅针对本设备的 HID 读取权限，读取六轴与按键，确认中位、方向及断连行为。
2. 接入增量末端位姿 → 逆运动学 → 厂家关节控制；明确坐标系、操控灵敏度、按住控制/松开保持和退出行为。沿用厂家参数及保护，不移植旧 RL 插入实验的任务限制。
3. 离线检查方向、逆解失败、断连和退出处理。首次机械臂启动须由现场操作者明确就绪；从当前姿态初始化，不自动回零。
4. 先空载验证平移，再验证旋转；夹爪操作单独验证。

本次仅检查连接、系统和进程状态，未使能电机、未发送 READY、未执行运动，也未修改 Jetson 配置。
