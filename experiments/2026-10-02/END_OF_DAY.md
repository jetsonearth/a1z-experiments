# 2026-10-02 收工交接

## 今天已验证
- 厂家默认重力辅助可用；用户最新手感反馈可以接受，未继续修改重力或阻尼。
- 连续采集、独立 BEGIN/END、夹爪命令与实测反馈已实现。最新键位：空格切换夹爪、R 开始/结束；Kokoro Heart 英文提示音。
- 702 帧/14.17 秒旧示范的视频已在 Jetson 恢复；954 帧/19.39 秒新示范包含完整夹爪反馈，6.98 秒闭合、12.72 秒张开。两条均 finalized，camera success。
- 相机边采边编码验证：15 秒452帧，停止整理1.773秒。Mac 视频同步可暂停并续传；完整视频仍在 Jetson，Mac 副本未完整，不能声称已备份视频。
- ACT 只完成旧诊断数据 CPU 单步训练、存盘与重新加载推理。没有正式 USB-C policy，没有实机 policy eval。

## 路线纠偏
原实验计划优先评估 HIL-SERL 式少量示范+在线交互+人工纠正，尚未最终选定算法。ACT 是链路测试，不是已选主线，也不是 HIL-SERL 前置要求。π 微调、DAgger、HIL-SERL 在线训练均未接入。用户最新问 ACT 是否值得继续，尚未决定；明天先回答这个选择，再投入正式训练。不要默认继续采10-20条训ACT。

## USB 支线（暂停）
Mac ↔ Jetson 唯一 Debug/Device USB-C 口当前恢复 REC OFF 串口。REC 不是电源开关。ON 下未完成 NVIDIA USB gadget 枚举，没有测得吞吐量。当前串口有效负载理论上限约8.64kB/s，录像实测平均约2.06Mbps，串口不足实时视频。
曾临时设 usb2-0 role=device 并运行 NVIDIA state-change handler，l4tbr0出现192.168.55.1，但Mac仍未连通。未改启动文件、未刷机；Jetson 保持开机，临时 USB device role 和网络服务仍可能存在；下次重启后需重新检查。不要将USB理论带宽等同实测连通，不要继续盲试开关。

## 停机
用户已确认臂体支撑、夹爪清空、REC OFF。现场检查无collect_task/vendor_control/teach/camera_session/ffmpeg进程，controller.lock可独占。SDK stop失能执行结果见 diagnostics/end-of-day-shutdown-console.txt；不调用start或home。用户最后明确要求 Jetson 保持开机，只关闭 A1Z。未发送任何 Jetson 关机命令。SDK stop 已成功返回，CAN1 已置 DOWN；没有独立逐电机失能反馈证明。A1Z物理电源由用户手动关闭。

## 明天最先做
1. 明确本轮实验问题与路线：ACT基线是否有信息价值，还是直接完成HIL-SERL可行性适配；不再把链路测试冒充路线决定。
2. 上电前REC OFF；恢复Jetson连接、按厂家设置恢复CAN1（USB-CAN插拔后接口可能DOWN），不自动使能机械臂。
3. 核对任务视野与示范真正在做什么。HIL-SERL尚缺环境step/observation接口、成功奖励、人工接管切换、复位和在线训练整合。
4. Vast机器尚未收到SSH信息；不宣称训练已开始。实时预览和关机UI留后。

## 最终保存偏好
用户要求视频留在 Jetson，Mac 空间紧张。dashboard 默认仅同步关节/夹爪 JSON；自动视频下载已关闭，只有显式设置 A1Z_STUDIO_FETCH_VIDEO=1 才会传视频。现有原件与已传片段均保留，未进行删除。
