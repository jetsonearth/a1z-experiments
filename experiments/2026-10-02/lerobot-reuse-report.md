# A1Z：LeRobot 可以复用什么

调研日期：2026-10-02。结论依据官方文档与源码静态检查；本报告没有安装、连接或控制机械臂，也没有把源码支持写成 A1Z 实机验证。

**值得复用，而且不必从头造数据平台。保留目前已跑通的 A1Z 拖拽控制与运动触发，只做薄界面；数据集、视频编码、离线回看和后续训练优先接 LeRobot。** 今天先补 D405 与时间戳，不为换框架暂停硬件实验。

## 可复用边界

| 需求 | 已有能力 | 本实验还要做什么 |
|---|---|---|
| 数据集与视频存储 | LeRobotDataset：Parquet 状态数据、MP4 视频、episode/task 元数据、统计、训练加载 | 单臂字段映射、真实时间戳保留、采样对齐、动作语义 |
| 录制流程 | 标准录制、保存/重录、定时 episode、视频编码 | 保留“按开始立即待录，动了开始，静止两秒结束”和预录缓冲 |
| D405 | RealSenseCamera 使用 pyrealsense2，支持 RGB/depth；源码明确处理 D405 特性 | 核对安装版本、实际支持的模式、出图和帧龄；不假定现成硬件同步 |
| 看录过的数据 | 本地 dataset-viz 可同步看图像、state/action；有 Rerun 和 Foxglove 路径，也有 Hub 网页查看器 | 首条带图像数据导出后直接试，不必先写完整查看器 |
| 简单前端 | 官方 LeLab 有录制、训练与部署 GUI | 文档明确目前仅 SO-ARM101；不是 A1Z 即插即用替代品 |
| 机器人适配 | Galaxea 的确发布了 **A1Z** 适配，不是 R1/A1/Unitree G1 的替代推断 | 当前为双臂 14 维、3 相机、位置控制，需改成现有单臂控制边界 |
| 在线 RL / 人工接管 | LeRobot 已有 HIL-SERL 工作流、SAC actor/learner、奖励分类器与干预处理 | A1Z 控制、复位、接管、奖励与算力仍须逐项验证 |

依据：[Dataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)、[数据回看工具](https://huggingface.co/docs/lerobot/using_dataset_tools)、[LeLab 兼容范围](https://huggingface.co/docs/lerobot/lelab)、[HIL-SERL 官方指南](https://huggingface.co/docs/lerobot/hilserl)。

## A1Z 官方适配确实存在，但不能直接启动

厂商仓库 [`userguide-galaxea/a1z-lerobot`](https://github.com/userguide-galaxea/a1z-lerobot) 当前检查提交 `4cdfd682a1f946dbb71d8952390dbef1c1f21bdc`。README 主要面向双臂训练与 rollout/RTC；另有 HDF5 → LeRobot 转换器，可参考字段与图像写入方式。

实际源码发现：

- `connect()` 启动双臂控制循环；底层配置为 `zero_gravity_mode=False`、`with_gripper=True`，包装器说明 SDK 启动包含夹爪使能和归位。
- **`disconnect()` 主动打开夹爪，然后尝试 `move_to_home()`，底层目标是六关节零位。** 这与本实验“退出不回零、不主动操作夹爪”的要求冲突。
- `send_action()` 的 EMA 与每步限幅可作参考，不能替代已验证的反馈超时、发送节奏和故障保护。

源码：[连接/退出/动作接口](https://github.com/userguide-galaxea/a1z-lerobot/blob/4cdfd682a1f946dbb71d8952390dbef1c1f21bdc/a1z_lerobot/robots/a1z_follower/a1z_follower.py)、[单臂包装器](https://github.com/userguide-galaxea/a1z-lerobot/blob/4cdfd682a1f946dbb71d8952390dbef1c1f21bdc/a1z_lerobot/robots/a1z_follower/hardware/a1z.py)。

因此现在只参考适配结构，不用其 `connect/disconnect/rollout` 接管硬件。真正迁移时需要单臂配置、显式控制模式、无自动归位生命周期，以及保留当前保护逻辑；这不是改一个 robot 名字就完成。

## 数据同步与 action 是真正需要补的部分

上游 LeRobot 源码检查基准：`6e1fa4faf2a42927d463591aebaa0f62c2e654b7`。在线文档和本机安装版可能不同，应按安装版 API 落地。

1. **LeRobot 的标准 `timestamp` 是 `frame_index / fps`。** 当前 writer 自动计算它；它不是 D405 曝光时间，也不是现有约 49 Hz 关节数据的原始时间。需保留原始 `joint_monotonic`、相机 frame number、device timestamp/domain 和 host receipt monotonic；导出时另建固定采样网格，记录选帧偏差、重复帧与缺帧，不能将名义 fps 当同步证明。[DatasetWriter](https://github.com/huggingface/lerobot/blob/6e1fa4faf2a42927d463591aebaa0f62c2e654b7/src/lerobot/datasets/dataset_writer.py)
2. RealSenseCamera 当前代码在取图后记主机 `perf_counter`；`read_latest()` 实际只返回图像，尽管顶部示例还写成 `(image, timestamp)`。D405 没有独立 RGB Camera sensor，手动 RGB exposure/gain 等接口也存在限制。直接保留 pyrealsense2 原始时序更利于这轮核验。[相机实现](https://github.com/huggingface/lerobot/blob/6e1fa4faf2a42927d463591aebaa0f62c2e654b7/src/lerobot/cameras/realsense/camera_realsense.py)
3. Stock `lerobot-record` 的循环在没有 teleoperator action 时跳过后面的 dataset 写入；所检查入口未提供我们这种运动阈值开始/静止结束机制。直接运行它并不等于录到 kinesthetic demonstration。可在现有触发循环后调用 Dataset API，而不是伪造一个会发送位置指令的 teleoperator。[录制循环](https://github.com/huggingface/lerobot/blob/6e1fa4faf2a42927d463591aebaa0f62c2e654b7/src/lerobot/scripts/lerobot_record.py)
4. 当前 551 帧是关节反馈与辅助托重 MIT 指令；**这些 MIT 指令不是人意图的关节目标，更不是末端增量 action。** 不把 q 原样贴上 action 标签就称“可训练 HIL-SERL”。若后续由示范轨迹构造 position/delta action，要显式定义时移、坐标、单位和推导规则；在线 RL 另需 observation/action/reward/next observation/termination 与干预事件。

厂商 HDF5 转换器采用相机最近邻选帧，但不会自动为我们验证时钟域、间隔、帧龄或动作语义；可以借代码结构，不能把转换成功当同步通过。[厂商转换器](https://github.com/userguide-galaxea/a1z-lerobot/blob/4cdfd682a1f946dbb71d8952390dbef1c1f21bdc/a1z_lerobot/tools/hdf5_to_lerobot.py)

## 回看与真机 replay 分开

页面的播放按钮只读文件。数据集 visualization 能看视频、状态和动作；是否有适配 A1Z 的三维模型另行确认，不是查看同步图像的前提。`lerobot-replay` 则读取 action 并调用 `robot.send_action()`，确实会驱动硬件，不能拿来实现网页回看。[真机 replay 源码](https://github.com/huggingface/lerobot/blob/6e1fa4faf2a42927d463591aebaa0f62c2e654b7/src/lerobot/scripts/lerobot_replay.py)

## 建议执行顺序

1. **现在：** 主进程继续 D405 只读探测，录数秒视频+时间戳；不等待 UI 或框架迁移。
2. **下一条示范：** 沿用已验证触发与保护，相机独立采集，阻塞/编码不进入控制环；保存原始数据与退出原因，核对覆盖整个预录和动作区间。
3. **保存后：** 用 LeRobotDataset API 做本地导出（`create → add_frame → save_episode → finalize`），不默认上传。保留原始 JSON/视频/时间戳；先做可检查的 observation 数据，动作定义完成后才称训练数据。
4. **UI：** 仅做开始待录、停止、状态、episode 列表与本地回看。不要继续自建训练平台、视频编码系统或通用数据集格式；先复用 LeRobot 回看工具验证一条记录。
5. **之后：** 实机低速空中执行通过，才实现 LeRobot Robot/环境 adapter。LeRobot HIL-SERL 可作为优先候选实现；它是独立工程实现，不能将其配置/结果直接等同于 [Berkeley 原版 HIL-SERL](https://github.com/rail-berkeley/hil-serl)，也不必为用它先训 ACT。

今天的完成标准仍是：**一条可检查的 D405 + 六关节同步记录；框架选择不能替代这条实证。**

## 追加：按 Jetson 已安装 LeRobot 0.4.4 落地

主进程已通过包元数据确认 Jetson 安装版本为 **0.4.4**。下面检查的是 `v0.4.4` 标签源码，非上面的 main；只做 API 核验，尚未由本调研进程执行导出。

### 最小方案：SDK 取帧，LeRobot 写图与编码

暂用已验证能出图的 pyrealsense2 pipeline 取得 RGB 数组和原始 frame metadata；**不切换到 LeRobot 的 RealSenseCamera 来获取同步证据**。0.4.4 相机包装器把 SDK frame 转为数组，内部只记取图后的 `perf_counter`，公共读图 API 不返回 frame number、device timestamp/domain；顶部 `read_latest()` 二元返回示例与实际实现不符。[0.4.4 相机源码](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/cameras/realsense/camera_realsense.py)

不需要 fake action：`validate_frame` 要求 schema 中自定义 features 加 `task`，没有强制 `action` 或 `observation.state`。先做 RGB-only probe，再添加六维 `observation.state`，称为 observation-only 检查数据。[0.4.4 验证器](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/utils.py)

API 配置示意（不是已运行的 exporter）：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

features = {
    "observation.images.wrist": {
        "dtype": "video", "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    # 加入关节时声明：
    # "observation.state": {"dtype": "float32", "shape": (6,),
    #                       "names": ["j1", "j2", "j3", "j4", "j5", "j6"]},
}
ds = LeRobotDataset.create(
    repo_id="local/a1z_observation_probe",
    root=NEW_NONEXISTENT_DIRECTORY,
    robot_type="a1z_single_observation_only",
    fps=30, features=features, use_videos=True,
    image_writer_processes=0, image_writer_threads=2,
    streaming_encoding=False, batch_encoding_size=1,
    vcodec="h264", encoder_threads=2,
)
# 每帧传入新的字典；输入为 RGB uint8 HWC 数组。
ds.add_frame({"observation.images.wrist": rgb_array.copy(),
              "task": "camera-only observation probe"})
# 所有帧加入且采集结束后：
ds.save_episode()
ds.finalize()
ds.stop_image_writer()
```

这个模式复用 LeRobot 临时 PNG writer，在 `save_episode()` 才编码 MP4；真实采集与编码分开。实际已验证 `vcodec="h264"` 可用；直接传 `libx264` 被 0.4.4 参数验证器拒绝。成功小样 107 帧，编码后解码仍为 107 帧。`root` 必须是尚不存在的目录。完整脚本需 `try/finally` 清理相机和 writer，且异常 episode 不标成通过；上面只说明 API。[0.4.4 Dataset API](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/lerobot_dataset.py)

**时间戳版本陷阱：** 0.4.4 `add_frame()` 内虽然有 `frame.pop("timestamp")` 分支，但它之前的 `validate_frame()` 将 DEFAULT_FEATURES 排除，传入 `timestamp` 会被当作额外字段拒绝。不要依赖该分支；仍使用标准 index/fps 作为派生视频时间，原始 host/device 时间写独立自定义 feature 或 sidecar。后续转换已有约 49 Hz JSON 到 30 Hz 视频网格，必须明示插值/选帧方法和误差，不能把原始 551 行当 30 fps 直接写入而改变时长。

### `.db3` 不是 D405 特殊格式；当前本机 native 录制尚未跑通

新版 librealsense 的官方原生录制格式为 ROS2 SQLite `.db3`，文件后缀必须 `.db3`；旧 `.bag` 是 legacy ROS1。SDK 自带所需存储组件，不要求为此安装 ROS2。[官方格式说明](https://github.com/realsenseai/librealsense/blob/master/doc/record-and-playback.md)

但主进程实测反馈：native RGB+depth `.db3` 录制在 16 帧左右超时，而相同 profile 不启用 native recorder 的对照流完成 107 帧、约 30 Hz。因此当前不能把 native `.db3` 作为已可用方案，尚不能仅凭这个对照归因到具体驱动层。先验证上面的 **SDK 取帧 + LeRobot 图像 writer + 停采后编码**；保留 raw depth 的方案待 RGB 记录路径通过后再定，不把 16-bit depth 强塞进普通 RGB 视频编码。


主进程后续已验证独立相机 worker 的完整启动、健康检查、停止和编码：93 帧、30.03 Hz、无跳帧，H.264 解码 93 帧；尚未运行新的关节＋相机联合示范。
