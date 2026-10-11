# HIL-SERL：A1Z 上的完整 USB 插入

撰写：2026-10-09，Claude Code。状态：代码完成，Mac 上 13 项离线测试通过（含分类器训练，以及以分类器为 reward 的 learner 与假机械臂 actor 走 gRPC 的端到端测试）；**还没有上过 Jetson，也没有任何真机结果**。对应 Linear 项目 P-END-200。

> 更新 2026-10-11：已上真机。主结果 HIL-SERL 90/100（`runs/20261011-112045-eval`）；对照 HG-DAgger 42/100、BC 3/30，见仓库根目录 README。下文的「状态」一句是 10-09 写的历史描述。

## 这次做什么

按论文（Luo 等，arXiv 2410.21845）复刻 HIL-SERL：插头预先夹好，每一轮从插口上方一个随机起点出发，策略看腕部和侧面两路画面，以 10 Hz 输出末端 xyz 小位移，直到 reward 判定成功。reward 按论文用图像二值分类器（侧面相机），USB 枚举信号全程记录但不当 reward，用来当裁判：统计分类器的误报漏报，以及真实的插入成功率。人随时可以接管（kinesthetic，按空格让手臂浮动、直接拖），接管数据同时进示范 buffer 和在线 buffer。learner 跑在 Vast 的 GPU 上，Jetson 上只做推理和控制。

和上周 `../2026-10-02/RL-RECOVERY.md` 的区别：那次只练最后一段（起点离插口 2.5–7 mm，接在 ACT 后面），成功标准是进入口部，learner 跑在 Jetson 上、缓存冻结特征。这次是完整插入、USB 判成功、learner 在 GPU 上照常做图像增强，超参按论文。上周的底层部件原样复用：正逆运动学、限力（指令最多比实测低 `press_mm`）、kinesthetic 接管的动作换算、复位、USB 检测、热保护、节拍修复、Jetson 上的 ResNet-10 移植、操作流程（`rl_run.Runner`）。

## 文件

| 文件 | 在哪跑 | 内容 |
| --- | --- | --- |
| `common.py` | 两边 | 同一份 SAC 策略定义（经 `rl_agent.make_policy`）、transition 格式、uint8 图像 buffer |
| `env.py` | Jetson | `FullInsertEnv`：随机悬空起点、两路画面、分类器或 USB 当 reward（每步都记 USB、分类器概率、几何判定）、以插口为中心的边界、贴表面检查夹持、脚本插入测试 |
| `classifier.py` | Jetson | reward 分类器：冻结 ResNet-10 + 两层 MLP，按时间切验证集，报告准确率、误报漏报和与 USB 的一致率 |
| `side_camera.py`、`side_preview.py` | Jetson | 侧面相机（按型号打开，不会和腕部 D405 混）；架设时的实时画面 |
| `actor.py` | Jetson | 六种模式 snap / check / classify / demo / train / eval；连 learner 的 gRPC 客户端 |
| `learner.py` | Vast | gRPC 服务、两个 buffer、SAC 更新、定时推权重、存 checkpoint |
| `hilserl.sh` | Jetson | 在 tmux 里启动 actor |
| `tunnel.sh` | Mac | Jetson → Mac → Vast 的 SSH 隧道 |
| `sync.sh` | Mac | 把代码同步到 Jetson 或 Vast |
| `vast_setup.sh` | Vast | 装 LeRobot 0.4.4 和新版 grpcio，自检 |
| `test_hilserl.py` | Mac | 离线测试 |
| `setup-log.md` | — | 搭建流水账（DoD 第 6 条） |

## 和论文、和 LeRobot 的偏差

- **动作只有平移**，姿态锁在标定时的插入姿态。论文 RAM 插入是 6D twist；LeRobot 的 HIL-SERL 也只控平移。
- **没有力传感器**。观测里用指令与实测位置之差代替接触力（上周的设计），加相对本轮起点的位移和上一步动作，共 9 维。
- **一路腕部相机加一路固定侧面相机（D435i）**。论文 USB 任务是两路腕部加一路侧面。
- **reward 分类器按论文**：人按键标注成功帧（约 200 正、1000 负），分类器只看侧面画面。验证集取每段录制的最后 20%（相邻帧几乎一样，随机切分会高估准确率）。
- **起点范围先保守**：插口周围半径 10 mm、表面上方 4–12 mm。论文 RAM 插入是 ±4 cm。跑通后用 `--start-r` 放大。
- **增益和限速按仓库 2026-10-09 的默认值**：dimos 增益 kp 80/80/80/50/20/20、kd 3/3/3/0.7/0.4/0.4，不加 0.5 rad/s 限速（见仓库 AGENTS.md）。`--stiffness` 只用于回到旧的厂家增益。
- **积分补偿打开**（`--ki 2.0 --ki-max 1.0`，2026-10-09 决定，见 AGENTS.md）：不开积分时同一段示范回放误差 2.8 mm，比每步 1.5 mm 还大，小步执行不了；开了是 0.3–0.5 mm。接触时多出的力受每关节 1° 上限约束。夹爪力矩跟随 `gripper_max_torque.txt`（现为 2.0 N·m，热平衡约 60°C），`--grip-full` 不设。
- **LeRobot 0.4.4 只用了它的部件**：SACPolicy 和损失、DrQ 随机平移、gRPC 服务和传输。它的 `gym_manipulator` 是为舵机总线机械臂写的，接不上 A1Z；它的 actor 不传接管标记（主分支已修）；它的 ReplayBuffer 把图像存成 float32、读写没加锁（主分支已修）。所以 actor 循环、learner 循环和 buffer 是自己写的，更新步骤照 LeRobot 的 `learner.py`。
- **超参**：γ 0.97、初始温度 0.01（LeRobot 默认 0.99 和 1.0）、Adam 3e-4、batch 256（在线、离线各半）、UTD 2、每 2 秒推一次权重、更新次数不超过 500 + 8 × 在线数据量。

## 第一次怎么跑

1. **Mac 同步代码到 Jetson**：`bash sync.sh jetson`。它会列出 Jetson 上 `2026-10-02` 里复用文件和 Mac 的差异，不会覆盖。有差异先停下来看。
2. **标定**（插头夹好之后，换夹持就要重做）：Jetson 上 `~/a1z-infer/eval.sh --calib`，按 1（表面）、2（刚进口）。
3. **侧相机和裁剪框**：A1Z Console → Cameras（07）→ Start preview，边看实时画面边调相机位置；把红色方框拖到插口上，滑条调大小，Save crop 存进 Jetson 上的 `cameras.json`，之后 actor 不写 `--crop-side` 就用它（启动 actor 会自动关掉预览，训练时这一页继续显示 actor 的侧相机画面）。方框里要始终看得到插头、口和一点 hub，右边的 128×128 小图就是分类器看到的。腕部 `--crop` 仍用 `hilserl.sh snap` 定，也可以写进 `cameras.json` 的 `crop`。
4. **检查**：`hilserl.sh check --crop …`。必须看到"检查通过"：跟踪误差、3 次从正上方脚本插入都要 USB 成功并能拔出、两侧贴表面高度对得上、复位无误报。插不到底就调 `--press-mm`（加力）或 `--floor-mm`，再测。
5. **分类器**：`hilserl.sh classify --crop … --crop-side …`，手臂浮动，你拿着插头在口附近移动、插进去、拔出来；插到底时按 s，离开时再按 s。多录压在口旁边、插一半、对准没进去这些难的反例。然后 `~/a1z-infer/run.sh classifier.py train runs/<classify 会话> --out classifier.pt`，看验证集误报率。
6. **示范 20 条**：`hilserl.sh demo --n 20 --clf classifier.pt --crop … --crop-side …`。复位后手臂浮动，你把插头拖进去，分类器判定成功就自动结束。
7. **起 learner**：Mac 上 `bash sync.sh vast root@HOST PORT`；Vast 上 `bash ~/a1z-hilserl/2026-10-09-hilserl/vast_setup.sh`，再按脚本开头那行命令在 tmux 里启动 `learner.py`，看到 `LEARNER_READY`。
8. **开隧道**：Mac 上 `bash tunnel.sh root@HOST PORT`，窗口保持打开。
9. **在线训练**：`hilserl.sh train --demos runs/<示范会话> --clf classifier.pt --crop … --crop-side …`。只在明显走错或卡住时按空格接管，纠正一下就交还；不要一路带到插进去。每 10 轮自动贴表面检查一次夹持。learner 断了：重启 learner（`--resume runs/<名字>/ckpt-last.pt`），actor 加 `--resume runs/<训练会话>` 把已有数据重发一遍。
10. **评测**：`hilserl.sh eval --n 100 --ckpt runs/<训练会话>/actor-last.pt --clf classifier.pt --crop … --crop-side …`。以 USB 为准输出成功率、Wilson 95% 区间、最长连续成功、成功节拍。

## 已验证与未验证

已验证（Mac，假机械臂，2026-10-09）：策略权重在两边逐键一致；transition 格式与接管、示范标记；buffer 路由（示范只进离线，接管两边都进，策略数据只进在线）；新环境的起点分布、USB 成功、贴边下压不误判、以插口为中心的边界、拔不出来时拒绝复位、脚本插入、贴表面检查；learner 与 actor 两个进程走 gRPC 跑 4 轮训练（发示范、更新、回传权重、写日志、存 checkpoint），再用 checkpoint 跑评测。

未验证：Jetson 上的依赖和节拍、Vast 环境、隧道、真机上的一切，包括力预算能否插到底、USB 枚举的实际延迟、手机作为 USB 设备是否稳定、拔出时插头会不会在手指里滑。假机械臂是几十行的软件替身，不是仿真，它跑出来的成功率没有意义。
