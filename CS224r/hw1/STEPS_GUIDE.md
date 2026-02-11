# CS224R HW1 步骤指南

按教学文档和代码依赖顺序，每一步只说明**要做什么**和**如何验证**，具体实现由你完成。

---

## 第 0 步：先读代码（不写代码）

**要做的：** 按 PDF 推荐的阅读顺序通读一遍，建立整体数据流概念。

1. `scripts/run_hw1.py`（只读）— 入口、参数、如何调 BCTrainer。
2. `infrastructure/bc_trainer.py` — 训练循环：收集数据 → (DAgger 时) 专家重标 → 训练 agent → 日志。
3. `agents/bc_agent.py`（只读）— agent 的 train/sample 如何调用 actor 和 replay buffer。
4. `policies/MLP_policy.py` — 策略网络接口：get_action / forward / update。
5. `infrastructure/replay_buffer.py` — 数据存什么、如何被采样。
6. `infrastructure/utils.py` — 如何从 env + policy 采轨迹。
7. `infrastructure/pytorch_util.py` — 如何构建 MLP。

**验证：** 能口头说出：BC 时数据从哪来、DAgger 多出的步骤是什么、训练时 batch 从哪取、loss 在哪个文件里算。

---

## 第 1 步：实现 `build_mlp`（pytorch_util.py） ✅ 已完成

**要做的：** 实现一个多层全连接网络。输入维数、输出维数、隐藏层数和每层宽度、激活函数都由参数给定。输出层可以另设激活（如 identity）。返回一个 `nn.Module`（例如用 `nn.Sequential` 组装）。

**思考点：** 层与层之间如何连接？第一层输入维、最后一层输出维、中间层维数如何用参数控制？

**验证：**
- 在 `pytorch_util.py` 里写一两行测试（或临时在 `run_hw1.py` 里 import 并调用 `build_mlp`），构造一个小 MLP，喂入随机 tensor，能跑通 forward 且输出形状正确。
- 不报错后删掉临时测试，继续下一步。

---

## 第 2 步：实现 MLP 策略（MLP_policy.py）

**要做的：**

1. **`get_action(obs)`**  
   用当前策略对给定观测输出动作。注意 obs 可能是单条或多条；要转成 tensor、放到正确设备、调用可微的 forward 得到分布后**采样**一次，再转回 numpy 给 env 用。

2. **`forward(observation)`**  
   可微前向：用 `mean_net` 和 `logstd` 得到动作的分布（连续动作用高斯），返回的东西要能既用来采样（get_action）又能用来算 loss（update）。

3. **`update(observations, actions)`**  
   用这批 (obs, action) 做一步监督学习：前向得到分布，用负对数似然（或你选的等价目标）算 loss，backward + optimizer.step()，返回包含 `'Training Loss'` 的 dict。

**思考点：** 训练时用 `sample()` 还是 `rsample()`？为什么？连续动作下 NLL 的公式是什么？

**验证：**
- `get_action`：用随机 obs 调用，得到 shape 为 `(ac_dim,)` 或 `(batch, ac_dim)` 的 numpy 数组。
- `forward`：同样输入转成 tensor，输出可 backward。
- `update`：用一小批随机 obs/action 调用，返回的 dict 里有 `'Training Loss'` 且为标量；多调用几次，loss 会变化（说明在更新）。

---

## 第 3 步：实现轨迹采集（utils.py）

**要做的：**

1. **`sample_trajectory(env, policy, max_path_length, ...)`**  
   重置 env，循环：用当前 obs 调 policy.get_action → env.step → 记录 obs、action、reward、next_obs、是否结束。结束条件：`done` 为 True 或步数达到 `max_path_length`。用已有的 `Path(...)` 把列表打包成一条轨迹字典返回。

2. **`sample_trajectories(...)`**  
   反复调用 `sample_trajectory`，直到累计步数 ≥ `min_timesteps_per_batch`。返回 paths 列表和总步数（用 `get_pathlength` 算每条长度）。

3. **`sample_n_trajectories(...)`**  
   调用 `sample_trajectory` 共 `ntraj` 次，返回这 n 条 paths。

**思考点：** 单条轨迹的 `terminals` 每个元素是 0 还是 1？最后一步的 next_obs 还要不要记？

**验证：**
- 写一个最小脚本：创建 env（如 Ant-v4）、用随机策略（或已实现的 MLP 策略）调 `sample_trajectory` 一次，检查返回的 path 里 `observation`、`action` 的 shape 和长度是否合理，且长度 ≤ max_path_length。
- 再测 `sample_trajectories`（较小的 min_timesteps_per_batch）和 `sample_n_trajectories`，确认返回的 paths 条数和总步数符合预期。

---

## 第 4 步：实现 replay buffer 采样（replay_buffer.py）

**要做的：** 实现 `sample_random_data(batch_size)`。从 buffer 的 obs、acs、rews、next_obs、terminals 里**用同一组随机下标**采样 batch_size 条，返回五个数组。提示：先用 `np.random.permutation` 或类似方式得到不重复的索引。

**验证：**
- 先往 buffer 里塞一些数据（或看 trainer 里怎么调 `add_path`），再调 `sample_random_data(batch_size)`，检查返回的五个数组长度都是 batch_size，且同一索引在各数组里对应同一条转移。

---

## 第 5 步：实现 BC 训练器（bc_trainer.py）

**要做的：**

1. **`collect_training_trajectories`**  
   - 第一次迭代：从 `initial_expertdata` 路径用 pickle 加载专家数据，转成代码里期望的 `paths` 格式，并算出 `envsteps_this_batch`（即这批轨迹的总步数）。  
   - 若为 DAgger 且不是第一次迭代：用当前 `collect_policy` 调用你在 utils 里实现的采样函数收集轨迹。  
   - 若需要录视频，用 `sample_n_trajectories` 得到 `train_video_paths`（trainer 里已有调用，你只需保证 `sample_n_trajectories` 已实现）。  
   返回 `paths, envsteps_this_batch, train_video_paths`。

2. **`train_agent`**  
   在每次迭代里循环若干步：从 agent 的 replay buffer 里 sample 一个 batch（用 agent.sample），用该 batch 调 agent.train，把返回的 log 收集起来，最后返回 all_logs。

3. **`do_relabel_with_expert`**  
   对每条 path，用 path 里的 observation 去问专家策略得到专家动作，把 path 的 action 替换成这些专家动作（为 DAgger 准备“专家标注”）。

**思考点：** 专家数据 pkl 里存的结构是什么？怎么和 `Path` 的 key（observation, action, ...）对应？DAgger 时第一次迭代用专家数据还是用当前策略采的数据？

**验证：**
- 先跑通 **BC**：`n_iter 1`、不用 `--do_dagger`。命令见 README（Ant + expert_data）。  
  - 若报错：根据报错位置回到对应步骤（例如 KeyError 在 trainer → 检查专家数据加载格式；在 agent.train → 检查 MLP_policy.update 的输入输出）。  
  - 若跑通：看终端或 tensorboard 里是否有 Eval AverageReturn 等；用 `--video_log_freq -1` 先不录视频加快调试。
- 再跑 **DAgger**：加上 `--do_dagger`、`n_iter 10`。确认能跑完多轮且每轮有收集、重标、训练。

---

## 第 6 步：完成作业报告（按 PDF）

**要做的：**

1. **Question 1.2（Table 1）**  
   在 Ant 上跑 BC，达到至少 30% 专家性能；再选一个环境（Hopper / Walker2d / HalfCheetah）跑 BC。两个任务都报告**多条 rollout 的回报均值与标准差**（用表格）。  
   - 验证：eval 时 `eval_batch_size` 要大于 `ep_len`，这样会采多条轨迹，Eval AverageReturn / Eval StdReturn 才是多条的均值和标准差。

2. **Question 1.3（Figure 1）**  
   选一个超参数（训练步数、专家数据量、网络大小等），在一个任务上做多组实验，画一张图：横轴超参数取值，纵轴 BC 回报；附简短说明为什么选这个超参数。

3. **Question 2.2（Figure 2）**  
   对 BC 用过的两个任务再跑 DAgger。画学习曲线：横轴 DAgger 迭代次数，纵轴回报均值，带标准差 error bar；同一张图里标出专家和 BC 的水平线。  
   - 验证：同一任务、相同网络和数据设置，和 BC 公平对比。

**验证：**  
- 报告里表格和图的数字与 tensorboard / 日志一致。  
- 提交前确认：data 里实验文件夹带 `q1`、`q2` 前缀；交代码时关掉视频日志（`--video_log_freq -1`）；README 里写清复现每条结果的完整命令。

---

## 小结：顺序与依赖

| 步骤 | 模块 | 验证方式 |
|------|------|----------|
| 0 | 通读代码 | 能说出数据流和 BC/DAgger 区别 |
| 1 | pytorch_util.build_mlp ✅ | 小 MLP forward 输出形状正确 |
| 2 | MLP_policy (get_action, forward, update) | 单步调用与 update 返回 loss |
| 3 | utils (sample_trajectory, sample_trajectories, sample_n_trajectories) | 单条/多条轨迹 shape 与长度 |
| 4 | replay_buffer.sample_random_data | 采样 batch 形状与对应关系 |
| 5 | bc_trainer (collect, train_agent, do_relabel) | 先 BC 跑通，再 DAgger 跑通 |
| 6 | 报告 Table 1, Figure 1, Figure 2 | 数字与日志一致、命令可复现 |

每一步验证通过后再进行下一步，遇到报错先定位到具体函数再回到对应步骤检查。
