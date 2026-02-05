---
name: text2sql-plan
description: Text2SQL + GRPO 强化学习智能体的开发规范与工作流（本仓库）。当需要实现/改造 Text-to-SQL agent（多步工具调用/ReAct、自我纠错）、设计 reward、实现 rollout 采样、编写 GRPO 训练脚本、完善评估指标与 badcase 记录时使用。
---

# Text2SQL + RL Agent 核心开发规范与技能指南

本文件定义了构建基于 GRPO 强化学习的 Text2SQL 智能体的核心逻辑、奖励策略及工程实现规范。作为 AI 辅助开发者，请严格遵循以下准则进行代码生成和逻辑设计。

## 1. 项目总体目标 (Project Objectives)

- **核心任务**：构建一个能够通过自然语言查询 SQLite 数据库的智能体（Agent）。
- **关键特性**：
  - **ReAct 范式**：不仅生成 SQL，还具备“思考-行动-观察”的闭环能力。
  - **自我纠错**：当 SQL 执行报错时，Agent 能根据错误信息修正查询。
  - **强化学习驱动**：利用 GRPO (Group Relative Policy Optimization) 算法，通过执行结果反馈来提升模型的逻辑推理准确率，消除幻觉。
  - **资源高效**：基于 7B-14B 模型，配合 LoRA 微调，适应单卡训练环境。

## 2. Agent 构建策略与多步工作流 (Agent Strategy & Workflow)

Agent 必须被设计为一个状态机，能够处理多轮工具调用。

### 2.1 交互流程 (The Loop)

1. **Thought (思考)**：模型分析用户意图，决定下一步行动（是查 Schema 还是写 SQL）。
2. **Action (行动)**：模型输出特定 Token 触发工具调用。
   - 格式规范：`[SQL] SELECT * FROM ...` 或 `[SCHEMA]`
3. **Observation (观察)**：环境执行工具，将结果或错误信息返回给模型。
   - 成功：返回截断后的查询结果（避免 Context 溢出）。
   - 失败：返回 `Error: <具体错误信息>`，强制模型进入纠错模式。
4. **Answer (回答)**：当获取到足够信息后，输出最终答案。
   - 格式规范：`[ANSWER] <最终结论>`

### 2.2 工具函数定义 (Tool Definitions)

请在 `tools.py` 中实现以下逻辑：

- **`execute_sql(query: str) -> str`**
  - **安全限制**：必须检查 SQL 是否包含 `DROP`, `DELETE`, `INSERT`, `UPDATE` 等写操作，仅允许 `SELECT`。
  - **异常处理**：捕获 `sqlite3.OperationalError`，绝对不要 crash，而是将异常信息转为字符串返回（这对 RL 负反馈至关重要）。
  - **结果格式化**：如果结果行数 > 5，仅返回前 5 行并附加 `...(truncated)` 提示。

- **`show_schema() -> str`**
  - 返回数据库的 `CREATE TABLE` 语句或精简后的表结构描述（表名+列名+类型）。
  - 这是解决 Schema Linking（模式链接）问题的关键工具。

## 3. 奖励函数设计策略 (Reward Function for GRPO)

在 `reward.py` 中，实现基于执行反馈的混合奖励机制。这是 GRPO 训练的核心。

### 奖励计算公式

$$ R_{total} = R_{correctness} + R_{validity} + R_{efficiency} + P_{penalty} $$

### 3.1 详细评分标准

1. **结果正确性 (Hard Reward)**
   - **+1.0 分**：`Set(Agent_Result) == Set(Ground_Truth_Result)`。忽略顺序，关注数据内容一致性。
   - **-1.0 分**：结果不一致，或最终答案错误。

2. **过程合法性 (Validity Reward)**
   - **+0.1 分**：SQL 语法正确且成功执行（即使结果是错的）。
   - **-0.5 分**：SQL 执行抛出异常（Syntax Error 或 Runtime Error）。

3. **幻觉惩罚 (Hallucination Penalty)**
   - **-1.0 分**：SQL 中引用了数据库中不存在的表名或列名（需解析 SQL AST 或正则匹配 Schema）。

4. **效率惩罚 (Efficiency Penalty)**
   - **-0.2 分**：Agent 生成的 SQL 包含的 JOIN 数量明显多于 Ground Truth SQL（惩罚冗余查询）。
   - **-0.1 分**：每多一轮无效的交互步骤（鼓励简洁）。

## 4. Rollout 采样逻辑 (Rollout Implementation)

在 `rollout.py` 中实现数据采集逻辑。GRPO 需要对同一个问题采集多条不同的轨迹（Trajectory）以计算基线。

### 逻辑实现

- **输入**：一批 Prompt（问题 + Schema）。
- **参数**：
  - `Group Size (G)` = 4 ~ 8（每组样本数）。
  - `Temperature` = 0.8 ~ 1.0（必须较高，以保证生成的多样性，否则 GRPO 无法计算优势）。
- **执行**：
  - 对每个 Prompt 并行或串行运行 Agent Loop `G` 次。
  - 收集完整的对话历史（Prompt + Model Output）。
  - 立即调用 `compute_reward` 计算每条轨迹的分数。
- **输出数据结构**：

```python
[
  {
    "prompt": "...",
    "completion": "<think>...</think>[SQL]...",
    "reward": 1.0,
    "group_id": 1
  },
  ...
]
```

## 5. 训练脚本开发 (GRPO Training Script)

在 `train.py` 中，使用 HuggingFace TRL 或自定义 Loop 实现 GRPO。

### 关键逻辑

1. **加载模型**：Base Model + LoRA Adapter (Trainable)。
2. **优势计算 (Advantage Calculation)**：
   - 对于同一组（Group）内的轨迹 $i$：
   - $Adv_i = \frac{R_i - \text{Mean}(R_{group})}{\text{Std}(R_{group}) + \epsilon}$
   - 注意：GRPO 不需要 Critic 网络，只通过组内归一化来作为 Baseline。
3. **Loss 计算**：
   - 仅针对 Token 生成概率计算 Policy Gradient Loss，加权 $Adv_i$。
   - 可选：增加 KL 散度惩罚项，防止模型偏离初始模型太远。
4. **显存优化**：
   - 使用 `Gradient Checkpointing`。
   - Reference Model 可以卸载到 CPU 或量化加载。

## 6. 评估脚本开发 (Evaluation)

在 `eval.py` 中实现严格的测试逻辑。

- **模式**：`Temperature = 0` (Greedy Search)，确保结果可复现。
- **指标计算**：
  1. **Execution Accuracy (EX)**：执行结果匹配率（核心指标）。
  2. **Valid SQL Rate**：生成可执行 SQL 的比例。
  3. **Logic Error Rate**：SQL 可执行但结果错误的比例。
- **Bad Case 记录**：必须将错误案例保存到 JSON 文件，包含：Question, Generated SQL, Error Message, Ground Truth。

## 7. 指标对比分析 (Metrics Comparison)

验证强化学习效果的最终标准。需要在 README 或报告中呈现以下对比：

| 指标 (Metric) | Base Model (SFT only) | RL Agent (GRPO+LoRA) | 预期提升 |
| :--- | :--- | :--- | :--- |
| **准确率 (Accuracy)** | Baseline (e.g., 40%) | **Target (e.g., 75%+)** | +30% |
| **幻觉率 (Hallucination)** | High | **Near Zero** | 显著下降 |
| **语法错误率** | Medium | **Low** | 模型学会自我修正 |
| **平均交互步数** | High (乱试) | **Optimal** (直击痛点) | 收敛 |

