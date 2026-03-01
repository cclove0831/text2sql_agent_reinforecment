# text2sql_rlhf_agent

一个面向 SQLite 的 Text-to-SQL 多步智能体项目，训练路线为：

1. `LoRA SFT`：让模型学会稳定的 SQL 生成格式与基本语法
2. `GRPO + LoRA`：在“多步工具调用 + 自我纠错”的 agent 轨迹上做强化学习微调，核心信号是执行等价（EX）

本仓库的重点不是“写出一条 SQL”，而是把 Text-to-SQL 做成一个 **可执行、可观察、可纠错、可训练** 的闭环系统：

- Agent：`[SCHEMA] -> [SQL] -> Observation -> ... -> [ANSWER]`
- Reward：`execution_match` 的硬正确性 + trace 过程塑形（避免投机与策略崩溃）
- Eval：区分 `Valid SQL` / `Agent OK` / `EX` / `Logic Error`，并输出 badcase 便于归因

---

## 1. 项目背景与问题定义

Text-to-SQL 的常见痛点（尤其在 Spider/CSpider 这类多 DB 多 schema 场景）：

- **可执行不等于正确**：模型能写出语法正确、可执行的 SQL，但语义偏了（典型表现：`valid_sql` 高、`EX` 低、`logic_err` 高）。
- **首次失败后缺少纠错闭环**：SQL 报错（no such column/table、语法错、JOIN 键错）时，one-shot 模型往往无法稳定修正。
- **训练信号不自然**：正确性最终体现在执行结果，不适合简单 token-level 监督；而 RL 又容易出现 tie、塌缩、reward hacking。

本项目的任务定义（训练/评估一致）：

- 输入：`question` +（由环境提供的）`schema`
- 输出：只读 SQL（`SELECT`/`WITH ... SELECT`，禁止写操作），通过执行结果与 `gt_sql` 的执行结果做等价判断（EX）

---

## 2. 代码结构与脚本职责

- `tools.py`
  - `show_schema(db_path, schema_path)`：读取 `schema_path` 或 introspection（`PRAGMA table_info`）得到 schema 文本
  - `execute_sql_dict(sql, db_path, max_rows=5)`：只读校验 + 执行 SQL + 截断返回行（用于 agent 的 Observation）
- `agent_llm.py`
  - 远端 API 版本 agent（OpenAI 兼容接口），实现多步循环、动作解析、Observation 回灌、trace 记录
- `reward.py`
  - `execution_match`（EX）与 `compute_reward`（硬正确性 + 轨迹塑形）
- `train_sft_lora.py`
  - one-shot SFT：监督输出 `[SQL] <gt_sql>`
- `train_grpo_lora.py`
  - GRPO：对每个样本采样 `group_size` 条多步轨迹，按组内优势更新 LoRA
  - 内置 rollout（与 `agent_llm.py` 协议一致，但使用本地 HF 模型）
  - 支持 TensorBoard、KL 约束（reference adapter/model）、低信息组跳过、collapse guard 等稳定性工程
- `eval_sft_local.py`
  - one-shot 本地评估（base / SFT LoRA），输出 `EX/valid/logic_err` + badcase
- `eval_agent_local.py`
  - 多步 agent 本地评估（base / SFT LoRA / GRPO LoRA），输出 `EX/valid_sql/agent_ok/no_sql/logic_err/avg_steps/...` + badcase
- `eval.py`
  - 多步 agent 远端 API 评估（与 `agent_llm.py` 对齐），同样输出 `EX/valid_sql/agent_ok/...` + badcase

---

## 3. 数据格式（CSpider jsonl）

本仓库使用的训练/验证数据为 `jsonl`，每行 1 个样本，核心字段：

- `id`：样本 id（可选）
- `db_id`：数据库 id（可选）
- `question`：自然语言问题
- `gt_sql`：gold SQL（“金标准” SQL，用于执行等价对比）
- `db_path`：SQLite 数据库文件路径
- `schema_path`：schema 文本路径（可选；用于避免每次 introspection）

注意：

- `gt_sql` 是训练与评估的唯一 “gold” 来源，本项目 **不要求** 模型生成的 SQL 与 `gt_sql` 字符串一致，只要执行结果等价即可（见第 6 节 EX 定义）。
- `schema_path`/`db_path` 是 **数据集给定** 的，不是 LLM“自己选择 schema”。在 Spider/CSpider 里，db 的选择天然由 `db_id/db_path` 决定，这就是常见 NL2SQL 设定。

---

## 4. Agent：执行协议与实现细节（面试重点）

### 4.1 为什么需要多步 Agent（而不是 one-shot）

多步的价值来自两个闭环：

1. **Schema Linking 闭环**：先拿 schema，再写 SQL，减少编造表/列
2. **执行纠错闭环**：SQL 报错时，把 `Error:` 作为 Observation 回灌，让模型修正下一条 SQL

这种“工具调用”形式，天然更适合后续用 RL 做策略优化：reward 可以绑定在可观测的执行反馈上。

### 4.2 动作协议（Action Space）

Agent 每一轮 **必须且只能** 输出以下三种 action 之一（大小写不敏感，严格推荐大写标签）：

- `[SCHEMA]`
  - 没有参数，只能单独输出（`agent_llm.py`、`train_grpo_lora.py`、`eval_agent_local.py` 都会按此解析）
- `[SQL] <query>`
  - `<query>` 必须为单条 SQLite `SELECT`/`WITH ... SELECT`
  - 禁止写操作/PRAGMA/VACUUM/ATTACH 等（只读约束在 `tools.py` 与 `reward.py` 都有校验）
- `[ANSWER] <answer>`
  - `<answer>` 必须复用 Observation 中的 `Answer:` 值（agent prompt 明确要求“不要添加额外文字”）

### 4.3 动作解析（parse_action 的优先级与容错）

实现位置：

- 远端：`agent_llm.py:_parse_action`
- 训练 rollout：`train_grpo_lora.py:parse_action`
- 本地评估：`eval_agent_local.py:parse_action`

解析规则（非常关键，决定 trace 的统计与 reward 行为）：

1. 优先匹配最后出现的标签（使用 `rfind`）
2. 优先级：`[ANSWER]` > `[SQL]` > `[SCHEMA]`
3. 容错：
  - 模型偶尔会在 `[SCHEMA]` 后面直接跟一条 `SELECT`，解析器会把它当作 `sql`（减少格式问题导致的无意义 INVALID）
  - 如果完全没有标签但文本以 `SELECT/WITH` 开头，也会当作 `sql`

### 4.4 状态机与循环（Text2SQLAgent.run / rollout_trajectory）

Agent 是显式状态机，核心状态变量：

- `schema_calls`：schema 请求次数（用于防止 schema 循环）
- `last_sql`：最后一次尝试的 SQL（注意：这不一定是最后一次成功的 SQL）
- `last_answer`：最近一次 **成功执行** 的 SQL 对应的自然语言答案（由环境 `rows_to_answer` 生成）
- `last_error`：最近一次 SQL 的 error（用于提示纠错）

循环伪代码（与实现一致）：

1. 组织 `messages`：
  - system：`agent_llm.py:SYSTEM`（中文，强约束 action 格式与执行规范）
  - user：`问题 + 约束（第一步必须 schema、无 ok SQL 不得 answer）`
2. for step in `range(max_steps)`：
  - 生成 `model_text`
  - `action,payload = parse_action(model_text)`
  - 根据 action 执行工具：
    - SCHEMA：
      - `show_schema(db_path, schema_path)`
      - Observation 回灌到 messages
    - SQL：
      - `execute_sql_dict(sql, db_path)`
      - Observation 回灌（成功时附加 `Answer:`）
      - 如果 ok：更新 `last_answer`
    - ANSWER：
      - 若 `last_answer is None`：判 INVALID，回灌错误 Observation，继续下一轮（强制进入纠错）
      - 否则：结束
3. 超过 max_steps：
  - 若 `last_answer` 存在：返回 ok（回退）
  - 否则：返回失败

### 4.4.1 Observation 与 Answer 的具体格式

Observation 的格式是 agent 是否能稳定纠错的关键。本项目里 Observation 由两部分组成：

1. `tools.py:format_sql_output`：把执行状态、列名、行数据格式化为文本
2. `utils.py:rows_to_answer`：把 `rows` 归一成单行 `Answer:`（agent 提示词要求最终 `[ANSWER]` 必须逐字复用它）

执行失败时（模型必须修正 SQL）：

```text
Observation:
Error: sqlite3.OperationalError: no such column: foo
```

执行成功时（注意 `Rows` 只保留前 5 行用于上下文控制）：

```text
Observation:
OK
Columns: ['name']
Rows: [('France',), ('Netherlands',), ('United States',)]
Answer: France | Netherlands | United States
```

`Answer:` 的生成规则（`utils.py:rows_to_answer`）：

- 0 行：返回空字符串 `""`
- 1 行 1 列：返回标量（数值会做简单格式化）
- 多行 1 列：用 `" | "` 拼接每行的第 1 列
- 1 行多列：用 `", "` 拼接
- 其它情况：回退到 `str(rows)`

重要区别：

- Observation 的 `Rows` 会在 `tools.py` 里截断到最多 5 行（避免上下文爆炸）
- EX/Reward 的执行等价比较在 `reward.py` 里执行（默认遍历全结果；若设置 `max_compare_rows` 才会截断并判 `truncated`）

### 4.5 关键保护逻辑（防“策略崩溃/投机”）

1. **禁止早答（Answer-before-SQL-ok）**
  - 在没有任何 `ok=True` 的 SQL 之前输出 `[ANSWER]`，直接记录 `INVALID`，并回灌错误 Observation
  - 目的：逼迫模型走“执行纠错闭环”，否则会学到“直接回答”这种不可训练的捷径
2. **Schema 循环上限**
  - `schema_calls > 2` 直接 `INVALID(too_many_schema_calls)`，并回灌“请直接输出 SQL”
  - 目的：避免策略收敛到“反复要 schema、从不写 SQL”的局部最优
3. **只读 SQL 双重约束**
  - `tools.py`：执行前直接拒绝危险 SQL
  - `reward.py`：r_exec 计算时也会 `_validate_readonly_sql`（保证训练信号和执行一致）

### 4.6 Schema 是怎么“选”的

结论：schema 是 **数据集与环境确定的**，不是 LLM 自己“选择数据库/选择 schema 文件”。

- 数据集每条样本自带 `db_path/schema_path`
- Agent 的 `[SCHEMA]` 只是触发工具 `show_schema(db_path, schema_path)`
- `tools.py:show_schema` 的优先级：
  1. 如果传入 `schema_path` 且文件存在：直接读 schema 文本
  2. 否则如果 `schema_path` 为空且 `db_path` 为空，并且存在默认 `data/schema.txt`：直接读该文件
  3. 否则：连接 SQLite，用 `sqlite_master + PRAGMA table_info` introspection（当 `db_path` 为空时使用默认 `data/chinook.db`）

这是 Spider/CSpider 的常见设定：db_id 已知，模型不需要在多库中搜索。

---

## 5. Reward：硬正确性 + 轨迹塑形（面试重点）

### 5.1 Reward 的目标分解

如果只用结果奖励（execution_match），在多步 agent 里会出现两个问题：

- **组内 tie（stdR=0）太多**：全对/全错组无法排序，GRPO 没梯度信号
- **错误内部无法分级**：全错时每条轨迹 reward 都一样，策略很难从“完全不会”推进到“更像 SQL”

因此使用混合 reward：

- `r_exec`：硬正确性（执行等价）为主
- `r_trace`：小幅塑形（tie-break + 约束 + 错内分级）为辅

最终：

- `R = weight_exec * r_exec + weight_trace * r_trace`
- 默认：`weight_exec=0.65`，`weight_trace=0.35`（训练可覆盖）

### 5.2 r_exec：execution_match（EX）

实现：`reward.py:execution_match`，内部调用 `exec_sql_signature` 执行 SQL 并构造结果签名。

执行签名（order-insensitive，多集签名）：

- 对每一行 `row`：
  - `row_digest = blake2b(repr(canonicalized_row)) -> (a,b)` 两个 64-bit 整数
  - 聚合：`xor1 ^= a`，`xor2 ^= b`，`sum1 += a (mod 2^64)`，`sum2 += b`
  - `count += 1`
- 最终 `sig = (count, xor1, xor2, sum1, sum2)`

匹配条件（全部满足才算 EX=1）：

1. `pred.ok == True` 且 `gt.ok == True`（两条 SQL 都能执行）
2. `pred.sig == gt.sig`
3. `pred.truncated == False` 且 `gt.truncated == False`

`truncated` 的含义：

- 当设置 `max_compare_rows=N` 时，如果结果行数超过 N，会在第 N 行后停止遍历并标记 `truncated=True`
- 为避免“只比前 N 行造成假阳性”，只要任意一方 truncated，就直接判不匹配

重要结论（回答你之前的疑问）：

- **完全可能出现 pred_sql != gt_sql 但 EX=1**：只要执行结果等价（多种 SQL 写法等价很常见）。
- EX 低不代表 SQL 字符串不一样，而通常意味着：语义错、执行报错、或被 `max_compare_rows` 截断导致严格不匹配。

### 5.3 pred_sql_used：从 trace 里选“用于打分的 SQL”

这是多步 agent 的关键细节：最后一次尝试的 SQL 不一定是最终答案依赖的 SQL。

实现：`reward.py:compute_reward`

优先级：

1. `trace_last_ok`：trace 里最后一次 `action=="SQL" and ok==True` 的 SQL
2. `trace_last`：如果没有 ok SQL，退化到 trace 里最后一次 SQL
3. `arg`：如果 trace 里没有 SQL，才用外部传入的 `pred_sql`

输出字段：

- `pred_sql_used`
- `pred_sql_source`（trace_last_ok / trace_last / arg / none）

这能避免一种常见误判：前面成功 SQL 正确，后面又试了一个失败 SQL，最后 `[ANSWER]` 复用成功结果；如果只拿“最后一次 SQL”打分，会把正确轨迹判成错误。

### 5.4 r_trace：轨迹塑形（具体项、方向与幅度）

实现：`reward.py:compute_reward`，输出在 `reward_detail` 里。

核心原则：

1. 正向塑形只在 **r_exec 正确时** 给（避免把策略推向“可执行但不正确”）
2. 失败时主要给约束项惩罚（防止 stuck、早答、schema 循环、重复 SQL）
3. 通过 tie-break 与错内分级打破 `stdR=0`

具体项（按代码顺序）：

- schema-first：
  - 如果第 1 步不是 `SCHEMA`：`-0.25`
  - 如果 schema-first 且 `r_exec>0`：`+0.25`
- 至少一次 SQL ok：
  - 如果 `sql_ok_count==0`：`-0.25`
  - 如果 `sql_ok_count>0` 且 `r_exec>0`：`+0.25`
- degenerate：`sql_calls==0`（只要 schema 不写 SQL）：`-0.75`
- schema 循环：`schema_calls>1`：`-0.10 * min(5, schema_calls-1)`
- answered 但无 ok SQL：`-0.50`
- INVALID 动作：`-0.15 * min(3, invalid_count)`
- 早答 INVALID：`-0.50 * min(2, early_answer_count)`
- SQL 尝试次数：`-0.05 * min(6, max(0, sql_calls-1))`
- SQL 失败次数：`-0.03 * min(6, sql_fail_count)`
- 重复同一条 SQL：`-0.05 * min(4, repeated_sql_count)`
- 幻觉错误（no such table/column）：`-0.10 * min(3, hallucination_err_count)`
- 非法 SQL（写操作/pragma）：`-0.20 * min(1, illegal_sql_count)`
- 额外步数：`-0.05 * min(6, extra_steps)`，其中最短成功轨迹基线为 3 步（SCHEMA->SQL->ANSWER）
- 错误时的 join 约束：若 `r_exec<0 and join_pred>join_gt`：额外 `-0.10`

正确轨迹的 tie-break（打破“全对组 stdR=0”）：

- 仅当 `r_exec>0`：
  - `-0.0001 * min(2000, sql_len)`
  - `-0.02 * min(6, join_pred)`

错误内部的分级信号（打破“全错组 stdR=0”）：

- 仅当 `r_exec<0`：
  - `overlap_bonus = 0.05*table_recall + 0.05*column_recall`
  - 若 `gt_rows>0` 且 pred 也 ok：非空结果 +0.02，空结果 -0.02（很小的信号）

最终 `r_trace` 会 clamp 到 `[-1, 1]`，再与 `r_exec` 混合。

---

## 6. 评估：指标定义、EX 计算与对比方式（面试重点）

### 6.1 三类评估脚本（不要混用）

1. `eval_sft_local.py`：one-shot SQL 生成评估
  - 输入：`Question + Schema(text)`
  - 输出：单条 SQL
  - 适合比较 base vs SFT LoRA（同一 prompt 下）
2. `eval_agent_local.py`：多步 agent 本地评估（HF 模型）
  - 输入：多步协议（SCHEMA/SQL/ANSWER）
  - 输出：`trace + pred_sql_last + pred_sql_used + answer`
  - 适合比较 base vs SFT LoRA vs GRPO LoRA（同一 agent 策略下）
3. `eval.py`：多步 agent 远端 API 评估
  - 调用 `agent_llm.py:Text2SQLAgent`
  - **注意**：本仓库已修复一个关键隐患：评估 EX 时不再只使用最后一次 SQL，而是从 trace 里选最后一次 ok SQL（与训练一致）。

结论：

- one-shot 的 EX 和 agent 的 EX 不应直接横向比较（prompt/策略不同）。
- 你应该选择同一脚本、同一 `max_steps/max_new_tokens/max_compare_rows` 来对比 base/SFT/GRPO。

### 6.2 指标含义（以 eval_agent_local 为准）

`eval_agent_local.py` 最终打印：

- `Execution Accuracy (EX)`：
  - 定义：`execution_match(pred_sql_used, gt_sql) == True` 的样本占比
  - **不要求** SQL 字符串一致，只看执行结果签名一致
- `Valid SQL Rate`：
  - 定义：`pred_sql_used` 执行成功（`pred_ok=True`）的占比
  - 常见现象：valid 很高但 EX 不高，说明“语义错误”居多
- `Agent OK Rate`：
  - 定义：agent 运行结束时 `ok=True` 的占比（本项目里意味着至少有一次 SQL ok，并成功输出 ANSWER）
  - 不代表正确，只代表“跑完了闭环”
- `No SQL Rate`：
  - 定义：整条轨迹 `sql_calls==0` 的占比（只要 schema、不写 SQL）
- `Logic Error Rate`：
  - 定义：`Valid SQL` 但 `EX=False` 的占比
  - 这是 Text2SQL 的核心错误类型（比语法错更重要）
- `Avg Steps`：
  - trace 的平均长度（包含 INVALID/SCHEMA/SQL/ANSWER）
- `Avg SQL Attempts`：
  - 每条样本平均执行 SQL 次数（反映纠错强度）

### 6.3 EX 为什么可能被“低估”

即使 pred 与 gt 语义等价，仍可能 EX=0（严格匹配造成的 false negative），常见原因：

1. `max_compare_rows` 太小导致任意一方 `truncated=True`（严格判不等价）
2. 非确定性结果：
  - 缺少 `ORDER BY` 的 SQL 可能返回顺序不稳定
  - 本项目的签名是 order-insensitive，因此对“顺序变化”不敏感，但如果 DB 本身数据/类型导致行表示差异，也可能出现
3. 数值表示差异：
  - `reward.py` 对 float 做了 `1e-6` 容差与 round，但仍可能因极端浮点/NaN/Inf 等导致差异

### 6.4 badcase 文件结构（用于归因）

`eval_agent_local.py`/`eval.py` 的 badcase 至少包含：

- `pred_sql_last`：最后一次尝试的 SQL（可能失败）
- `pred_sql_used`：用于 EX 打分的 SQL（优先取最后一次 ok 的 SQL）
- `pred_sql_source`：`trace_last_ok/trace_last/arg/...`
- `execution_detail`：包含 `pred_ok/pred_error/pred_rows/pred_truncated/gt_*`
- `trace`：每步 action、SQL、error、rows（截断）、token_ids/prompt_messages（训练 rollout 才有）

面试官常问：你如何证明“评估和训练是一致的”？这里的 `pred_sql_used` 与 `execution_detail` 是关键证据。

---

## 7. GRPO 训练：为什么会崩、如何防崩、怎么看趋势

### 7.1 GRPO 的更新单元是什么

`train_grpo_lora.py` 的更新单位是 **group**（一个样本 + 多条 rollout）。

对每个训练样本：

1. rollout `group_size` 条轨迹（`rollout_trajectory`，采样解码）
2. 用 `compute_reward` 得到每条轨迹的 reward
3. 组内标准化优势：
  - `adv_i = (R_i - mean(R)) / (std(R) + 1e-6)`，并可 `adv_clip`
4. 对每条轨迹计算 policy logprob（按 token）并反传：
  - `loss_i = -(adv_i * adv_weight) * lp + kl_beta * kl_term`

### 7.2 为什么训练中会出现大量 `stdR=0`

两类根因，面试要能分清：

1. **reward tie（全对/全错且 reward 常数）**
  - 解决：reward 引入 tie-break（正确内部）与错内分级（overlap_bonus）
2. **采样塌缩（do_sample 形同虚设，组内输出高度一致）**
  - 解决：确保 `do_sample=True`，并增大温度/提高 top_p
  - 本仓库还提供 `--debug_tie_groups`：在 low-std/skip 的 group 打印 SQL hash，快速判断是不是采样塌缩

### 7.3 防“策略崩溃”的工程保护（关键）

`train_grpo_lora.py` 内置 3 个层级的保护，避免“越训越差”：

1. **低信息组跳过更新**
  - `std_reward < --skip_update_std`：直接 skip（默认 1e-3，可调高到 5e-3）
  - 目的：避免在 tie 上做无意义更新，放大噪声
2. **no-EX 组的处理策略**
  - `ex@group==0` 时由 `--no_ex_update` 控制：
    - `skip`（推荐）：不更新，避免追着错误信号跑
    - `kl_only`：adv_weight=0，仅保 KL（需要 kl_beta>0）
    - `scale`：adv_weight 缩小
3. **collapse_guard**
  - 当 `ex@group==0` 且组内轨迹高度一致（deterministic）且 `no_sql_rate` 很高时，强制 skip
  - 目的：避免在“坏吸引子”上反复强化

### 7.4 KL 约束是怎么做的（为什么不用 RL 框架也能做）

脚本支持两种 reference policy：

1. `--ref_model_name_or_path`：单独加载一个 ref model（更耗显存）
2. 同一个模型里加载一个 **冻结的 ref adapter**（更省显存，推荐）：
  - `--adapter_path outputs/sft_lora`：policy 从 SFT adapter 开始训练（trainable）
  - `--ref_adapter_path outputs/sft_lora` + `--ref_adapter_name ref`：加载 frozen ref adapter
  - 训练时在 policy/ref adapter 之间切换计算 logprob

KL 项的实现是 “log-ratio 的平方”：

- `log_ratio = lp - ref_lp`
- `kl_term = log_ratio^2`

这不是严格的 token-level KL，但在工程上更稳定，且与 `kl_beta` 联动用于“防发散”。

### 7.5 TensorBoard 日志写到哪里

`train_grpo_lora.py`：

- 开启 `--tensorboard` 后，默认写到：`<output_dir>/tb`
- 也可以显式指定：`--tb_logdir /some/path`

如果你在 AutoDL 面板里打开 TensorBoard 但显示空：

1. 大概率是 logdir 指错了（面板默认可能是 `/root/tf-logs`）
2. 把面板的目录改成你实际的 `<output_dir>/tb`（例如 `/root/text2sql_agent_reinforecment/outputs/grpo_lora_final/tb`）

### 7.6 如何判断“没必要继续训”

训练时不要只看 `train/mean_reward`，至少同时看：

- `train/ex_rate`：越高越好，但会抖动
- `train/std_reward`：长期接近 0 说明组内无可学习信号（要么全对、要么全错、要么采样塌缩）
- `train/kl_mean`：持续飙升说明策略偏离过快（需要增大 kl_beta 或降低 lr）
- `train/no_sql_rate`、`train/avg_sql_calls`：
  - `no_sql_rate` 上升通常是坏信号（策略退化为不写 SQL）
  - `avg_sql_calls` 过低可能是“过早停机”，过高可能是“乱试”

真正的 early-stop 依据：

- 定期在 dev 上跑 `eval_agent_local.py`（见第 8 节），如果 EX 连续 N 次评估不再上升，且 badcase 类型没有改善，就可以停。

---

## 8. 推荐复现流程与命令（可直接复制）

### 8.1 数据准备与校验

```bash
python scripts/prepare_cspider_jsonl.py
python scripts/validate_dataset.py --data_path data/cspider_train.jsonl
python scripts/validate_dataset.py --data_path data/cspider_dev.jsonl
```

### 8.2 SFT 预热（LoRA）

```bash
python train_sft_lora.py \
  --model_name_or_path /root/models/Qwen2.5-7B-Instruct \
  --train_path data/cspider_train.jsonl \
  --output_dir outputs/sft_lora \
  --max_seq_len 2048 \
  --bf16 --gradient_checkpointing \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2
```

### 8.3 GRPO + LoRA（稳定优先的推荐参数）

先跑小规模 sanity（比如 `--max_groups 200`），确认不会 OOM、日志正常、ex_rate/std_reward 不异常，再跑全量。

```bash
python train_grpo_lora.py \
  --model_name_or_path /root/models/Qwen2.5-7B-Instruct \
  --adapter_path outputs/sft_lora \
  --ref_adapter_path outputs/sft_lora --ref_adapter_name ref \
  --train_path data/cspider_train.jsonl \
  --output_dir outputs/grpo_lora_stable_v1 \
  --bf16 --gradient_checkpointing \
  --attn_implementation sdpa \
  --group_size 8 \
  --adaptive_group_sampling --min_group_size 4 --target_reward_std 0.30 \
  --temperature 1.0 --top_p 0.95 \
  --max_steps 6 --max_new_tokens 256 \
  --max_compare_rows 1000 \
  --learning_rate 3e-5 --warmup_steps 200 --max_grad_norm 0.5 \
  --weight_exec 0.70 --weight_trace 0.30 \
  --kl_beta 0.02 \
  --no_ex_update skip --skip_update_std 0.005 \
  --log_every 10 --save_every 500 \
  --num_epochs 1 \
  --tensorboard
```

如果仍然 OOM，优先按顺序降：

1. `--group_size`（8 -> 6 -> 4）
2. `--max_new_tokens`（256 -> 192 -> 128）
3. 保持 `--gradient_checkpointing` 开启

### 8.4 评估（base / SFT / GRPO）

one-shot：

```bash
python eval_sft_local.py \
  --model_name_or_path /root/models/Qwen2.5-7B-Instruct \
  --data_path data/cspider_dev.jsonl \
  --badcase_path data/badcase_base_dev.jsonl \
  --limit 0 --print_every 20 --bf16

python eval_sft_local.py \
  --model_name_or_path /root/models/Qwen2.5-7B-Instruct \
  --adapter_path outputs/sft_lora \
  --data_path data/cspider_dev.jsonl \
  --badcase_path data/badcase_sft_dev.jsonl \
  --limit 0 --print_every 20 --bf16
```

多步 agent：

```bash
python eval_agent_local.py \
  --model_name_or_path /root/models/Qwen2.5-7B-Instruct \
  --data_path data/cspider_dev.jsonl \
  --badcase_path data/badcase_base_agent_dev.jsonl \
  --limit 0 --print_every 20 \
  --max_steps 6 --max_new_tokens 256 --max_compare_rows 1000 \
  --bf16

python eval_agent_local.py \
  --model_name_or_path /root/models/Qwen2.5-7B-Instruct \
  --adapter_path outputs/sft_lora \
  --data_path data/cspider_dev.jsonl \
  --badcase_path data/badcase_sft_agent_dev.jsonl \
  --limit 0 --print_every 20 \
  --max_steps 6 --max_new_tokens 256 --max_compare_rows 1000 \
  --bf16

python eval_agent_local.py \
  --model_name_or_path /root/models/Qwen2.5-7B-Instruct \
  --adapter_path outputs/grpo_lora_stable_v1/checkpoint_group006500 \
  --data_path data/cspider_dev.jsonl \
  --badcase_path data/badcase_grpo_agent_ckpt6500_dev.jsonl \
  --limit 0 --print_every 20 \
  --max_steps 6 --max_new_tokens 256 --max_compare_rows 1000 \
  --bf16
```

---

## 9. 已验证的指标对比（示例）

以下数字来自本项目实跑（CSpider Dev 1034）：

### 9.1 one-shot（`eval_sft_local.py`）

| Setting | EX | Valid SQL | Logic Error |
| --- | ---: | ---: | ---: |
| Base (`Qwen2.5-7B-Instruct`) | 0.5542 | 0.8491 | 0.2950 |
| SFT+LoRA (`outputs/sft_lora`) | 0.7650 | 0.9342 | 0.1692 |

### 9.2 多步 Agent（`eval_agent_local.py`）

| Setting | EX | Valid SQL | Agent OK | No SQL | Logic Error | Avg Steps | Avg SQL Attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.5658 | 0.9188 | 0.9188 | 0.0000 | 0.3530 | 3.53 | 1.54 |
| GRPO ckpt6500 | 0.7708 | 0.9797 | 0.9797 | 0.0000 | 0.2089 | 3.08 | 1.10 |

结论（面试口径）：

- GRPO 在 agent 场景下同时提升 `EX` 与 `Valid SQL`，并显著降低 `Logic Error` 和 SQL 尝试次数。
- `Avg Steps` 变小是好是坏要结合 EX 看：
  - 如果 EX 上升且 steps 下降：通常代表更快找到正确 SQL（好）
  - 如果 EX 下降且 steps 下降：可能是策略退化成“少尝试、早停机”（坏）

---

## 10. 训练中遇到的问题与解决（面试重点）

### 10.1 “valid 越训越高，但 EX 越训越差”

典型原因：

- reward/更新策略把“可执行”当成了主要目标（或在 no-EX 组上也强更新），导致策略朝“安全可执行但不正确”收敛

本仓库的对应修复：

- `reward.py`：正向塑形只在 `r_exec>0` 时给，错误时主要是约束惩罚与错内分级
- `train_grpo_lora.py`：
  - `--no_ex_update skip`：no-EX 组不更新
  - `--skip_update_std`：低信息组不更新
  - `collapse_guard`：deterministic 且 no-ex 且 no-sql 高的组强制跳过

### 10.2 大量 `stdR=0` 导致训练无效

原因分两类：

- reward tie：全对/全错都同分
- 采样塌缩：组内 SQL 输出几乎一样

修复：

- reward 增加 tie-break（正确内部）与 overlap_bonus（错内分级）
- 训练开启 `--debug_tie_groups`，直接打印 SQL hash 定位是否“采样问题”
- 训练开启 `--adaptive_group_sampling`：无方差时提前停止采样，节省计算

### 10.3 评估与训练不一致导致 EX 被低估

多步 agent 常见坑：最后一次 SQL 不一定是最后一次成功 SQL。

修复：

- `reward.py`：从 trace 里选 `pred_sql_used`
- `eval.py`（远端评估）：同样改为使用 `compute_reward` 的 `pred_sql_used` 逻辑（避免低估 EX）

---

## 11. 如果我是面试官，我会怎么追问

你应该能用代码级细节回答这些问题（README 的第 4/5/6 节就是为此写的）：

1. Agent 的 action 协议是什么？解析优先级是什么？如果模型输出不规范怎么处理？
2. schema 从哪里来？LLM 是否会“自己选 schema”？为什么这样设计符合 Spider/CSpider？
3. 什么情况下 `out["sql"]` 不是用于打分的 SQL？你怎么修复评估与训练的一致性？
4. EX 是怎么计算的？为什么 pred_sql != gt_sql 也可能 EX=1？有哪些 false negative？
5. 为什么要引入 trace reward？你如何防止 reward 把策略推向“可执行但不正确”？
6. GRPO 里 `stdR=0` 是 reward tie 还是采样塌缩？你用什么证据区分？
7. 训练中出现长期 `ex@group=0` 时，你如何判断是数据问题、reward 问题、还是策略已经崩？
8. 你为什么不用现成 RL 框架？不用的好处是什么？（本项目：可控、可 debug、可定制，代价是工程要自己兜底）
