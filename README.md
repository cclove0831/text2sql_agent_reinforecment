# text2sql_rlhf_agent

Text-to-SQL 多步智能体 + LoRA SFT 热身 + GRPO（on-policy）强化学习微调。

## 指标对比（CSpider Dev）

下表记录在 **`data/cspider_dev.jsonl`（1034 条样本）** 上的 **one-shot** SQL 生成评估结果，评估脚本为 `eval_sft_local.py`（贪心解码，指标为 `execution_match`）。

定义说明：
- `EX`：执行等价（与 `gt_sql` 结果一致，使用 `reward.execution_match`）
- `valid`：预测 SQL 能成功执行
- `logic_err`：`valid == True` 但 `EX == False`（可执行但逻辑错误）

| Setting | EX | valid | logic_err | Badcases |
| --- | ---: | ---: | ---: | --- |
| Base (`Qwen2.5-7B-Instruct`, no adapter) | 0.5542 (573/1034) | 0.8491 (878/1034) | 0.2950 (305/1034) | `data/badcase_base_dev.jsonl` |
| SFT+LoRA (1 epoch, adapter=`outputs/sft_lora`) | 0.7650 (791/1034) | 0.9342 (966/1034) | 0.1692 (175/1034) | `data/badcase_sft_dev.jsonl` |

提升（SFT - Base）：`EX +0.2108`，`valid +0.0851`，`logic_err -0.1258`。

### 复现方式

Base（不加载 LoRA adapter）：
```bash
python eval_sft_local.py \
  --model_name_or_path /root/models/Qwen2.5-7B-Instruct \
  --data_path data/cspider_dev.jsonl \
  --badcase_path data/badcase_base_dev.jsonl \
  --limit 0 --print_every 20 --bf16
```

SFT+LoRA（加载 `outputs/sft_lora`）：
```bash
python eval_sft_local.py \
  --model_name_or_path /root/models/Qwen2.5-7B-Instruct \
  --adapter_path outputs/sft_lora \
  --data_path data/cspider_dev.jsonl \
  --badcase_path data/badcase_sft_dev.jsonl \
  --limit 0 --print_every 20 --bf16
```

备注：
- 这里统计的是 **one-shot** 指标，不是多步 agent loop（`agent_llm.py` / `eval.py`）。
- 运行评估的机器需要具备 `data/cspider_dev.jsonl` 中引用的 `db_path` 对应数据库文件。
