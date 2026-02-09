import os
from typing import Any

from openai import OpenAI

from tools import execute_sql_dict, format_sql_output, show_schema
from utils import rows_to_answer

SYSTEM = """你是一个面向 SQLite 数据库的 Text-to-SQL 智能体。

你必须遵循 Thought/Action/Observation 循环。每一轮你只能输出且必须输出 **一个** action：

[SCHEMA]
[SQL] <一条 SQLite 的 SELECT/CTE 查询>
[ANSWER] <最终答案>

【格式规范（非常重要）】
- `[SCHEMA]` **没有参数**：只输出一行 `[SCHEMA]`，后面不能跟任何 SQL/文字。
- 任何 SQL 语句必须放在 `[SQL]` 后面（不能写在 `[SCHEMA]` 后面）。
- `[SQL]` 后的 SQL 可以多行，但必须是 **单条语句**，并且必须以 `SELECT` 或 `WITH` 开头。
- 你可以写可选的 `<think>...</think>`；但在 `<think>` 之外，只能输出 action token 以及它的 payload（不要输出解释性文字）。

【执行规范】
- 第 1 步必须是 `[SCHEMA]`。如果你还没看到 schema，就立刻输出 `[SCHEMA]`。
- 只能使用 schema 里出现的表名/列名，禁止编造任何名称。
- 只允许只读 SQL：`SELECT` 或 `WITH ... SELECT`。禁止 `PRAGMA`，禁止任何写操作（INSERT/UPDATE/DELETE/DROP/ALTER/VACUUM 等）。
- 尽量写明确的列名列表；除非调试需要，否则不要 `SELECT *`。
- 如果 Observation 以 `Error:` 开头，必须修正 SQL 再尝试。

【取值落地（解决“法国/France”等问题）】
- **不要翻译字符串常量**：如果 SQL 里写了 `'...'`，必须使用数据库里真实存在的值。
- 如果问题要求按文本值过滤，且可能存在不同写法/语言/大小写：先用 SQL 探查取值，再使用你观察到的精确值。例如：
  - [SQL] SELECT DISTINCT <col> FROM <table> LIMIT 50
  - [SQL] SELECT <col> FROM <table> WHERE <col> LIKE '%来自问题的关键词%' LIMIT 50
- 如果你的查询结果意外为空，优先检查：JOIN 是否多余/键是否正确/过滤值是否真实存在。

【常见模板】
- 最高/最低/最早/最晚/最年轻 等：`ORDER BY ... LIMIT 1`。
- 计数：`COUNT(*)`；问“不同/去重”的数量用 `COUNT(DISTINCT col)`。
- 列表去重：`DISTINCT`。
- JOIN：能不 JOIN 就不 JOIN；需要 JOIN 时，只在 schema 显示的外键/ID 列上连接。

【语义正确性自检（写在 <think> 里）】
- 最终 SELECT **只选问题要求的列**（不要多选无关列）。
- 只有当问题问“按组分别统计/每个…多少”时才用 `GROUP BY`；全局聚合（AVG/MIN/MAX/COUNT 总体）不要 `GROUP BY`。
- 问单个最值项时必须 `ORDER BY + LIMIT 1`（或合适的 LIMIT）。
- 你加了 JOIN 的话，确认两边表/列都在 schema 中，且 JOIN 是必要的、连接键方向正确。

【回答】
- 当 SQL 成功执行时，Observation 会包含一行 `Answer:`（由环境基于返回行计算得到）。
- 准备好回答时，输出 `[ANSWER]`，并且紧跟 **完全一致** 的 `Answer:` 值（不要加多余文字）。
"""


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        t = "\n".join(lines).strip()
    return t.strip("`").strip()


def _parse_action(text: str) -> tuple[str, str]:
    t = (text or "").strip()
    upper = t.upper()

    if "[ANSWER]" in upper:
        idx = upper.rfind("[ANSWER]")
        payload = t[idx + len("[ANSWER]") :].strip()
        return "answer", _strip_code_fences(payload)

    if "[SQL]" in upper:
        idx = upper.rfind("[SQL]")
        payload = t[idx + len("[SQL]") :].strip()
        return "sql", _strip_code_fences(payload)

    if "[SCHEMA]" in upper:
        # Some models mistakenly output SQL after [SCHEMA]. Treat it as SQL for robustness.
        idx = upper.rfind("[SCHEMA]")
        after = t[idx + len("[SCHEMA]") :].strip()
        maybe = _strip_code_fences(after)
        if maybe.lower().lstrip().startswith(("select", "with")):
            return "sql", maybe
        return "schema", ""

    # Fallbacks for slightly off-format outputs.
    maybe = _strip_code_fences(t)
    if maybe.lower().lstrip().startswith(("select", "with")):
        return "sql", maybe

    return "invalid", t


def _render_observation_schema(schema_text: str) -> str:
    return f"Observation:\n{schema_text}"


def _render_observation_sql(exec_out: dict[str, Any]) -> str:
    base = format_sql_output(exec_out)
    if not exec_out.get("ok"):
        return f"Observation:\n{base}"
    answer = rows_to_answer(exec_out.get("rows") or [])
    return f"Observation:\n{base}\nAnswer: {answer}"


class Text2SQLAgent:
    def __init__(
        self,
        model_name: str = "qwen2.5-7b-instruct",
        api_key: str | None = None,
        base_url: str | None = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_steps: int = 8,
        temperature: float = 0.0,
    ):
        """
        model_name: Identifier for the chat-completion model exposed by the API.
        api_key: OpenAI-compatible API key (defaults to env `OPENAI_API_KEY`).
        base_url: Optional custom endpoint for proxies/self-hosted gateways.
        max_steps: Max ReAct steps (model calls).
        temperature: Sampling temperature (0.0 for eval; higher for rollout).
        """
        self.model_name = model_name
        resolved_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Missing API key. Pass --api_key (or api_key=...) or set one of "
                "OPENAI_API_KEY / DASHSCOPE_API_KEY / QWEN_API_KEY."
            )
        self.client = OpenAI(api_key=resolved_key, base_url=base_url)
        self.max_steps = max_steps
        self.temperature = temperature

    def _chat(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def run(
        self,
        question: str | dict[str, Any],
        *,
        db_path: str | None = None,
        schema_path: str | None = None,
        max_steps: int | None = None,
        temperature: float | None = None,
    ):
        max_steps = self.max_steps if max_steps is None else max_steps
        temperature = self.temperature if temperature is None else temperature

        # Allow passing either a dict-style question payload or a plain string.
        # If a dict is provided (e.g., from CSpider converted jsonl), we extract routing fields.
        if isinstance(question, dict):
            payload = question
            question = str(payload.get("question") or "")
            db_path = payload.get("db_path")
            schema_path = payload.get("schema_path")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n"
                    "记住：第一步必须输出 [SCHEMA]。\n"
                    "在至少有一次 SQL 成功执行（ok=True）之前，不要输出 [ANSWER]。\n"
                    "如果 SQL 执行失败，请继续输出新的 [SQL] 进行纠错（必要时可再输出 [SCHEMA] 查看表结构）。"
                ),
            },
        ]

        trace: list[dict[str, Any]] = []
        sql_history: list[str] = []
        last_sql = ""
        last_error: str | None = None
        last_answer: str | None = None

        for step_idx in range(max_steps):
            model_text = self._chat(messages, temperature=temperature, max_tokens=256)
            action, payload = _parse_action(model_text)

            messages.append({"role": "assistant", "content": model_text})

            if action == "schema":
                schema_text = show_schema(db_path=db_path, schema_path=schema_path)
                obs = _render_observation_schema(schema_text)
                messages.append({"role": "user", "content": obs})
                trace.append(
                    {
                        "step": step_idx,
                        "action": "SCHEMA",
                        "model": model_text,
                        "schema_path": schema_path,
                        "db_path": db_path,
                    }
                )
                continue

            if action == "sql":
                sql = payload.strip()
                last_sql = sql
                sql_history.append(sql)
                exec_out = execute_sql_dict(sql, db_path=db_path)

                if exec_out.get("ok"):
                    last_error = None
                    last_answer = rows_to_answer(exec_out.get("rows") or [])
                else:
                    last_error = exec_out.get("error")

                obs = _render_observation_sql(exec_out)
                messages.append({"role": "user", "content": obs})
                trace.append(
                    {
                        "step": step_idx,
                        "action": "SQL",
                        "sql": sql,
                        "ok": bool(exec_out.get("ok")),
                        "error": exec_out.get("error"),
                        "columns": exec_out.get("columns"),
                        "rows": exec_out.get("rows"),
                        "truncated": exec_out.get("truncated"),
                        "answer": last_answer if exec_out.get("ok") else None,
                        "model": model_text,
                    }
                )
                continue

            if action == "answer":
                answer = (payload or "").strip()
                if not answer and last_answer is not None:
                    answer = last_answer
                if last_answer is None:
                    err_hint = f"\n最近一次 SQL 错误: {last_error}" if last_error else ""
                    invalid_obs = (
                        "Observation:\nError: 在没有任何成功 SQL 执行之前禁止输出 [ANSWER]。"
                        "请输出 [SQL] 修正查询（或 [SCHEMA] 查看数据库结构）。"
                        f"{err_hint}"
                    )
                    messages.append({"role": "user", "content": invalid_obs})
                    trace.append(
                        {
                            "step": step_idx,
                            "action": "INVALID",
                            "model": model_text,
                            "error": invalid_obs,
                            "invalid_type": "answer_before_sql_ok",
                        }
                    )
                    continue

                trace.append({"step": step_idx, "action": "ANSWER", "answer": answer, "model": model_text})
                return {
                    "ok": True,
                    "sql": last_sql,
                    "sql_history": sql_history,
                    "error": None,
                    "answer": answer,
                    "trace": trace,
                    "db_path": db_path,
                    "schema_path": schema_path,
                }

            # Invalid action format: provide a strict reminder and continue.
            invalid_obs = "Observation:\nError: Invalid action format. Output exactly one of [SCHEMA], [SQL] <query>, or [ANSWER] <answer>."
            messages.append(
                {
                    "role": "user",
                    "content": invalid_obs,
                }
            )
            trace.append({"step": step_idx, "action": "INVALID", "model": model_text, "error": invalid_obs})

        # Max steps exceeded: fall back to last computed answer if any.
        if last_answer is not None:
            return {
                "ok": True,
                "sql": last_sql,
                "sql_history": sql_history,
                "error": None,
                "answer": last_answer,
                "trace": trace,
                "db_path": db_path,
                "schema_path": schema_path,
            }

        return {
            "ok": False,
            "sql": last_sql,
            "sql_history": sql_history,
            "error": f"Max steps ({max_steps}) exceeded without a valid answer.",
            "answer": "",
            "trace": trace,
            "db_path": db_path,
            "schema_path": schema_path,
        }

    def answer(self, question: str, *, db_path: str | None = None, schema_path: str | None = None):
        return self.run(question, db_path=db_path, schema_path=schema_path)
