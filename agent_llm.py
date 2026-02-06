import os
from typing import Any

from openai import OpenAI

from tools import execute_sql_dict, format_sql_output, show_schema
from utils import rows_to_answer

SYSTEM = """You are a Text-to-SQL agent for an SQLite database.

You MUST follow a Thought/Action/Observation loop. On each turn, output exactly ONE action:

[SCHEMA]
[SQL] <one SQLite SELECT/CTE query>
[ANSWER] <final answer>

Formatting rules:
- `[SCHEMA]` has NO argument. Output exactly `[SCHEMA]` and nothing else.
- Any SQL statement MUST be under `[SQL]` (not under `[SCHEMA]`).
- SQL may span multiple lines, but must be a single statement and must start with SELECT or WITH.
- You may include an optional <think>...</think> block. Outside <think>, output only the action token plus its payload.

Operational rules:
- Step 1 MUST be [SCHEMA]. If you have not seen the schema in this conversation, do [SCHEMA] now.
- Use ONLY tables/columns that appear in the schema observation. Never invent names.
- Only read-only SQL: SELECT or WITH ... SELECT. No PRAGMA and no writes (INSERT/UPDATE/DELETE/DROP/ALTER/VACUUM).
- Prefer explicit column lists; avoid SELECT * unless necessary for debugging.
- If the Observation starts with "Error:", fix the SQL and try again.

Value grounding rules (important):
- If the question filters by a string value that may vary in spelling/language/casing (e.g., "France" vs a translated name),
  first ground it by inspecting actual values, then use the exact value you observed, e.g.:
  - [SQL] SELECT DISTINCT <col> FROM <table> LIMIT 20
  - [SQL] SELECT <col> FROM <table> WHERE <col> LIKE '%keyword%' LIMIT 20
- If your query returns empty unexpectedly, re-check joins and grounded values before answering.

Common patterns:
- Superlatives (most/highest/lowest/first/last): ORDER BY ... LIMIT 1.
- Counting: COUNT(*) or COUNT(DISTINCT col) when asked for unique counts.
- Unique lists: DISTINCT.
- Use JOINs only when needed; join on foreign keys / id columns shown in the schema.

Answering:
- When a SQL runs successfully, the Observation includes an `Answer:` line computed from the first rows.
- When ready, output `[ANSWER]` followed by that exact `Answer:` value (no extra words).
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
                "content": f"Question: {question}\n\nRemember: start by outputting [SCHEMA].",
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

                ok = last_answer is not None
                trace.append({"step": step_idx, "action": "ANSWER", "answer": answer, "model": model_text})
                return {
                    "ok": ok,
                    "sql": last_sql,
                    "sql_history": sql_history,
                    "error": None if ok else (last_error or "No successful SQL executed."),
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
