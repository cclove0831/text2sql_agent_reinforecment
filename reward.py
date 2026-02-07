import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chinook.db"

_DISALLOWED_SQL = re.compile(
    r"\b(drop|delete|insert|update|alter|create|replace|truncate|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)


def normalize_answer(s: str) -> str:
    return (s or "").strip()


def reward_exact(pred: str, gt: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gt) else 0.0


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _resolve_db_path(db_path: str | Path | None) -> Path:
    if db_path is None:
        return DEFAULT_DB_PATH
    return _resolve_path(db_path)


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


def _sanitize_sql(sql: str) -> str:
    s = _strip_code_fences(sql)
    if s.upper().startswith("[SQL]"):
        s = s[5:].strip()
    return s


def _validate_readonly_sql(sql: str) -> str | None:
    s = (sql or "").strip()
    if not s:
        return "Empty SQL."

    lowered = s.lower()
    if not lowered.startswith(("select", "with")):
        return "Only SELECT/with queries are allowed."

    if _DISALLOWED_SQL.search(s):
        return "Unsafe SQL detected (write/pragma/vacuum/etc). Only read-only queries are allowed."

    semicolons = s.count(";")
    if semicolons > 1:
        return "Multiple statements are not allowed."
    if semicolons == 1 and not s.rstrip().endswith(";"):
        return "Semicolon is only allowed at the end of the query."

    return None


def _join_count(sql: str) -> int:
    return len(re.findall(r"\bjoin\b", sql or "", flags=re.IGNORECASE))


def _is_hallucination_error(error: str | None) -> bool:
    if not error:
        return False
    e = error.lower()
    return ("no such table" in e) or ("no such column" in e)


@dataclass(frozen=True)
class ExecSignature:
    ok: bool
    error: str | None
    row_count: int
    truncated: bool
    sig: tuple[int, int, int, int, int] | None  # (count, xor1, xor2, sum1, sum2)
    sample_rows: list[tuple]  # up to 5 rows for debugging


def _row_digest(row: tuple[Any, ...]) -> tuple[int, int]:
    data = repr(row).encode("utf-8", errors="replace")
    digest = hashlib.blake2b(data, digest_size=16).digest()
    a = int.from_bytes(digest[:8], "little", signed=False)
    b = int.from_bytes(digest[8:], "little", signed=False)
    return a, b


def exec_sql_signature(
    sql: str,
    *,
    db_path: str | Path | None = None,
    max_rows: int | None = None,
) -> ExecSignature:
    query = _sanitize_sql(sql)
    err = _validate_readonly_sql(query)
    if err:
        return ExecSignature(
            ok=False,
            error=err,
            row_count=0,
            truncated=False,
            sig=None,
            sample_rows=[],
        )

    resolved_db = _resolve_db_path(db_path)
    if not resolved_db.exists():
        return ExecSignature(
            ok=False,
            error=f"DB not found: {resolved_db}",
            row_count=0,
            truncated=False,
            sig=None,
            sample_rows=[],
        )

    xor1 = 0
    xor2 = 0
    sum1 = 0
    sum2 = 0
    count = 0
    truncated = False
    sample: list[tuple] = []

    try:
        with sqlite3.connect(str(resolved_db)) as conn:
            cur = conn.cursor()
            cur.execute(query)
            for row in cur:
                if max_rows is not None and count >= max_rows:
                    truncated = True
                    break
                count += 1
                if len(sample) < 5:
                    sample.append(row)
                a, b = _row_digest(row)
                xor1 ^= a
                xor2 ^= b
                sum1 = (sum1 + a) & ((1 << 64) - 1)
                sum2 = (sum2 + b) & ((1 << 64) - 1)
    except sqlite3.OperationalError as e:
        return ExecSignature(
            ok=False,
            error=f"sqlite3.OperationalError: {e}",
            row_count=0,
            truncated=False,
            sig=None,
            sample_rows=[],
        )
    except Exception as e:
        return ExecSignature(
            ok=False,
            error=f"{type(e).__name__}: {e}",
            row_count=0,
            truncated=False,
            sig=None,
            sample_rows=[],
        )

    return ExecSignature(
        ok=True,
        error=None,
        row_count=count,
        truncated=truncated,
        sig=(count, xor1, xor2, sum1, sum2),
        sample_rows=sample,
    )


def execution_match(
    pred_sql: str,
    gt_sql: str,
    *,
    db_path: str | Path | None = None,
    max_rows: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    pred = exec_sql_signature(pred_sql, db_path=db_path, max_rows=max_rows)
    gt = exec_sql_signature(gt_sql, db_path=db_path, max_rows=max_rows)

    match = (
        pred.ok
        and gt.ok
        and (pred.sig is not None)
        and (gt.sig is not None)
        and (pred.sig == gt.sig)
        and (not pred.truncated)
        and (not gt.truncated)
    )
    detail = {
        "pred_ok": pred.ok,
        "pred_error": pred.error,
        "pred_rows": pred.row_count,
        "pred_truncated": pred.truncated,
        "gt_ok": gt.ok,
        "gt_error": gt.error,
        "gt_rows": gt.row_count,
        "gt_truncated": gt.truncated,
    }
    return match, detail


def compute_reward(
    *,
    pred_sql: str | None,
    gt_sql: str,
    db_path: str | Path | None = None,
    trace: list[dict[str, Any]] | None = None,
    trace_steps: int | None = None,
    max_compare_rows: int | None = None,
    weight_exec: float = 0.65,
    weight_trace: float = 0.35,
) -> tuple[float, dict[str, Any]]:
    """
    Skill-inspired mixed reward for multi-step agents.

    We compute:
    - r_exec (hard): execution_match(pred_sql, gt_sql) -> +1 / -1
    - r_trace (shaping): based on the agent trace (schema-first, invalid actions, repeated SQL failures, hallucinations, etc.)

    Final reward:
        total = weight_exec * r_exec + weight_trace * r_trace
    """
    pred_sql = (pred_sql or "").strip() or None
    join_pred = _join_count(pred_sql or "")
    join_gt = _join_count(gt_sql or "")
    if trace_steps is None and trace is not None:
        trace_steps = len(trace)
    extra_steps = 0 if trace_steps is None else max(0, int(trace_steps) - 2)

    detail: dict[str, Any] = {
        "join_pred": join_pred,
        "join_gt": join_gt,
        "trace_steps": trace_steps,
        "max_compare_rows": max_compare_rows,
        "weight_exec": weight_exec,
        "weight_trace": weight_trace,
    }

    # ---- r_exec (hard correctness) ----
    if not pred_sql:
        r_exec = -1.0
        exec_detail = {"execution_match": False, "pred_error": "no_sql"}
    else:
        match, match_detail = execution_match(pred_sql, gt_sql, db_path=db_path, max_rows=max_compare_rows)
        r_exec = 1.0 if match else -1.0
        exec_detail = {**match_detail, "execution_match": match}

    detail.update(exec_detail)

    # ---- r_trace (shaping) ----
    # Keep shaping in [-1, 1] and smaller than the hard reward.
    schema_first = None
    invalid_count = 0
    sql_calls = 0
    sql_ok_count = 0
    sql_fail_count = 0
    hallucination_err_count = 0
    illegal_sql_count = 0
    answered = False

    if trace:
        schema_first = bool(trace and trace[0].get("action") == "SCHEMA")
        for step in trace:
            act = step.get("action")
            if act == "INVALID":
                invalid_count += 1
                continue
            if act == "SQL":
                sql_calls += 1
                sql_text = step.get("sql") or ""
                if _DISALLOWED_SQL.search(sql_text):
                    illegal_sql_count += 1
                ok = bool(step.get("ok"))
                if ok:
                    sql_ok_count += 1
                else:
                    sql_fail_count += 1
                    if _is_hallucination_error(step.get("error")):
                        hallucination_err_count += 1
            if act == "ANSWER":
                answered = True

    # Shaping terms (tuned for stability, not maximal reward hacking).
    # Positive shaping is intentionally small.
    r_trace = 0.0
    if schema_first is True:
        r_trace += 0.25
    elif schema_first is False:
        r_trace -= 0.25

    if sql_ok_count > 0:
        r_trace += 0.25
    else:
        r_trace -= 0.25

    if answered and sql_ok_count == 0:
        r_trace -= 0.25

    r_trace += -0.15 * min(3, invalid_count)
    r_trace += -0.05 * min(6, max(0, sql_calls - 1))
    r_trace += -0.10 * min(3, hallucination_err_count)
    r_trace += -0.20 * min(1, illegal_sql_count)
    r_trace += -0.05 * min(6, extra_steps)
    r_trace += -0.10 if (join_pred > join_gt) else 0.0

    # Clamp to [-1, 1] so weights are meaningful.
    r_trace = max(-1.0, min(1.0, r_trace))

    detail.update(
        {
            "r_exec": r_exec,
            "r_trace": r_trace,
            "schema_first": schema_first,
            "invalid_count": invalid_count,
            "sql_calls": sql_calls,
            "sql_ok_count": sql_ok_count,
            "sql_fail_count": sql_fail_count,
            "hallucination_err_count": hallucination_err_count,
            "illegal_sql_count": illegal_sql_count,
            "extra_steps": extra_steps,
        }
    )

    total = float(weight_exec) * r_exec + float(weight_trace) * r_trace
    detail["total"] = total
    return total, detail
