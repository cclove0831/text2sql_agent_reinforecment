import re
import sqlite3
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "chinook.db"
SCHEMA_PATH = PROJECT_ROOT / "data" / "schema.txt"

MAX_RESULT_ROWS = 5  # Skill spec: truncate results if > 5 rows.

_DISALLOWED_SQL = re.compile(
    r"\b(drop|delete|insert|update|alter|create|replace|truncate|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)


def get_schema_text() -> str:
    return get_schema_text_for_db(db_path=None, schema_path=None)


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def get_schema_text_for_db(*, db_path: str | Path | None, schema_path: str | Path | None) -> str:
    if schema_path is not None:
        p = _resolve_path(schema_path)
        if not p.exists():
            return f"Error: schema not found: {p}"
        return p.read_text(encoding="utf-8")

    if db_path is None and SCHEMA_PATH.exists():
        return SCHEMA_PATH.read_text(encoding="utf-8")

    resolved_db = DB_PATH if db_path is None else _resolve_path(db_path)
    if not resolved_db.exists():
        return f"Error: DB not found: {resolved_db}"

    with sqlite3.connect(str(resolved_db)) as conn:
        cur = conn.cursor()
        tables = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        ).fetchall()
        lines = []
        for (table_name,) in tables:
            cols = cur.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            col_str = ", ".join([f"{c[1]}:{c[2]}" for c in cols])
            lines.append(f"Table {table_name}({col_str})")
        return "\n".join(lines)


def show_schema(db_path: str | Path | None = None, schema_path: str | Path | None = None) -> str:
    return get_schema_text_for_db(db_path=db_path, schema_path=schema_path)


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


def execute_sql_dict(
    query: str, db_path: str | Path | None = None, max_rows: int = MAX_RESULT_ROWS
) -> dict[str, Any]:
    sql = _sanitize_sql(query)

    resolved_db = DB_PATH if db_path is None else _resolve_path(db_path)
    if not resolved_db.exists():
        return {
            "ok": False,
            "error": f"DB not found: {resolved_db}",
            "rows": [],
            "columns": [],
            "truncated": False,
        }

    err = _validate_readonly_sql(sql)
    if err:
        return {"ok": False, "error": err, "rows": [], "columns": [], "truncated": False}

    try:
        with sqlite3.connect(str(resolved_db)) as conn:
            cur = conn.cursor()
            cur.execute(sql)
            columns = [d[0] for d in (cur.description or [])]
            rows = cur.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            rows = rows[:max_rows]
            return {
                "ok": True,
                "error": None,
                "rows": rows,
                "columns": columns,
                "truncated": truncated,
            }
    except sqlite3.OperationalError as e:
        return {"ok": False, "error": f"sqlite3.OperationalError: {e}", "rows": [], "columns": [], "truncated": False}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "rows": [], "columns": [], "truncated": False}


def format_sql_output(out: dict[str, Any]) -> str:
    if not out.get("ok"):
        return f"Error: {out.get('error')}"

    columns = out.get("columns") or []
    rows = out.get("rows") or []
    lines = ["OK", f"Columns: {columns}", f"Rows: {rows}"]
    if out.get("truncated"):
        lines.append("...(truncated)")
    return "\n".join(lines)


def execute_sql(query: str) -> str:
    """
    Skill spec: execute_sql(query: str) -> str
    - Read-only guard: only SELECT/with.
    - Never crash: returns error as string.
    - Truncate to first 5 rows and append ...(truncated).
    """
    out = execute_sql_dict(query, max_rows=MAX_RESULT_ROWS)
    return format_sql_output(out)
