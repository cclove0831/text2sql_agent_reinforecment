import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "chinook.db"

def get_schema_text(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    tables = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    ).fetchall()
    lines = []
    for (tname,) in tables:
        cols = cur.execute(f"PRAGMA table_info('{tname}')").fetchall()
        col_str = ", ".join([f"{c[1]}:{c[2]}" for c in cols])  # (cid, name, type, notnull, dflt, pk)
        lines.append(f"Table {tname}({col_str})")
    return "\n".join(lines)

def execute_sql(conn: sqlite3.Connection, sql: str, limit: int = 20) -> str:
    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchmany(limit)
        col_names = [d[0] for d in (cur.description or [])]
        return f"OK\nColumns: {col_names}\nRows: {rows}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    try:
        schema = get_schema_text(conn)
        print("=== SCHEMA (first 40 lines) ===")
        sch_lines = schema.splitlines()
        print("\n".join(sch_lines[:40]))
        print("\n=== TEST QUERIES ===")
        print(execute_sql(conn, "SELECT COUNT(*) AS n FROM Artist;"))
        print(execute_sql(conn, "SELECT Name FROM Artist ORDER BY Name LIMIT 5;"))
    finally:
        conn.close()

if __name__ == "__main__":
    main()
