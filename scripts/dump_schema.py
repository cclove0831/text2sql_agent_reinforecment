import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "chinook.db"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "schema.txt"

def main():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    tables = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    ).fetchall()
    lines = []
    for (tname,) in tables:
        cols = cur.execute(f"PRAGMA table_info('{tname}')").fetchall()
        col_str = ", ".join([f"{c[1]}:{c[2]}" for c in cols])
        lines.append(f"Table {tname}({col_str})")
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote schema to: {OUT_PATH}")

if __name__ == "__main__":
    main()
