import json
import sqlite3
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chinook.db"

from utils import rows_to_answer  # noqa: E402


def resolve_db_path(db_path: str | None) -> Path:
    if not db_path:
        return DEFAULT_DB_PATH
    p = Path(db_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def resolve_file_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def run_sql(conn: sqlite3.Connection, sql: str, max_rows: int | None):
    cur = conn.cursor()
    cur.execute(sql)
    if max_rows is None:
        rows = cur.fetchall()
    else:
        rows = cur.fetchmany(max_rows)
    return rows



def check_file(path: Path, *, max_rows: int | None):
    ok = 0
    total = 0

    conn_cache: dict[Path, sqlite3.Connection] = {}
    try:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            total += 1
            obj = json.loads(line)

            db_path = resolve_db_path(obj.get("db_path"))
            if not db_path.exists():
                print(f"FAIL {obj.get('id')} | db not found: {db_path}")
                continue

            conn = conn_cache.get(db_path)
            if conn is None:
                conn = sqlite3.connect(str(db_path))
                conn_cache[db_path] = conn

            try:
                rows = run_sql(conn, obj["gt_sql"], max_rows=max_rows)
                ans = rows_to_answer(rows)
                gt = obj.get("gt_answer")
                passed = True if gt is None else (ans == gt)
                status = "PASS" if passed else "FAIL"
                if gt is None:
                    print(f"{status} {obj['id']} | got={ans}")
                else:
                    print(f"{status} {obj['id']} | got={ans} | gt={gt}")
                if passed:
                    ok += 1
            except Exception as e:
                print(f"FAIL {obj.get('id')} | sql error: {type(e).__name__}: {e}")
    finally:
        for conn in conn_cache.values():
            conn.close()
    print(f"\n{path.name}: {ok}/{total} passed")

def main():
    parser = argparse.ArgumentParser(description="Validate jsonl datasets by executing gt_sql on the referenced DB.")
    default_paths: list[str] = []
    cspider_train = PROJECT_ROOT / "data" / "cspider_train.jsonl"
    cspider_dev = PROJECT_ROOT / "data" / "cspider_dev.jsonl"
    if cspider_train.exists():
        default_paths.append(str(cspider_train))
    if cspider_dev.exists():
        default_paths.append(str(cspider_dev))
    if not default_paths:
        default_paths = [str(PROJECT_ROOT / "data" / "train.jsonl"), str(PROJECT_ROOT / "data" / "eval.jsonl")]

    parser.add_argument(
        "paths",
        nargs="*",
        default=default_paths,
        help="Paths to jsonl files (default: CSpider jsonl if present, else data/train.jsonl data/eval.jsonl).",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=5,
        help="Max rows to fetch per query (default: 5). Use -1 to fetch all rows.",
    )
    args = parser.parse_args()
    max_rows = None if args.max_rows < 0 else int(args.max_rows)

    for p in args.paths:
        check_file(resolve_file_path(p), max_rows=max_rows)

if __name__ == "__main__":
    main()
