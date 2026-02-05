import argparse
import json
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import rows_to_answer  # noqa: E402
DEFAULT_CSPIDER_DIR = PROJECT_ROOT / "data" / "full_CSpider" / "full_CSpider" / "CSpider"
DEFAULT_SCHEMA_DIR = PROJECT_ROOT / "data" / "cspider_schema"


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_sqlite_path(db_id: str, db_dir: Path) -> Path:
    candidate = db_dir / db_id / f"{db_id}.sqlite"
    if candidate.exists():
        return candidate

    matches = list((db_dir / db_id).glob("*.sqlite"))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(f"Cannot locate sqlite db for db_id={db_id!r} under {db_dir / db_id}")


def dump_schema(db_path: Path) -> str:
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        tables = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        ).fetchall()
        lines: list[str] = []
        for (table_name,) in tables:
            cols = cur.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            col_str = ", ".join([f"{c[1]}:{c[2]}" for c in cols])
            lines.append(f"Table {table_name}({col_str})")
        return "\n".join(lines)


def ensure_schema_file(db_id: str, db_path: Path, schema_dir: Path) -> Path:
    schema_dir.mkdir(parents=True, exist_ok=True)
    out = schema_dir / f"{db_id}.txt"
    if out.exists():
        return out
    out.write_text(dump_schema(db_path), encoding="utf-8")
    return out


def compute_gt_answer(db_path: Path, sql: str, max_rows: int | None) -> str:
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(max_rows) if max_rows is not None else cur.fetchall()
    return rows_to_answer(rows)


def convert_split(
    split: str,
    cspider_dir: Path,
    out_path: Path,
    schema_dir: Path,
    include_gt_answer: bool,
    max_gt_rows: int | None,
):
    db_dir = cspider_dir / "database"
    in_path = cspider_dir / f"{split}.json"
    data = json.loads(in_path.read_text(encoding="utf-8"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0

    with out_path.open("w", encoding="utf-8") as f:
        for idx, ex in enumerate(data):
            db_id = ex["db_id"]
            db_path = find_sqlite_path(db_id, db_dir)
            schema_path = ensure_schema_file(db_id, db_path, schema_dir)

            item = {
                "id": f"cspider_{split}_{idx:05d}",
                "db_id": db_id,
                "db_path": str(db_path.relative_to(PROJECT_ROOT)),
                "schema_path": str(schema_path.relative_to(PROJECT_ROOT)),
                "question": ex["question"],
                "gt_sql": (ex.get("query") or "").strip(),
            }

            if include_gt_answer:
                try:
                    item["gt_answer"] = compute_gt_answer(db_path, item["gt_sql"], max_rows=max_gt_rows)
                except Exception as e:
                    item["gt_answer"] = ""
                    item["gt_error"] = f"{type(e).__name__}: {e}"

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"[{split}] wrote {n_ok} examples -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert CSpider train/dev json into repo jsonl format.")
    parser.add_argument("--cspider_dir", type=str, default=str(DEFAULT_CSPIDER_DIR))
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--schema_dir", type=str, default=str(DEFAULT_SCHEMA_DIR))
    parser.add_argument("--splits", type=str, default="train,dev")
    parser.add_argument("--include_gt_answer", action="store_true")
    parser.add_argument(
        "--max_gt_rows",
        type=int,
        default=5,
        help="Only used with --include_gt_answer. Keep it aligned with tools.py truncation.",
    )
    args = parser.parse_args()

    cspider_dir = resolve_path(args.cspider_dir)
    out_dir = resolve_path(args.out_dir)
    schema_dir = resolve_path(args.schema_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        convert_split(
            split=split,
            cspider_dir=cspider_dir,
            out_path=out_dir / f"cspider_{split}.jsonl",
            schema_dir=schema_dir,
            include_gt_answer=bool(args.include_gt_answer),
            max_gt_rows=(None if not args.include_gt_answer else int(args.max_gt_rows)),
        )


if __name__ == "__main__":
    main()
