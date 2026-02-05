import argparse
import json
from pathlib import Path

from agent_llm import Text2SQLAgent
from reward import execution_match


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def default_eval_path() -> Path:
    cspider_dev = Path("data/cspider_dev.jsonl")
    if cspider_dev.exists():
        return cspider_dev
    return Path("data/eval.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Text2SQL agent on a jsonl dataset (remote API).")
    parser.add_argument("--data_path", type=str, default=str(default_eval_path()))
    parser.add_argument("--badcase_path", type=str, default="data/badcase_eval.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 means all examples.")
    parser.add_argument("--print_every", type=int, default=20)

    parser.add_argument("--model_name", type=str, default="qwen2.5-7b-instruct")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument(
        "--max_compare_rows",
        type=int,
        default=-1,
        help="Max rows to compare for execution match (-1 means fetch all).",
    )
    args = parser.parse_args()

    max_compare_rows = None if args.max_compare_rows < 0 else int(args.max_compare_rows)

    data_path = Path(args.data_path)
    data = load_jsonl(data_path)
    if args.limit > 0:
        data = data[: args.limit]

    badcase_path = Path(args.badcase_path)
    badcase_path.parent.mkdir(parents=True, exist_ok=True)
    badcase_f = badcase_path.open("w", encoding="utf-8")

    agent = Text2SQLAgent(
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        max_steps=args.max_steps,
        temperature=0.0,
    )

    total = 0
    n_valid = 0
    n_ex = 0
    n_logic_err = 0
    step_sum = 0
    sql_attempt_sum = 0

    for ex in data:
        total += 1
        out = agent.run(
            ex.get("question") or "",
            db_path=ex.get("db_path"),
            schema_path=ex.get("schema_path"),
            temperature=0.0,
            max_steps=args.max_steps,
        )

        trace = out.get("trace") or []
        step_sum += len(trace)
        sql_attempt_sum += sum(1 for t in trace if t.get("action") == "SQL")

        valid = bool(out.get("ok"))
        if valid:
            n_valid += 1

        ex_match = False
        ex_detail: dict | None = None
        if valid and out.get("sql"):
            ex_match, ex_detail = execution_match(
                out["sql"], ex.get("gt_sql") or "", db_path=ex.get("db_path"), max_rows=max_compare_rows
            )
            if ex_match:
                n_ex += 1
            else:
                n_logic_err += 1

        if (not valid) or (valid and not ex_match):
            bad = {
                "id": ex.get("id"),
                "db_id": ex.get("db_id"),
                "db_path": ex.get("db_path"),
                "schema_path": ex.get("schema_path"),
                "question": ex.get("question"),
                "gt_sql": ex.get("gt_sql"),
                "pred_sql": out.get("sql"),
                "pred_ok": out.get("ok"),
                "pred_error": out.get("error"),
                "pred_answer": out.get("answer"),
                "steps": len(trace),
                "sql_attempts": sum(1 for t in trace if t.get("action") == "SQL"),
                "execution_match": ex_match,
                "execution_detail": ex_detail,
                "trace": trace,
            }
            badcase_f.write(json.dumps(bad, ensure_ascii=False) + "\n")

        if args.print_every > 0 and (total % args.print_every == 0):
            print(
                f"[{total}/{len(data)}] EX={n_ex/total:.3f} valid={n_valid/total:.3f} logic_err={n_logic_err/total:.3f}"
            )

    badcase_f.close()

    ex_rate = n_ex / total if total else 0.0
    valid_rate = n_valid / total if total else 0.0
    logic_err_rate = n_logic_err / total if total else 0.0
    avg_steps = step_sum / total if total else 0.0
    avg_sql_attempts = sql_attempt_sum / total if total else 0.0

    print("\n=== Eval Summary ===")
    print(f"Data: {data_path}")
    print(f"Total: {total}")
    print(f"Execution Accuracy (EX): {n_ex}/{total} = {ex_rate:.4f}")
    print(f"Valid SQL Rate: {n_valid}/{total} = {valid_rate:.4f}")
    print(f"Logic Error Rate: {n_logic_err}/{total} = {logic_err_rate:.4f}")
    print(f"Avg Steps: {avg_steps:.2f}")
    print(f"Avg SQL Attempts: {avg_sql_attempts:.2f}")
    print(f"Badcases saved to: {badcase_path}")


if __name__ == "__main__":
    main()

