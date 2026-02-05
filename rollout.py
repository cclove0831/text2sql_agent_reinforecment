import argparse
import json
from pathlib import Path

from agent_llm import Text2SQLAgent
from reward import compute_reward


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def default_train_path() -> Path:
    cspider_train = Path("data/cspider_train.jsonl")
    if cspider_train.exists():
        return cspider_train
    return Path("data/train.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Collect multi-sample rollouts and rewards (remote API).")
    parser.add_argument("--data_path", type=str, default=str(default_train_path()))
    parser.add_argument("--out_path", type=str, default="data/rollout.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 means all examples.")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--include_trace", action="store_true")
    parser.add_argument("--include_completion", action="store_true")
    parser.add_argument(
        "--max_compare_rows",
        type=int,
        default=1000,
        help="Max rows for reward execution-compare (-1 means fetch all).",
    )

    parser.add_argument("--model_name", type=str, default="qwen2.5-7b-instruct")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()

    max_compare_rows = None if args.max_compare_rows < 0 else int(args.max_compare_rows)

    data_path = Path(args.data_path)
    data = load_jsonl(data_path)
    if args.limit > 0:
        data = data[: args.limit]

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    agent = Text2SQLAgent(
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        max_steps=args.max_steps,
        temperature=args.temperature,
    )

    n_written = 0
    with out_path.open(mode, encoding="utf-8") as f:
        for group_id, ex in enumerate(data, start=1):
            for sample_id in range(args.group_size):
                out = agent.run(
                    ex.get("question") or "",
                    db_path=ex.get("db_path"),
                    schema_path=ex.get("schema_path"),
                    temperature=args.temperature,
                    max_steps=args.max_steps,
                )

                trace = out.get("trace") or []
                reward, detail = compute_reward(
                    pred_sql=out.get("sql"),
                    gt_sql=ex.get("gt_sql") or "",
                    db_path=ex.get("db_path"),
                    trace_steps=len(trace),
                    max_compare_rows=max_compare_rows,
                )

                record = {
                    "id": ex.get("id"),
                    "group_id": group_id,
                    "sample_id": sample_id,
                    "db_id": ex.get("db_id"),
                    "db_path": ex.get("db_path"),
                    "schema_path": ex.get("schema_path"),
                    "question": ex.get("question"),
                    "gt_sql": ex.get("gt_sql"),
                    "pred_ok": out.get("ok"),
                    "pred_sql": out.get("sql"),
                    "pred_error": out.get("error"),
                    "pred_answer": out.get("answer"),
                    "reward": reward,
                    "reward_detail": detail,
                }

                if args.include_completion:
                    record["completion"] = "\n".join(
                        step.get("model", "") for step in trace if isinstance(step, dict) and step.get("model")
                    )

                if args.include_trace:
                    record["trace"] = trace

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

            if group_id % 10 == 0:
                print(f"[{group_id}/{len(data)}] wrote={n_written} -> {out_path}")

    print(f"\nDone. Wrote {n_written} rollouts -> {out_path}")


if __name__ == "__main__":
    main()

