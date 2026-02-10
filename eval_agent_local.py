import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel

from reward import compute_reward
from tools import execute_sql_dict, format_sql_output, show_schema
from utils import rows_to_answer


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_repo_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _normalize_rel_path(p: str | None) -> str | None:
    if p is None:
        return None
    return p.replace("\\", "/")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
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


def _load_system_prompt() -> str:
    # Keep this script independent from the remote API dependency (openai).
    # If agent_llm is importable, we reuse its SYSTEM prompt for consistency.
    try:
        import agent_llm

        s = getattr(agent_llm, "SYSTEM", None)
        if isinstance(s, str) and s.strip():
            return s
    except Exception:
        pass

    return (
        "You are a Text-to-SQL agent for an SQLite database.\n"
        "Follow a Thought/Action/Observation loop and output exactly ONE action per turn:\n"
        "[SCHEMA] or [SQL] <query> or [ANSWER] <answer>."
    )


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


def parse_action(text: str) -> tuple[str, str]:
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
        idx = upper.rfind("[SCHEMA]")
        after = t[idx + len("[SCHEMA]") :].strip()
        maybe = _strip_code_fences(after)
        if maybe.lower().lstrip().startswith(("select", "with")):
            return "sql", maybe
        return "schema", ""

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


def _apply_chat_template(
    tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool, return_tensors: str | None
):
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        # Keep output type stable across transformers versions by rendering to text first.
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        if return_tensors is None:
            return text
        return tokenizer(text, return_tensors=return_tensors, add_special_tokens=False).input_ids

    text_lines: list[str] = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = m.get("content") or ""
        if role == "system":
            text_lines.append(f"System:\n{content}\n")
        elif role == "user":
            text_lines.append(f"User:\n{content}\n")
        else:
            text_lines.append(f"Assistant:\n{content}\n")
    if add_generation_prompt:
        text_lines.append("Assistant:\n")
    text = "\n".join(text_lines).strip() + "\n"
    if return_tensors is None:
        return text
    return tokenizer(text, return_tensors=return_tensors, add_special_tokens=True).input_ids


@torch.inference_mode()
def _generate_one_greedy(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    input_ids = _apply_chat_template(tokenizer, messages, add_generation_prompt=True, return_tensors="pt").to(device)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=1,
        pad_token_id=int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None,
        eos_token_id=int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        return_dict_in_generate=True,
    )
    seq = out.sequences[0]
    gen_ids = seq[input_ids.shape[1] :].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True) or ""
    return text.strip()


def run_agent_local(
    model: Any,
    tokenizer: Any,
    *,
    system_prompt: str,
    question: str,
    db_path: str | None,
    schema_path: str | None,
    device: torch.device,
    max_steps: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    db_path = _normalize_rel_path(db_path) if db_path else db_path
    schema_path = _normalize_rel_path(schema_path) if schema_path else schema_path

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
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
    last_answer: str | None = None
    last_error: str | None = None
    schema_calls = 0

    for step_idx in range(int(max_steps)):
        model_text = _generate_one_greedy(
            model,
            tokenizer,
            messages,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        action, payload = parse_action(model_text)

        messages.append({"role": "assistant", "content": model_text})

        if action == "schema":
            schema_calls += 1
            if schema_calls > 2:
                invalid_obs = "Observation:\nError: Schema 已经提供过多次。请直接输出 [SQL] 继续完成查询与纠错。"
                messages.append({"role": "user", "content": invalid_obs})
                trace.append(
                    {
                        "step": step_idx,
                        "action": "INVALID",
                        "model": model_text,
                        "error": invalid_obs,
                        "invalid_type": "too_many_schema_calls",
                    }
                )
                continue
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
            sql = (payload or "").strip()
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

            ok = True
            trace.append({"step": step_idx, "action": "ANSWER", "answer": answer, "model": model_text})
            return {
                "ok": ok,
                "sql": last_sql,
                "sql_history": sql_history,
                "error": None,
                "answer": answer,
                "trace": trace,
                "db_path": db_path,
                "schema_path": schema_path,
            }

        invalid_obs = (
            "Observation:\nError: Invalid action format. Output exactly one of [SCHEMA], [SQL] <query>, or [ANSWER] <answer>."
        )
        messages.append({"role": "user", "content": invalid_obs})
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-step Text2SQL agent (local HF model + optional LoRA adapter)."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional LoRA adapter (outputs/sft_lora or grpo).")
    parser.add_argument("--data_path", type=str, default=str(default_eval_path()))
    parser.add_argument("--badcase_path", type=str, default="data/badcase_eval_agent_local.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 means all examples.")
    parser.add_argument("--print_every", type=int, default=20)

    parser.add_argument("--system_prompt", type=str, default=_load_system_prompt())
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_compare_rows", type=int, default=1000, help="-1 means fetch all rows.")

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = None
    if bool(args.bf16):
        dtype = torch.bfloat16
    elif bool(args.fp16):
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    model.to(device)

    data_path = _resolve_repo_path(_normalize_rel_path(args.data_path) or args.data_path)
    items = load_jsonl(data_path)
    if int(args.limit) > 0:
        items = items[: int(args.limit)]

    badcase_path = _resolve_repo_path(_normalize_rel_path(args.badcase_path) or args.badcase_path)
    badcase_path.parent.mkdir(parents=True, exist_ok=True)
    bad_f = badcase_path.open("w", encoding="utf-8")

    max_compare_rows = None if int(args.max_compare_rows) < 0 else int(args.max_compare_rows)

    total = 0
    n_agent_ok = 0
    n_valid_sql = 0
    n_ex = 0
    n_logic_err = 0
    step_sum = 0
    sql_attempt_sum = 0
    n_no_sql = 0

    for ex in items:
        total += 1
        question = (ex.get("question") or "").strip()
        out = run_agent_local(
            model,
            tokenizer,
            system_prompt=str(args.system_prompt),
            question=question,
            db_path=ex.get("db_path"),
            schema_path=ex.get("schema_path"),
            device=device,
            max_steps=int(args.max_steps),
            max_new_tokens=int(args.max_new_tokens),
        )

        trace = out.get("trace") or []
        step_sum += len(trace)
        sql_attempt_sum += sum(1 for t in trace if t.get("action") == "SQL")

        agent_ok = bool(out.get("ok"))
        if agent_ok:
            n_agent_ok += 1

        gt_sql = (ex.get("gt_sql") or "").strip()
        pred_sql_last = (out.get("sql") or "").strip()
        if not pred_sql_last.strip():
            n_no_sql += 1

        ex_match = False
        exec_detail: dict[str, Any] | None = None
        valid_sql = False
        if gt_sql:
            _, exec_detail = compute_reward(
                pred_sql=pred_sql_last,
                gt_sql=gt_sql,
                db_path=_normalize_rel_path(ex.get("db_path")) if ex.get("db_path") else None,
                trace=trace,
                max_compare_rows=max_compare_rows,
            )
            valid_sql = bool(exec_detail.get("pred_ok")) if exec_detail else False
            if valid_sql:
                n_valid_sql += 1
            ex_match = bool(exec_detail.get("execution_match")) if exec_detail else False

        if ex_match:
            n_ex += 1
        elif valid_sql:
            n_logic_err += 1

        if (not agent_ok) or (agent_ok and not ex_match):
            pred_sql_used = exec_detail.get("pred_sql_used") if isinstance(exec_detail, dict) else None
            pred_sql_source = exec_detail.get("pred_sql_source") if isinstance(exec_detail, dict) else None
            bad = {
                "id": ex.get("id"),
                "db_id": ex.get("db_id"),
                "db_path": ex.get("db_path"),
                "schema_path": ex.get("schema_path"),
                "question": ex.get("question"),
                "gt_sql": ex.get("gt_sql"),
                "pred_sql_last": pred_sql_last,
                "pred_sql_used": pred_sql_used,
                "pred_sql_source": pred_sql_source,
                "pred_ok": out.get("ok"),
                "pred_error": out.get("error"),
                "pred_answer": out.get("answer"),
                "steps": len(trace),
                "sql_attempts": sum(1 for t in trace if t.get("action") == "SQL"),
                "execution_match": ex_match,
                "execution_detail": exec_detail,
                "trace": trace,
            }
            bad_f.write(json.dumps(bad, ensure_ascii=False) + "\n")

        if int(args.print_every) > 0 and (total % int(args.print_every) == 0):
            print(
                f"[{total}/{len(items)}] EX={n_ex/total:.3f} valid_sql={n_valid_sql/total:.3f} "
                f"agent_ok={n_agent_ok/total:.3f} logic_err={n_logic_err/total:.3f}"
            )

    bad_f.close()

    ex_rate = n_ex / total if total else 0.0
    valid_sql_rate = n_valid_sql / total if total else 0.0
    agent_ok_rate = n_agent_ok / total if total else 0.0
    no_sql_rate = n_no_sql / total if total else 0.0
    logic_err_rate = n_logic_err / total if total else 0.0
    avg_steps = step_sum / total if total else 0.0
    avg_sql_attempts = sql_attempt_sum / total if total else 0.0

    print("\n=== Eval Summary (Agent Local) ===")
    print(f"Data: {data_path}")
    print(f"Total: {total}")
    print(f"Execution Accuracy (EX): {n_ex}/{total} = {ex_rate:.4f}")
    print(f"Valid SQL Rate: {n_valid_sql}/{total} = {valid_sql_rate:.4f}")
    print(f"Agent OK Rate: {n_agent_ok}/{total} = {agent_ok_rate:.4f}")
    print(f"No SQL Rate: {n_no_sql}/{total} = {no_sql_rate:.4f}")
    print(f"Logic Error Rate: {n_logic_err}/{total} = {logic_err_rate:.4f}")
    print(f"Avg Steps: {avg_steps:.2f}")
    print(f"Avg SQL Attempts: {avg_sql_attempts:.2f}")
    print(f"Badcases saved to: {badcase_path}")


if __name__ == "__main__":
    main()
