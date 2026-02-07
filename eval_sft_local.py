import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel

from reward import execution_match


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_SYSTEM = (
    "You are a Text-to-SQL assistant for SQLite.\n"
    "Given a user question and the database schema, output exactly one line:\n"
    "[SQL] <one SQLite SELECT/CTE query>\n"
    "Rules: only use tables/columns from the schema; read-only SELECT/WITH only."
)


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


def _extract_sql(model_text: str) -> str:
    t = _strip_code_fences(model_text)
    upper = t.upper()
    if "[SQL]" in upper:
        idx = upper.rfind("[SQL]")
        payload = t[idx + len("[SQL]") :].strip()
        payload = _strip_code_fences(payload)
        # Stop at other tags if they appear.
        up = payload.upper()
        for tag in ("[ANSWER]", "[SCHEMA]"):
            j = up.find(tag)
            if j >= 0:
                payload = payload[:j].strip()
                up = payload.upper()
        return payload.strip()

    maybe = t.strip()
    if maybe.lower().startswith(("select", "with")):
        return maybe
    return ""


def _get_schema_text(schema_path: str | None, *, max_lines: int) -> str:
    if not schema_path:
        return ""
    p = _resolve_repo_path(_normalize_rel_path(schema_path) or schema_path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]
    return "\n".join(lines)


@torch.inference_mode()
def generate_sql(
    model: Any,
    tokenizer: Any,
    *,
    system_prompt: str,
    question: str,
    schema_text: str,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Question: {question}\n\nSchema:\n{schema_text}\n\nOutput one SQL.",
        },
    ]
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
    sql = _extract_sql(text)
    return sql, text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SFT(+LoRA) one-shot Text2SQL (local HF model).")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional LoRA adapter (outputs/sft_lora).")
    parser.add_argument("--data_path", type=str, default=str(default_eval_path()))
    parser.add_argument("--badcase_path", type=str, default="data/badcase_eval_sft_local.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 means all examples.")
    parser.add_argument("--print_every", type=int, default=20)

    parser.add_argument("--max_schema_lines", type=int, default=120)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_compare_rows", type=int, default=1000, help="-1 means fetch all rows.")

    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM)
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
    n_valid = 0
    n_ex = 0
    n_logic_err = 0

    for ex in items:
        total += 1
        question = (ex.get("question") or "").strip()
        schema_text = _get_schema_text(ex.get("schema_path"), max_lines=int(args.max_schema_lines))
        pred_sql, pred_text = generate_sql(
            model,
            tokenizer,
            system_prompt=str(args.system_prompt),
            question=question,
            schema_text=schema_text,
            device=device,
            max_new_tokens=int(args.max_new_tokens),
        )

        match, detail = execution_match(
            pred_sql,
            (ex.get("gt_sql") or "").strip(),
            db_path=_normalize_rel_path(ex.get("db_path")) if ex.get("db_path") else None,
            max_rows=max_compare_rows,
        )

        valid = bool(detail.get("pred_ok"))
        if valid:
            n_valid += 1
        if match:
            n_ex += 1
        elif valid:
            n_logic_err += 1

        if (not valid) or (not match):
            bad = {
                "id": ex.get("id"),
                "db_id": ex.get("db_id"),
                "db_path": ex.get("db_path"),
                "schema_path": ex.get("schema_path"),
                "question": ex.get("question"),
                "gt_sql": ex.get("gt_sql"),
                "pred_sql": pred_sql,
                "pred_text": pred_text,
                "execution_match": match,
                "execution_detail": detail,
            }
            bad_f.write(json.dumps(bad, ensure_ascii=False) + "\n")

        if int(args.print_every) > 0 and (total % int(args.print_every) == 0):
            print(
                f"[{total}/{len(items)}] EX={n_ex/total:.3f} valid={n_valid/total:.3f} logic_err={n_logic_err/total:.3f}"
            )

    bad_f.close()

    ex_rate = n_ex / total if total else 0.0
    valid_rate = n_valid / total if total else 0.0
    logic_err_rate = n_logic_err / total if total else 0.0

    print("\n=== Eval Summary (SFT Local) ===")
    print(f"Data: {data_path}")
    print(f"Total: {total}")
    print(f"Execution Accuracy (EX): {n_ex}/{total} = {ex_rate:.4f}")
    print(f"Valid SQL Rate: {n_valid}/{total} = {valid_rate:.4f}")
    print(f"Logic Error Rate: {n_logic_err}/{total} = {logic_err_rate:.4f}")
    print(f"Badcases saved to: {badcase_path}")


if __name__ == "__main__":
    main()

