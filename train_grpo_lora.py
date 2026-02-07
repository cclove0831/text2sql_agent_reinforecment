import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from peft import LoraConfig, PeftModel, TaskType, get_peft_model

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


def default_train_path() -> Path:
    cspider_train = Path("data/cspider_train.jsonl")
    if cspider_train.exists():
        return cspider_train
    return Path("data/train.jsonl")


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
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=return_tensors is not None,
            return_tensors=return_tensors,
        )

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


def _encode_prompt_ids(tokenizer: Any, messages: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    ids = _apply_chat_template(tokenizer, messages, add_generation_prompt=True, return_tensors="pt")
    return ids.to(device)


@torch.no_grad()
def _generate_one(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, list[int]]:
    input_ids = _encode_prompt_ids(tokenizer, messages, device=device)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        return_dict_in_generate=True,
    )
    seq = out.sequences[0]
    gen_ids = seq[input_ids.shape[1] :].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return (text or "").strip(), gen_ids


def rollout_trajectory(
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
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    db_path = _normalize_rel_path(db_path) if db_path else db_path
    schema_path = _normalize_rel_path(schema_path) if schema_path else schema_path

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\n\nRemember: start by outputting [SCHEMA]."},
    ]

    trace: list[dict[str, Any]] = []
    last_sql = ""
    last_answer: str | None = None
    last_error: str | None = None

    for step_idx in range(int(max_steps)):
        prompt_snapshot = [dict(m) for m in messages]
        model_text, token_ids = _generate_one(
            model,
            tokenizer,
            messages,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        action, payload = parse_action(model_text)

        messages.append({"role": "assistant", "content": model_text})

        if action == "schema":
            schema_text = show_schema(db_path=db_path, schema_path=schema_path)
            obs = _render_observation_schema(schema_text)
            messages.append({"role": "user", "content": obs})
            trace.append(
                {
                    "step": step_idx,
                    "action": "SCHEMA",
                    "model": model_text,
                    "token_ids": token_ids,
                    "prompt_messages": prompt_snapshot,
                    "schema_path": schema_path,
                    "db_path": db_path,
                }
            )
            continue

        if action == "sql":
            sql = (payload or "").strip()
            last_sql = sql
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
                    "model": model_text,
                    "token_ids": token_ids,
                    "prompt_messages": prompt_snapshot,
                    "sql": sql,
                    "ok": bool(exec_out.get("ok")),
                    "error": exec_out.get("error"),
                    "columns": exec_out.get("columns"),
                    "rows": exec_out.get("rows"),
                    "truncated": exec_out.get("truncated"),
                    "answer": last_answer if exec_out.get("ok") else None,
                }
            )
            continue

        if action == "answer":
            answer = (payload or "").strip()
            if not answer and last_answer is not None:
                answer = last_answer
            trace.append(
                {
                    "step": step_idx,
                    "action": "ANSWER",
                    "model": model_text,
                    "token_ids": token_ids,
                    "prompt_messages": prompt_snapshot,
                    "answer": answer,
                }
            )
            break

        invalid_obs = (
            "Observation:\nError: Invalid action format. Output exactly one of "
            "[SCHEMA], [SQL] <query>, or [ANSWER] <answer>."
        )
        messages.append({"role": "user", "content": invalid_obs})
        trace.append(
            {
                "step": step_idx,
                "action": "INVALID",
                "model": model_text,
                "token_ids": token_ids,
                "prompt_messages": prompt_snapshot,
                "error": invalid_obs,
            }
        )

    ok = last_answer is not None
    return {
        "ok": ok,
        "pred_sql": last_sql,
        "error": None if ok else (last_error or "No successful SQL executed."),
        "trace": trace,
        "db_path": db_path,
        "schema_path": schema_path,
    }


def _logprob_of_tokens(
    model: Any,
    *,
    input_ids: torch.Tensor,
    prompt_len: int,
    gen_ids: torch.Tensor,
) -> torch.Tensor:
    # input_ids = [prompt_ids, gen_ids]
    logits = model(input_ids=input_ids).logits  # [B, T, V]
    # For each generated token at position t, use logits from position t-1.
    start = prompt_len - 1
    end = input_ids.shape[1] - 1
    logits_gen = logits[:, start:end, :]  # [B, gen_len, V]
    logp = torch.log_softmax(logits_gen, dim=-1)
    gathered = torch.gather(logp, dim=-1, index=gen_ids.unsqueeze(-1)).squeeze(-1)  # [B, gen_len]
    return gathered.sum(dim=-1)  # [B]


def policy_logprob_sum(
    model: Any,
    tokenizer: Any,
    trace: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    total = torch.tensor(0.0, device=device)
    n_tokens = 0
    for step in trace:
        token_ids = step.get("token_ids") or []
        prompt_messages = step.get("prompt_messages") or []
        if not token_ids or not prompt_messages:
            continue
        prompt_ids = _encode_prompt_ids(tokenizer, prompt_messages, device=device)
        gen_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        input_ids = torch.cat([prompt_ids, gen_ids], dim=1)
        lp = _logprob_of_tokens(model, input_ids=input_ids, prompt_len=prompt_ids.shape[1], gen_ids=gen_ids)
        total = total + lp[0]
        n_tokens += int(gen_ids.numel())
    return total, n_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO + LoRA for multi-step Text2SQL agent (local HF model).")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional LoRA adapter to start from (SFT warmup).")
    parser.add_argument("--train_path", type=str, default=str(default_train_path()))
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_compare_rows", type=int, default=1000)

    parser.add_argument("--weight_exec", type=float, default=0.65)
    parser.add_argument("--weight_trace", type=float, default=0.35)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_groups", type=int, default=0, help="0 means all examples.")
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--kl_beta", type=float, default=0.0)
    parser.add_argument("--ref_model_name_or_path", type=str, default=None)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_system_prompt() -> str:
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


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    train_path = _resolve_repo_path(_normalize_rel_path(args.train_path) or args.train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"train_path not found: {train_path}")
    items = load_jsonl(train_path)
    if int(args.max_groups) > 0:
        items = items[: int(args.max_groups)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if bool(args.bf16):
        dtype = torch.bfloat16
    elif bool(args.fp16):
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )

    if bool(args.gradient_checkpointing):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        target_modules = [s.strip() for s in (args.lora_target_modules or "").split(",") if s.strip()]
        lora_cfg = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ref_model = None
    if float(args.kl_beta) > 0.0:
        if not args.ref_model_name_or_path:
            # Avoid silently doubling memory by deepcopy; require an explicit ref model path.
            raise ValueError("--kl_beta > 0 requires --ref_model_name_or_path")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=bool(args.trust_remote_code),
        ).to(device)
        ref_model.eval()

    optimizer = AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    total_updates = int(args.num_epochs) * len(items)
    total_updates = max(1, math.ceil(total_updates / max(1, int(args.grad_accum_steps))))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_steps),
        num_training_steps=int(total_updates),
    )

    system_prompt = _load_system_prompt()
    max_compare_rows = None if int(args.max_compare_rows) < 0 else int(args.max_compare_rows)

    step = 0
    model.train()

    for epoch in range(int(args.num_epochs)):
        for group_idx, ex in enumerate(items, start=1):
            question = (ex.get("question") or "").strip()
            if not question:
                continue
            db_path = ex.get("db_path")
            schema_path = ex.get("schema_path")
            gt_sql = (ex.get("gt_sql") or "").strip()
            if not gt_sql:
                continue

            model.eval()
            samples: list[dict[str, Any]] = []
            rewards: list[float] = []

            for _ in range(int(args.group_size)):
                traj = rollout_trajectory(
                    model,
                    tokenizer,
                    system_prompt=system_prompt,
                    question=question,
                    db_path=db_path,
                    schema_path=schema_path,
                    device=device,
                    max_steps=int(args.max_steps),
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                )
                r, detail = compute_reward(
                    pred_sql=traj.get("pred_sql"),
                    gt_sql=gt_sql,
                    db_path=_normalize_rel_path(db_path) if db_path else None,
                    trace=traj.get("trace"),
                    trace_steps=len(traj.get("trace") or []),
                    max_compare_rows=max_compare_rows,
                    weight_exec=float(args.weight_exec),
                    weight_trace=float(args.weight_trace),
                )
                traj["reward"] = float(r)
                traj["reward_detail"] = detail
                samples.append(traj)
                rewards.append(float(r))

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            mean_r = rewards_t.mean()
            std_r = rewards_t.std(unbiased=False)
            adv = (rewards_t - mean_r) / (std_r + 1e-6)

            model.train()
            group_loss = torch.tensor(0.0, device=device)
            for i, traj in enumerate(samples):
                lp_sum, n_tokens = policy_logprob_sum(model, tokenizer, traj["trace"], device=device)
                denom = max(1, int(n_tokens))
                lp = lp_sum / float(denom)

                kl_term = torch.tensor(0.0, device=device)
                if ref_model is not None:
                    with torch.no_grad():
                        ref_lp_sum, ref_n = policy_logprob_sum(ref_model, tokenizer, traj["trace"], device=device)
                        ref_lp = ref_lp_sum / float(max(1, int(ref_n)))
                    kl_term = lp - ref_lp

                loss_i = (-adv[i].detach() * lp) + (float(args.kl_beta) * kl_term)
                group_loss = group_loss + loss_i

            group_loss = group_loss / float(max(1, int(args.group_size)))
            group_loss = group_loss / float(max(1, int(args.grad_accum_steps)))
            group_loss.backward()

            if group_idx % int(args.grad_accum_steps) == 0:
                if float(args.max_grad_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

            if int(args.log_every) > 0 and group_idx % int(args.log_every) == 0:
                ex_rate = sum(1 for s in samples if (s.get("reward_detail") or {}).get("execution_match")) / float(
                    max(1, len(samples))
                )
                avg_steps = sum(len(s.get("trace") or []) for s in samples) / float(max(1, len(samples)))
                print(
                    f"[epoch={epoch+1} group={group_idx}/{len(items)} step={step}] "
                    f"meanR={mean_r.item():.4f} stdR={std_r.item():.4f} ex@group={ex_rate:.3f} avg_steps={avg_steps:.2f}"
                )

            if int(args.save_every) > 0 and group_idx % int(args.save_every) == 0:
                save_dir = output_dir / f"checkpoint_group{group_idx:06d}"
                save_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(save_dir))
                tokenizer.save_pretrained(str(save_dir))

        # Save at end of epoch
        save_dir = output_dir / f"epoch{epoch+1}"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))


if __name__ == "__main__":
    main()
