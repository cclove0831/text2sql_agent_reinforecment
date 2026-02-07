import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, TaskType, get_peft_model


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


def default_train_path() -> Path:
    cspider_train = Path("data/cspider_train.jsonl")
    if cspider_train.exists():
        return cspider_train
    return Path("data/train.jsonl")


def _get_schema_text(schema_path: str | None, *, max_lines: int) -> str:
    if not schema_path:
        return ""
    p = _resolve_repo_path(_normalize_rel_path(schema_path) or schema_path)
    if not p.exists():
        return f"Error: schema not found: {p}"
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]
    return "\n".join(lines)


def _apply_chat_template(
    tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool, return_tensors: str | None
):
    # transformers chat models expose apply_chat_template; fall back to a simple format.
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


@dataclass(frozen=True)
class SFTExample:
    question: str
    schema_path: str | None
    gt_sql: str


class CSpiderSFTDataset(Dataset):
    def __init__(
        self,
        *,
        data_path: Path,
        tokenizer: Any,
        max_seq_len: int,
        max_schema_lines: int,
        system_prompt: str,
    ):
        self.items_raw = load_jsonl(data_path)
        self.examples: list[SFTExample] = []
        for ex in self.items_raw:
            q = (ex.get("question") or "").strip()
            gt_sql = (ex.get("gt_sql") or "").strip()
            if not q or not gt_sql:
                continue
            self.examples.append(
                SFTExample(
                    question=q,
                    schema_path=ex.get("schema_path"),
                    gt_sql=gt_sql,
                )
            )

        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.max_schema_lines = int(max_schema_lines)
        self.system_prompt = system_prompt
        self._schema_cache: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def _schema_text(self, schema_path: str | None) -> str:
        if not schema_path:
            return ""
        key = _normalize_rel_path(schema_path) or schema_path
        cached = self._schema_cache.get(key)
        if cached is not None:
            return cached
        text = _get_schema_text(schema_path, max_lines=self.max_schema_lines)
        self._schema_cache[key] = text
        return text

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        schema_text = self._schema_text(ex.schema_path)

        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Question: {ex.question}\n\nSchema:\n{schema_text}\n\nOutput one SQL.",
            },
        ]

        prompt_ids = _apply_chat_template(
            self.tokenizer, prompt_messages, add_generation_prompt=True, return_tensors="pt"
        )[0].tolist()

        completion = f"[SQL] {ex.gt_sql}\n"
        completion_ids = self.tokenizer(
            completion,
            add_special_tokens=False,
            return_tensors=None,
        ).input_ids

        # Always end with EOS if available, so the model learns to stop.
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            completion_ids = list(completion_ids) + [int(eos_id)]

        input_ids = prompt_ids + list(completion_ids)
        labels = [-100] * len(prompt_ids) + list(completion_ids)
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]
            attention_mask = attention_mask[: self.max_seq_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DataCollatorForCausalLM:
    def __init__(self, *, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids_batch: list[list[int]] = []
        attn_batch: list[list[int]] = []
        labels_batch: list[list[int]] = []

        for f in features:
            input_ids = list(f["input_ids"])
            attn = list(f["attention_mask"])
            labels = list(f["labels"])

            pad = max_len - len(input_ids)
            if pad > 0:
                input_ids = input_ids + [self.pad_token_id] * pad
                attn = attn + [0] * pad
                labels = labels + [-100] * pad

            input_ids_batch.append(input_ids)
            attn_batch.append(attn)
            labels_batch.append(labels)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attn_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT warmup for Text2SQL (CSpider jsonl).")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, default=str(default_train_path()))
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_schema_lines", type=int, default=120)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA.",
    )

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_path = _resolve_repo_path(_normalize_rel_path(args.train_path) or args.train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"train_path not found: {train_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token_id is None:
        # Common for decoder-only LMs.
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

    dataset = CSpiderSFTDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_seq_len=int(args.max_seq_len),
        max_schema_lines=int(args.max_schema_lines),
        system_prompt=str(args.system_prompt),
    )

    collator = DataCollatorForCausalLM(pad_token_id=int(tokenizer.pad_token_id))

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(args.num_train_epochs),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        warmup_ratio=float(args.warmup_ratio),
        weight_decay=float(args.weight_decay),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
        remove_unused_columns=False,
        report_to="none",
        seed=int(args.seed),
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
