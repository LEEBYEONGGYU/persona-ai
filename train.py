import os
import json
import random
import math
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# =======================
# Config
# =======================
MODEL_NAME   = os.environ.get("MODEL_NAME", "beomi/KoAlpaca-Polyglot-5.8B")
DATA_PATH    = os.environ.get("DATA_PATH", "./sample_augmented.jsonl")  # <-- change here if needed
OUT_DIR      = os.environ.get("OUT_DIR", "./lora")
RESULTS_DIR  = os.environ.get("RESULTS_DIR", "./results")

MAX_LEN      = int(os.environ.get("MAX_LEN", "512"))
EPOCHS       = float(os.environ.get("EPOCHS", "3"))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "8"))
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", "2"))
LR           = float(os.environ.get("LR", "2e-5"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "20"))
SEED         = int(os.environ.get("SEED", "42"))

# =======================
# Seed
# =======================
set_seed(SEED)

# =======================
# Load dataset
# =======================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH}")

raw = load_dataset("json", data_files={"train": DATA_PATH})
# Small eval split (10%)
split = raw["train"].train_test_split(test_size=0.1, seed=SEED)
train_ds = split["train"]
eval_ds  = split["test"]

# =======================
# Tokenizer
# =======================
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# Ensure pad token
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

# =======================
# Prompt template
# =======================
def build_prompt(instruction: str, user_input: Optional[str] = "") -> str:
    instr = (instruction or "").strip()
    uin   = (user_input or "").strip()
    if uin:
        return f"### Instruction:\n{instr}\n\n### Input:\n{uin}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{instr}\n\n### Response:\n"

# =======================
# Preprocess (masking + padding)
# =======================
def tokenize_and_mask(example):
    instruction = example.get("instruction", "").strip()
    output = (example.get("output") or "").strip()
    user_input = example.get("input", "")  # optional

    prompt = build_prompt(instruction, user_input)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tok(output, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:MAX_LEN]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LEN]

    pad_len = MAX_LEN - len(input_ids)
    if pad_len > 0:
        input_ids += [tok.pad_token_id] * pad_len
        labels    += [-100] * pad_len

    attention_mask = [0 if t == tok.pad_token_id else 1 for t in input_ids]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

train_ds = train_ds.map(tokenize_and_mask, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(tokenize_and_mask,  remove_columns=eval_ds.column_names)

print("🔎 Sample lengths:", len(train_ds[0]["input_ids"]), len(train_ds[0]["labels"]))

# =======================
# 4-bit base model
# =======================
bnb = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb,
    device_map="auto",
)
base.config.use_cache = False
base = prepare_model_for_kbit_training(base)

# =======================
# Pick LoRA target modules
# =======================
def pick_target_modules(model):
    names = [n for n,_ in model.named_modules()]
    if any("q_proj" in n for n in names) and any("v_proj" in n for n in names):
        return ["q_proj","v_proj"]
    if any("query_key_value" in n for n in names):
        return ["query_key_value"]
    if any("c_attn" in n for n in names) and any("c_proj" in n for n in names):
        return ["c_attn","c_proj"]
    return ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]

targets = pick_target_modules(base)
print("🔧 LoRA target modules:", targets)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)




model = get_peft_model(base, peft_config)  

model.print_trainable_parameters()

collator = DataCollatorForSeq2Seq(
    tokenizer=tok,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

args = TrainingArguments(
    output_dir=RESULTS_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    gradient_checkpointing=True,
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    report_to="none",
    optim="paged_adamw_8bit",
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
)

trainer.train()

model.print_trainable_parameters()
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)

print(f"✅ LoRA saved to {OUT_DIR}")
print(f"👉 Now load with: PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained('{MODEL_NAME}'), '{OUT_DIR}')")