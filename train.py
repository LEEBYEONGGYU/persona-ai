import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
DATA_PATH  = "./sample_augmented.jsonl"
OUT_DIR    = "./lora_bizntc_only300"

MAX_LEN    = 512
EPOCHS     = 3
BATCH_SIZE = 4
GRAD_ACCUM = 2
LR         = 2e-5

# 1. Load dataset
raw = load_dataset("json", data_files={"train": DATA_PATH})
split = raw["train"].train_test_split(test_size=0.1, seed=42)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
tok.pad_token = tok.eos_token
tok.padding_side = "right"

def preprocess(batch):
    X = tok(batch["instruction"], truncation=True, padding="max_length", max_length=MAX_LEN)
    Y_ids = tok(batch["output"], truncation=True, padding="max_length", max_length=MAX_LEN)["input_ids"]
    Y = [[(t if t != tok.pad_token_id else -100) for t in seq] for seq in Y_ids]
    X["labels"] = Y
    return X

ds = split.map(preprocess, batched=True, remove_columns=split["train"].column_names)

# 2. Load model in 4bit
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
base.config.use_cache = False
base = prepare_model_for_kbit_training(base)

# 3. Detect LoRA modules
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

lora = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=targets,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base, lora)
model.print_trainable_parameters()

# 4. Train
args = TrainingArguments(
    output_dir="./results_bizntc_only300",
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
    warmup_steps=10,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
)

trainer.train()
trainer.model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print(f"✅ LoRA saved to {OUT_DIR}")