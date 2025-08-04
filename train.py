import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ======================
# 1. 데이터셋 로드
# ======================
dataset = load_dataset("json", data_files={
    "train": "./sample_augmented.jsonl"
})

# ======================
# 2. 토크나이저
# ======================
model_name = "nlpai-lab/kullm-polyglot-5.8b-v2"  # ✅ 기존 모델 유지
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def preprocess_function(examples):
    inputs = [
        f"Instruction: {instr}\nInput: {inp}" if inp else f"Instruction: {instr}"
        for instr, inp in zip(examples["instruction"], examples["input"])
    ]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=256)  # ✅ 길이 절반
    labels = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=256)["input_ids"]

    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels
    ]
    model_inputs["labels"] = labels
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# ======================
# 3. 4bit 양자화 설정
# ======================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16  # ✅ float16 → bfloat16
)

# ======================
# 4. 모델 로드 및 k-bit 준비
# ======================
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",  # ✅ CPU Offload 허용
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# ======================
# 5. LoRA 설정 (경량화)
# ======================
peft_config = LoraConfig(
    r=2,  # ✅ rank 크게 축소
    lora_alpha=8,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

# ======================
# 6. Custom Trainer
# ======================
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss

# ======================
# 7. 학습 인자
# ======================
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # ✅ VRAM 절약
    num_train_epochs=4,
    learning_rate=3e-4,
    fp16=True,
    save_total_limit=2,
    logging_steps=10,
    save_steps=100,
    warmup_steps=10,
    weight_decay=0.01,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

# ======================
# 8. Trainer 실행
# ======================
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
)

# ======================
# 9. 학습 시작
# ======================
trainer.train()

# ======================
# 10. LoRA 어댑터 저장
# ======================
trainer.model.save_pretrained("./lora_polyglot5.8b_vram")
tokenizer.save_pretrained("./lora_polyglot5.8b_vram")
