import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

base_model = "nlpai-lab/kullm-polyglot-5.8b-v2"
adapter_path = "./lora_polyglot5.8b_vram"

os.makedirs("./offload", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
    offload_folder="./offload"
)

model = PeftModel.from_pretrained(
    model,
    adapter_path,
    offload_dir="./offload"
)

prompts = ["누가 너를 만들었어?", "너의 역할은 뭐야?", "이병규님이 만든 앱은?"]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    inputs.pop("token_type_ids", None)  # ✅ 불필요한 키 제거
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2
    )
    print(f"Q: {prompt}")
    print(f"A: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")
