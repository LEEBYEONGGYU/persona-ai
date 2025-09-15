# test.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE   = "beomi/KoAlpaca-Polyglot-5.8B"
ADAPT  = "./lora_bizntc_only300"

base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, ADAPT)

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token

prompt = """### Instruction:
2025년 3월 21일은 무슨 날이야?

### Response:
"""

inputs = tok(prompt, return_tensors="pt")
# 핵심: token_type_ids 제거 + CUDA 이동
inputs = {k: v.to(model.device) for k,v in inputs.items() if k != "token_type_ids"}

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tok.eos_token_id  # ✅ 끝내는 조건
    )
    decoded = tok.decode(out[0], skip_special_tokens=True)

print(decoded.strip())  # ✅ 앞뒤 공백 및 줄바꿈 제거
