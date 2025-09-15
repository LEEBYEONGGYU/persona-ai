import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# 경로 설정
BASE_MODEL = "beomi/KoAlpaca-Polyglot-5.8B"
LORA_ADAPTER = "./lora_bizntc_only300"
PROMPT_FILE = "./eval_prompts.jsonl"
MAX_NEW_TOKENS = 128

# 모델 로드
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, LORA_ADAPTER)
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
tok.pad_token = tok.eos_token

# 후처리 함수 (선택)
def clean_response(resp: str) -> str:
    resp = re.sub(r"</?[^>]+>", "", resp)
    resp = re.sub(r"(Company|Location|Date|To):.*", "", resp)
    return resp.strip()

# 응답 생성 함수
def generate_response(prompt: str) -> str:
    full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tok(full_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if k != "token_type_ids"}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id
        )

    decoded = tok.decode(out[0], skip_special_tokens=False)

    if full_prompt in decoded:
        raw_resp = decoded.split(full_prompt)[-1].strip()
    else:
        raw_resp = decoded.strip()

    return clean_response(raw_resp)

# 평가 시작
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompts = [json.loads(line.strip()) for line in f]

print("📊 LoRA 응답 평가 시작\n")

for item in tqdm(prompts):
    instr = item["instruction"]
    resp = generate_response(instr)
    print(f"🧾 질문: {instr}\n📢 답변: {resp}\n{'-'*60}")