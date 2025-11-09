from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from router_holiday import holiday_router_2025  # ⬅️ 단체휴무 라우터 불러오기
from card import lawcard_router
import asyncio

# 🔧 모델 로딩 (4bit + offload + eval + 속도 최적화)
BASE = "beomi/KoAlpaca-Polyglot-5.8B"
ADAPT = "./lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # 또는 bfloat16도 가능
)

print("🔧 모델 로딩 중...")


base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder="./offload"
)
model = PeftModel.from_pretrained(base_model, ADAPT)
model.eval()  # 추론 모드
tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.pad_token = tokenizer.eos_token
print("✅ 모델 로딩 완료")
print("✅ 적용된 LoRA 모듈:", model.modules_to_save)

# FastAPI 앱 구성
app = FastAPI()

# ✅ CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptInput(BaseModel):
    prompt: str
    max_new_tokens: int = 100  # 기본값 줄임
    top_p: float = 0.9
    temperature: float = 0.7

@app.post("/generate")
async def generate_text(data: PromptInput):
    prompt = data.prompt.strip()  # ⬅️ 먼저 선언
    print("📝 사용자 입력:", prompt)  # 👈 사용자 입력 로그

     # ✅ Step 1: 단체휴무 룰 우선 처리
     
    routed = holiday_router_2025(prompt)
    if routed is not None:
        print("✅ 단체휴무 룰로 응답됨")
        await asyncio.sleep(2)
        return {"response": routed}

    routed = lawcard_router(prompt)
    if routed is not None:
        print("✅ 법인카드 룰로 응답됨")
        await asyncio.sleep(2)
        return {"response": routed}


    formatted = f"### Instruction:\n{data.prompt.strip()}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if k != "token_type_ids"}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=data.max_new_tokens,
            do_sample=False,
            top_p=data.top_p,
            temperature=data.temperature,
            repetition_penalty=1.05,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True).strip()

    # 후처리
    if data.prompt.strip() in decoded:
        decoded = decoded.replace(data.prompt.strip(), "").strip()

    decoded = re.sub(r"^#+\s*(응답|Response|Instruction)\s*:?[\n]*", "", decoded, flags=re.IGNORECASE)
    decoded = decoded.split("\n###")[0].strip()

    if not decoded or decoded == data.prompt.strip():
        return {"response": "죄송합니다. 해당 질문에 대한 정보를 찾지 못했어요."}

    return {"response": decoded}