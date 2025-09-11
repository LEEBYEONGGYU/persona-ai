from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 🔧 모델 로딩
BASE = "beomi/KoAlpaca-Polyglot-5.8B"
ADAPT = "./lora_bizntc"

print("🔧 모델 로딩 중...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, ADAPT)
tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.pad_token = tokenizer.eos_token
print("✅ 모델 로딩 완료")

# FastAPI 앱 구성
app = FastAPI()

# ✅ CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 시 전체 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 입력 데이터 모델 정의
class PromptInput(BaseModel):
    prompt: str
    max_new_tokens: int = 160
    top_p: float = 0.9
    temperature: float = 0.7

# ✅ 텍스트 생성 엔드포인트
@app.post("/generate")
async def generate_text(data: PromptInput):
    formatted = f"""### Instruction:
{data.prompt}

### Response:
"""
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
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True).strip()

    # ✅ 후처리: 질문 복붙 제거 + "### Response:" 제거
    if data.prompt.strip() in decoded:
        decoded = decoded.replace(data.prompt.strip(), "").strip()

    decoded = re.sub(r"^#+\s*(응답|Response|Instruction)\s*:?[\n]*", "", decoded, flags=re.IGNORECASE)
    decoded = decoded.split("\n###")[0].strip()

    # ✅ fallback: 출력이 너무 이상하거나 질문과 동일한 경우
    if not decoded or decoded == data.prompt.strip():
        return {"response": "죄송합니다. 해당 질문에 대한 정보를 찾지 못했어요."}

    return {"response": decoded}
