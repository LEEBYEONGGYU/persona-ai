import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import uvicorn
from fastapi.responses import JSONResponse

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "nlpai-lab/kullm-polyglot-5.8b-v2"   # ✅ 모델명
LORA_PATH = "./lora_falcon7b"     # ✅ LoRA 어댑터 경로

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ✅ 모델 로드 (오프로딩 포함)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder="./offload"     # 🔥 VRAM 부족 시 디스크 사용
)

# ✅ LoRA 어댑터 로드
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    offload_dir="./offload"         # 🔥 LoRA도 오프로딩 지원
)
model.eval()

@app.get("/")
async def root():
    return {"message": "서버 정상 동작 중"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("message", "")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # ✅ token_type_ids 제거
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return JSONResponse(content={"response": response}, headers={"Content-Type": "application/json; charset=utf-8"})



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
