from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


BASE = "beomi/KoAlpaca-Polyglot-5.8B"
LORA = "./lora"

print("🔧 Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("🔧 Loading LoRA...")
model = PeftModel.from_pretrained(base, LORA)
model.eval()

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def build_prompt(question: str) -> str:
    return f"### Instruction:\n{question}\n\n### Response:\n"


def clean_output(text: str) -> str:
    # Response 이후만 남기기
    if "### Response:" in text:
        text = text.split("### Response:")[-1]

    # 제거해야 할 패턴들
    stop_phrases = [
        "### Instruction:",
        "### Input:",
        "### Response:",
        "### 응답:",
        "응답:",
        "<|endoftext|>",
        "###",
    ]

    # stop phrase가 등장하면 거기서 잘라버림
    for p in stop_phrases:
        if p in text:
            text = text.split(p)[0]

    # 너무 자연어 섞인 경우 “.” 이후 cut
    # ex) "bglee입니다. 너는 누구야?" → 뒤 자르기
    if "." in text:
        first_sentence = text.split(".")[0] + "."
        if "bglee" in first_sentence:
            text = first_sentence

    # 유틸 정리
    return text.strip()



def infer(text: str) -> str:
    prompt = build_prompt(text)
    inputs = tok(prompt, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tok.eos_token_id
    )

    decoded = tok.decode(outputs[0], skip_special_tokens=False)
    return clean_output(decoded)


# FastAPI -----------------------------
app = FastAPI()

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    text: str


@app.post("/ask")
async def ask_api(q: Query):
    answer = infer(q.text)
    return {"answer": answer}


# Run ----------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
