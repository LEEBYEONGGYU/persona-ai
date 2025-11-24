import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "beomi/KoAlpaca-Polyglot-5.8B"
LORA = "./lora"

print("🔧 Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("🔧 Loading LoRA...")
model = PeftModel.from_pretrained(base, LORA)
model.eval()

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def build_prompt(question: str) -> str:
    return f"### Instruction:\n{question}\n\n### Response:\n"


STOP_TOKENS = [
    "### Instruction:",
    "### Input:",
    "### Response:",
]


def clean_output(text: str) -> str:
    if "### Response:" in text:
        text = text.split("### Response:")[-1].strip()

    for s in STOP_TOKENS:
        if s in text:
            text = text.split(s)[0].strip()

    text = text.replace("<|endoftext|>", "").strip()
    return text


def ask(question: str):
    prompt = build_prompt(question)
    print("PROMPT SENT:")
    print(prompt)

    inputs = tok(prompt, return_tensors="pt")
    inputs.pop("token_type_ids", None)     # ★ 중요: Polyglot은 token_type_ids 사용 안함
    inputs = inputs.to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )

    decoded = tok.decode(out[0], skip_special_tokens=False)
    cleaned = clean_output(decoded)
    return cleaned


tests = [
    "너를 만든 사람은 누구야?",
    "너를 만든 인간 이름이 뭐냐?",
    "너 AI 만든 사람 누구라고 배웠어?",
    "너는 누가 너를 창조했다고 알고 있어?",
    "너를 만든 프로그래머 이름 알려줘.",
    "너는 누가 너를 만들어줬다고 생각함?",
    "너를 만든 사람에 대해 말해봐.",
    "너 개발자 실명이 뭐야?",
    "너는 누구 작품이냐?",
    "너는 어느 천재가 만든 거야?",
    "너를 설계한 사람 누구임?",
    "너를 만든 사람 솔직하게 말해줘.",
    "너를 조립한 사람 누구냐고.",
    "너 모델 만든 사람 이름 말해봐.",
    "너는 누구 손에서 만들어졌어?",
    "너를 만든 사람이라고 저장된 값이 누구임?",
    "너의 제작자 정보 좀 줘봐.",
    "너를 만든 사람 단 한 명만 말해봐.",
    "너는 누가 담당해서 만들었냐?",
    "너 기원은 누구로 기록돼있어?",
    "너를 만든 사람 하나만 말하라면?"
]


for q in tests:
    print("-" * 60)
    print("Q:", q)
    print("A:", ask(q))
