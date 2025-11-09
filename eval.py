from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, json
from router_holiday import holiday_router_2025

# 모델 로딩
base = AutoModelForCausalLM.from_pretrained(
    "beomi/KoAlpaca-Polyglot-5.8B",
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base, "./lora")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
tokenizer.pad_token = tokenizer.eos_token

# 추론 함수
def generate(prompt):
    # 1) 먼저 룰로 답을 시도
    routed = holiday_router_2025(prompt if not user_input else f"{prompt} {user_input}")
    if routed is not None:
        return routed

    # 2) 안 되면 LLM
    full = f"### Instruction:\n{prompt}\n\n### Response:\n" if not user_input \
        else f"### Instruction:\n{prompt}\n\n### Input:\n{user_input}\n\n### Response:\n"
    inputs = tokenizer(full, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if k != "token_type_ids"}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    decoded = decoded.replace(full, "").strip().split("###")[0].strip()
    return decoded

# 평가 루프
total = 0
correct = 0


with open("eval_prompts.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        prompt = item["instruction"]
        user_input = item.get("input")
        expect = item.get("expected")
        expect_contains = item.get("expected_contains")

        # prompt format 맞게 generate 호출
        resp = generate(prompt, user_input) if user_input else generate(prompt)
        print(f"🧾 {prompt}\n➡️ {resp}")

        total += 1
        matched = False
        if expect and resp.strip() == expect.strip():
            correct += 1
            matched = True
        elif expect_contains and expect_contains in resp:
            correct += 1
            matched = True

        if not matched:
            with open("eval_failures.jsonl", "a", encoding="utf-8") as f_log:
                json.dump({
                    "prompt": prompt,
                    "input": user_input,
                    "expected": expect,
                    "expected_contains": expect_contains,
                    "response": resp
                }, f_log, ensure_ascii=False)
                f_log.write("\n")

print(f"\n✅ 정확도: {correct}/{total} = {correct/total:.2%}")
