import json
from collections import Counter
import re

# 파일 경로
FAIL_PATH = "eval_failures.jsonl"

# 정규 표현식: 날짜 추출용
date_pat = re.compile(r"(20\d{2})?년\s*(\d{1,2})월\s*(\d{1,2})일")

# 유형 분류 함수
def classify_failure(prompt: str, response: str) -> str:
    p = prompt.lower()
    r = response.lower()

    # 날짜 기반 질문
    if date_pat.search(prompt):
        if "아닙니다" in r or "단체휴무일이 아닙니다" in r:
            return "❌ 거짓 부정 오류 (틀린 부정)"
        elif "휴무일입니다" in r or "네" in r:
            return "❌ 거짓 긍정 오류 (틀린 긍정)"
        else:
            return "❓ 애매한 응답"

    # 전체 목록 질문
    if any(k in p for k in ["전체", "목록", "전부", "알려줘"]):
        if any(x in r for x in ["1월", "휴무일", "2025년", "30일"]):
            return "❌ 목록 불완전"
        elif "몰라" in r or "응답 없음" in r:
            return "❌ 목록 누락 또는 비응답"
        else:
            return "❓ 애매한 응답"

    # 기타
    return "❓ 기타"

# 분석 결과 저장
type_counter = Counter()
examples = {}

with open(FAIL_PATH, encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        prompt = item["prompt"]
        response = item["response"]
        fail_type = classify_failure(prompt, response)
        type_counter[fail_type] += 1
        examples.setdefault(fail_type, []).append((prompt, response))

# 결과 출력
print("📊 오답 유형 통계:")
for t, count in type_counter.most_common():
    print(f"{t}: {count}건")

print("\n🔎 유형별 예시:")
for t, ex_list in examples.items():
    print(f"\n▶ {t} ({len(ex_list)}건 중 일부):")
    for prompt, resp in ex_list[:3]:
        print(f"Q: {prompt}\nA: {resp}\n---")
