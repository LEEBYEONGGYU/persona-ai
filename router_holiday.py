
import re
from datetime import date, timedelta

# ✅ 2025년 고정 휴무일
HOLIDAY_RANGES = [
    ("2025-01-27", "2025-01-31"),
    ("2025-02-21", "2025-02-21"),
    ("2025-03-21", "2025-03-21"),
    ("2025-04-18", "2025-04-18"),
    ("2025-05-02", "2025-05-02"),
    ("2025-06-06", "2025-06-06"),
    ("2025-06-13", "2025-06-13"),
    ("2025-08-15", "2025-08-15"),
    ("2025-09-19", "2025-09-19"),
    ("2025-10-06", "2025-10-10"),
    ("2025-11-21", "2025-11-21"),
    ("2025-12-25", "2025-12-25"),
    ("2025-12-29", "2025-12-31"),
]

def _daterange(a: date, b: date):
    cur = a
    while cur <= b:
        yield cur
        cur += timedelta(days=1)

# 날짜 집합
HOLIDAYS_2025 = set()
for s, e in HOLIDAY_RANGES:
    sd = date.fromisoformat(s)
    ed = date.fromisoformat(e)
    for d in _daterange(sd, ed):
        HOLIDAYS_2025.add(d)

# 고정 응답 문장
CANONICAL_LIST_SENTENCE = (
    "2025년 기준 단체휴무일은 다음과 같습니다: "
    "1월 27~31일, 2월 21일, 3월 21일, 4월 18일, 5월 2일, "
    "6월 6일·13일, 8월 15일, 9월 19일, 10월 6~10일, 11월 21일, "
    "12월 25일·29~31일입니다."
)

# 날짜 패턴 (한글식 + 숫자식)
DATE_PAT_KR = re.compile(r"(?:\b(20\d{2})년\s*)?(\d{1,2})\s*월\s*(\d{1,2})\s*일(?:\s*에|\s*엔|\s*날)?")
DATE_PAT_NUM = re.compile(r"(?:\b(20\d{2})[-./])?(\d{1,2})[-./](\d{1,2})")

# 휴무 관련 키워드
휴무_키워드 = [
    "단체휴무", "쉬는 날", "전사", "출근", "휴무일", "휴무", "쉬나", "쉬냐",
    "쉬어", "쉬는지", "쉬나요", "쉬는가", "근무", "근무일", "출근해", "회사 쉬는 날"
]

def _kfmt(d: date) -> str:
    return f"{d.month}월 {d.day}일"

def holiday_router_2025(user_text: str) -> str | None:
    t = (user_text or "").strip()
    t_nospace = t.replace(" ", "")

    # 전체 목록 요청
    if any(k in t for k in ["전체", "목록", "리스트", "전부", "다 알려", "전체 알려", "전체 휴무", "단체휴무일 전체", "전부 쉬는 날"]):
        return CANONICAL_LIST_SENTENCE

    # 날짜 추출
    m = DATE_PAT_KR.search(t) or DATE_PAT_NUM.search(t)
    if not m:
        return None

    y, mth, day = m.groups()
    y = int(y) if y else 2025
    mth = int(mth)
    day = int(day)

    # 휴무 관련 의도 확인
    if not any(k in t_nospace for k in 휴무_키워드):
        return None

    try:
        q = date(y, mth, day)
    except ValueError:
        return "죄송합니다. 날짜 형식을 다시 확인해 주세요."

    if y != 2025:
        return None

    if q in HOLIDAYS_2025:
        return f"네 맞습니다. {_kfmt(q)}은 단체휴무일입니다."
    else:
        return f"아니요, {_kfmt(q)}은 단체휴무일이 아닙니다."
