🧠 개인 비서 AI 프로젝트
Polyglot 5.8B 모델을 기반으로 로컬 환경에서 직접 학습 및 추론이 가능한 개인 비서 AI를 만들어보는 프로젝트입니다.
외부 API(OpenAI 등)에 의존하지 않고, 내 GPU로 직접 LoRA 파인튜닝하여 맞춤형 챗봇을 구현할 수 있습니다.
(아직 지식과 경험이 많이 부족함)

🚀 주요 기능
🏗 Polyglot 5.8B 기반 LoRA 파인튜닝
⚡ 4bit 양자화로 VRAM 최적화 (RTX 3060 12GB 기준)
🤖 개인 데이터셋(질문·답변)으로 AI 학습
🌐 FastAPI 기반 API 서버 제공
🧩 모델 추론 테스트 스크립트 포함


🛠 환경 설정

1️⃣ WSL 설치
Windows 기능 켜기/끄기 →
✅ Hyper-V
✅ Linux용 Windows 하위 시스템(WLS2) 활성화 → 재부팅

명령 프롬프트(관리자 권한) 실행 후:
- wsl --install -d Ubuntu-22.04를 입력
- 사용자계정명 입력 후 암호입력
- 재실행시 시작메뉴에서 WSL을 입력하여 실행

2️⃣ Python 환경 구성
- sudo apt update && sudo apt install python3.10 python3.10-venv python3-pip -y
- python3.10 -m venv venv
- source venv/bin/activate
- cd /mnt/d/AI_python

3️⃣ 필수 패키지 설치
- pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
- pip install transformers peft datasets accelerate bitsandbytes
- pip install fastapi uvicorn

📂 프로젝트 구조
personal-ai-chatbot/
│── eval.json                # 평가용 데이터
│── sample_augmented.jsonl   # 학습용 데이터
│── train.py                 # LoRA 학습 코드
│── test.py                  # 학습된 모델 테스트
│── app.py                   # FastAPI 기반 웹 API 서버
│── requirements.txt
│── README.md

💻 실행 방법

학습
- python train.ty

테스트
- python test.py

API서버 실행(웹 채팅형태)
python app.py
(http://localhost:8000/chat로 접근가능)

⚠️ 주의사항
- RTX 3060 (VRAM 12GB) 기준으로 작성 → 낮은 사양 GPU에서는 OOM(Out of Memory) 가능성 있음
- 인텔 내장 GPU 노트북은 지원 불가
- 학습 데이터가 많을수록 학습 시간이 오래 걸립니다.
- 데스크톱 사용 시 에어컨 + 본체 뚜껑 열기, 노트북은 쿨러 필수

🔄 WSL 재실행 시
- source venv/bin/activate
