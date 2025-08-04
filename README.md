개인 비서AI 만들어보는 프로젝트

1. 실행방법
- 제어판->프로그램 및 기능->Windows 기능 켜기/끄기->Hyper-V와 Linux용 Windows 하위시스템을 체크하고 확인 -> 재부팅
- 명령프롬프트 관리자권한으로 실행하고 아래 명령어 입력
  wsl --install -d Ubuntu-22.04
  WSL이 설치되면 우분투가 설치되며 계정이름과 암호를 입력함

2. 우분투에서 아래 명령어를 순서대로 입력함
   
- sudo apt update && sudo apt install python3.10 python3.10-venv python3-pip -y
- python3.10 -m venv venv
- source venv/bin/activate
- cd /mnt/d/AI_python
- pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
- pip install transformers peft datasets accelerate bitsandbytes
- pip install fastapi uvicorn

4. 파일 구조
- eval.json 평가용 데이터
- sample_augumented.jsonl 학습용 데이터
- test.py 학습후 지정된 결과테스트
- app.ty 웹용 API서버 
- train.ty 학습

5. 기타 
- 현재 코드기준은 RTX3060(VRAM 12GB)기준으로 되어있으므로 하위시스템에서는 OOM(메모리 오버)가 발생 할 수 있음
- 인텔노트북 사용불가
- 학습용 데이터가 많을 수록 시간이 오래걸림(에어컨+본체 뚜껑 열어두시길)
- 노트북으로 사용시 쿨러 필수
