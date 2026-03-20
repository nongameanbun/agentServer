# nongameanbun - agentServer

## 프로젝트 구조
```text
.
├── main.py # 에이전트 서버 실행 엔트리포인트 및 API 라우터
├── agent.py # 에이전트 관련 핵심 로직 및 행동 결정
├── tools.py # 서버에서 사용하는 유틸리티 및 도구 함수
├── requirements.txt # 의존성 패키지 목록
├── Docs # API/기능별 세부 설명서
│   ├── buildRun.txt # buildRun 동작 관련 문서
│   ├── image.txt # 이미지 처리 관련 문서
│   ├── login.txt # 로그인 동작 관련 문서
│   ├── logout.txt # 로그아웃 동작 관련 문서
│   ├── mouse.txt # 마우스 제어 관련 문서
│   └── shortcuts.txt # 단축키 처리 관련 문서
├── response.txt # 서버 응답 예시 및 로그 저장
└── env.example # 환경 변수 예시 파일
```

## 사전 요구 사항

### 환경 변수 세팅 (`.env`)
환경에 맞게 각 포트 번호를 지정하여 프로젝트 루트에 `.env` 파일을 생성합니다.

```powershell
Copy-Item env.example .env
```

`env.example` 포맷 예시:
```ini
RUNE_SOLVER_PORT=8020
inputHandler_API_PORT=8001
statusChecker_API_PORT=8002
alarmHandler_API_PORT=8003
intrAction_API_PORT=8004
mainAction_API_PORT=8005
subaction_API_PORT=8006
streaning_API_PORT=8007
objectDetector_API_PORT=8008
agentServer_API_PORT=8009
```

## 실행 방법

```bash
pip install -r requirements.txt
python main.py
```

`localhost:8009/docs` 로 swagger 명세를 확인 가능
