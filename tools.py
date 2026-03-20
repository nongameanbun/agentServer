# ─── 시간 정보 Tool ───
from agent import get_current_hour_minute as _get_current_hour_minute
from langchain_core.tools import tool

@tool
def get_current_hour_minute_tool() -> str:
    """현재 시스템 시각의 (시, 분)을 반환합니다. 예: '14, 23'"""
    hour, minute = _get_current_hour_minute()
    return f"{hour}, {minute}"
"""
WEEING Agent Server — LangChain 기반 자연어 → Tool 호출 Agent

gateway.py의 모든 함수를 LangChain Tool로 래핑하여,
자연어 명령을 받으면 적절한 Tool(또는 Tool 시퀀스)을 호출합니다.
"""

import os
import dotenv
import requests
from typing import Optional
from langchain_core.tools import tool

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ─── MSA 서비스 URL 생성 ───

inputHandler_URL   = f"http://127.0.0.1:{os.getenv('inputHandler_API_PORT')}"
statusChecker_URL  = f"http://127.0.0.1:{os.getenv('statusChecker_API_PORT')}"
alarmHandler_URL   = f"http://127.0.0.1:{os.getenv('alarmHandler_API_PORT')}"
intrAction_URL     = f"http://127.0.0.1:{os.getenv('intrAction_API_PORT')}"
mainAction_URL     = f"http://127.0.0.1:{os.getenv('mainAction_API_PORT')}"
subAction_URL      = f"http://127.0.0.1:{os.getenv('subaction_API_PORT')}"
streaming_URL      = f"http://127.0.0.1:{os.getenv('streaning_API_PORT')}"
objectDetector_URL = f"http://127.0.0.1:{os.getenv('objectDetector_API_PORT')}"
runeSolver_URL     = f"http://127.0.0.1:{os.getenv('runeSolver_API_PORT')}"


# ─── HTTP 헬퍼 ───

def _post(url: str, timeout: int = 10) -> dict:
    try:
        r = requests.post(url, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def _get(url: str, timeout: int = 10) -> dict:
    try:
        r = requests.get(url, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════
# 1. inputHandler 관련 Tools
# ══════════════════════════════════════════════

@tool
def input_on() -> str:
    """입력 제어를 시작합니다. 키보드/마우스 입력을 게임에 전달하기 위해 반드시 먼저 호출해야 합니다."""
    return str(_post(f"{inputHandler_URL}/on"))

@tool
def input_off() -> str:
    """입력 제어를 종료합니다. 키보드/마우스 입력을 게임에 더 이상 전달하지 않습니다."""
    return str(_post(f"{inputHandler_URL}/off"))

@tool
def press_key(key_name: str) -> str:
    """키보드의 특정 키를 누릅니다 (누른 상태 유지). 사용 가능한 키 이름: a~z, num0~num9, f1~f12, space, enter, esc, tab, left_ctrl, left_shift, left_alt, up, down, left, right 등."""
    return str(_post(f"{inputHandler_URL}/press_key?key_name={key_name}"))

@tool
def release_key(key_name: str) -> str:
    """눌려있는 특정 키를 해제합니다. key_name은 press_key에서 사용한 것과 동일해야 합니다."""
    return str(_post(f"{inputHandler_URL}/release_key?key_name={key_name}"))

@tool
def release_all_keys() -> str:
    """현재 눌려있는 모든 키를 해제합니다."""
    return str(_post(f"{inputHandler_URL}/releaseAll"))

@tool
def press_key_with_delay(key_name: str, delay_ms: int) -> str:
    """키를 누르고, 지정한 시간(ms)만큼 대기한 후, 키를 해제합니다. 일반적인 키 입력에 사용합니다. 예: press_key_with_delay('a', 100)"""
    return str(_post(f"{inputHandler_URL}/press_key_with_delay?key_name={key_name}&delay={delay_ms}"))

@tool
def press_two_keys(key1: str, key2: str) -> str:
    """두 키를 조합하여 입력합니다 (key1을 먼저 누르고, key2를 누른 후 순서대로 해제). 예: press_two_keys('left_alt', 'a')"""
    return str(_post(f"{inputHandler_URL}/press_two_key?key1={key1}&key2={key2}"))

@tool
def add_delay(delay_ms: int) -> str:
    """지정된 밀리초(ms) 만큼 대기합니다. inputHandler가 켜져있을 때(input_on 상태)만 사용 가능한 큐 기반 대기 방식입니다."""
    return str(_post(f"{inputHandler_URL}/delay?delay_ms={delay_ms}"))

@tool
def wait_time(delay_sec: int) -> str:
    """순수 Python time.sleep을 사용해 지정된 초(sec) 만큼 대기합니다. (로그아웃~로그인 사이 등 input_off 상태일 때도 사용 가능합니다.)"""
    import time
    time.sleep(delay_sec)
    return f"{delay_sec} seconds passed."


# ══════════════════════════════════════════════
# 2. 마우스 관련 Tools
# ══════════════════════════════════════════════

@tool
def mouse_move(x: int, y: int) -> str:
    """마우스를 절대 좌표 (x, y)로 이동합니다. Human-like Bezier 곡선으로 자연스럽게 이동합니다."""
    return str(_post(f"{inputHandler_URL}/mouse/move?x={x}&y={y}"))

@tool
def mouse_relative_move(dx: int, dy: int) -> str:
    """마우스를 현재 위치에서 상대적으로 (dx, dy)만큼 이동합니다."""
    return str(_post(f"{inputHandler_URL}/mouse/dmove?dx={dx}&dy={dy}"))

@tool
def mouse_click(click_mode: str, delay_ms: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
    """마우스 클릭을 수행합니다. click_mode: 'left', 'right', 'middle'. x, y를 지정하면 해당 좌표로 이동 후 클릭합니다. delay_ms는 클릭 후 대기 시간(ms)입니다."""
    url = f"{inputHandler_URL}/mouse/click?click_mode={click_mode}&delay={delay_ms}"
    if x is not None and y is not None:
        url += f"&x={x}&y={y}"
    return str(_post(url))


# ══════════════════════════════════════════════
# 3. statusChecker 관련 Tools
# ══════════════════════════════════════════════

@tool
def get_game_status() -> str:
    """현재 게임 상태 스냅샷을 조회합니다. HP, MP, EXP 등의 정보를 포함합니다."""
    return str(_get(f"{statusChecker_URL}/status/get"))

@tool
def clear_game_status() -> str:
    """게임 상태를 초기화합니다."""
    return str(_post(f"{statusChecker_URL}/status/clear"))

@tool
def check_rune_status() -> str:
    """현재 룬(rune) 상태를 조회합니다."""
    return str(_get(f"{statusChecker_URL}/info/rune"))

@tool
def clear_rune_status() -> str:
    """룬 상태를 초기화합니다."""
    return str(_post(f"{statusChecker_URL}/info/rune_clear"))

@tool
def check_my_position() -> str:
    """현재 캐릭터의 위치 좌표를 조회합니다."""
    return str(_get(f"{statusChecker_URL}/info/mypos"))

@tool
def get_exp_cycle() -> str:
    """현재 경험치 사이클 번호를 조회합니다."""
    return str(_get(f"{statusChecker_URL}/cycle/get"))

@tool
def set_exp_cycle(cycle: int) -> str:
    """경험치 사이클 번호를 설정합니다."""
    return str(_post(f"{statusChecker_URL}/cycle/set?cycle={cycle}"))

@tool
def capture_on() -> str:
    """화면 캡처를 시작합니다 (상태 감지 등에 필요)."""
    return str(_post(f"{statusChecker_URL}/capture/on"))

@tool
def capture_off() -> str:
    """화면 캡처를 중지합니다."""
    return str(_post(f"{statusChecker_URL}/capture/off"))


# ══════════════════════════════════════════════
# 4. alarmHandler 관련 Tools
# ══════════════════════════════════════════════

@tool
def send_alarm_message(message: str) -> str:
    """알림 메시지를 전송합니다. FCM 푸시 알림으로 전달됩니다."""
    return str(_post(f"{alarmHandler_URL}/send_message?message={message}"))


# ══════════════════════════════════════════════
# 5. intrAction (인터럽트) 관련 Tools
# ══════════════════════════════════════════════

@tool
def add_interrupt(interrupt_name: str) -> str:
    """인터럽트를 추가하고 실행합니다. 사용 가능한 인터럽트: 'user pause'(일시정지), 'viol'(위반 감지), 'liecheck'(거짓말 탐지기), 'shape'(모양 판별), 'exception'(예외처리), 'booster'(부스터), 'exit'(종료), 'gomyster'(미스터리)."""
    return str(_post(f"{intrAction_URL}/add_intr/{interrupt_name}"))

@tool
def continue_main_process() -> str:
    """일시정지(wait) 상태인 메인 프로세스를 재개합니다."""
    return str(_post(f"{intrAction_URL}/continue"))

@tool
def get_interrupt_status() -> str:
    """현재 인터럽트 상태를 조회합니다. 'running' 또는 'idle' 상태를 반환합니다."""
    return str(_get(f"{intrAction_URL}/status"))

@tool
def clear_interrupt() -> str:
    """인터럽트 상태를 초기화합니다."""
    return str(_post(f"{intrAction_URL}/reset"))


# ══════════════════════════════════════════════
# 6. mainAction (위잉 제어) 관련 Tools
# ══════════════════════════════════════════════

@tool
def get_build_list() -> str:
    """등록된 빌드 목록을 조회합니다. 위잉 시작 시 빌드 이름이 필요합니다."""
    return str(_get(f"{mainAction_URL}/build/list"))

@tool
def start_weeing(build_name: str, start_hour: int, start_minute: int) -> str:
    """위잉(자동 사냥)을 시작합니다. build_name은 빌드 목록에서 선택, start_hour/start_minute은 시작 시간입니다."""
    return str(_post(f"{mainAction_URL}/weeing/start/{build_name}/{start_hour}/{start_minute}"))

@tool
def pause_weeing() -> str:
    """위잉(자동 사냥)을 일시정지합니다."""
    return str(_post(f"{mainAction_URL}/weeing/pause"))

@tool
def get_main_process_pid() -> str:
    """실행 중인 위잉 프로세스의 PID를 조회합니다."""
    return str(_get(f"{mainAction_URL}/pid"))

@tool
def wait_weeing_process(check_interval_sec: int = 60) -> str:
    """위잉 프로세스가 종료될 때까지 지정된 간격(초)으로 파악하며 대기합니다. 사냥 진행 혹은 빌드 완료를 길게 대기해야 할 때만 사용하세요."""
    import time
    while True:
        resp = _get(f"{mainAction_URL}/pid")
        if isinstance(resp, dict):
            if resp.get("resp") == -1 or resp.get("pid") is None:
                return "Weeing process finished."
        time.sleep(check_interval_sec)


# ══════════════════════════════════════════════
# 7. subAction (보조 동작) 관련 Tools
# ══════════════════════════════════════════════

@tool
def game_logout() -> str:
    """게임을 로그아웃합니다. 메인 프로세스가 실행 중이 아닐 때만 가능합니다."""
    return str(_post(f"{subAction_URL}/weeing/logout", timeout=30))

@tool
def game_login(game_id: str, game_pw: str) -> str:
    """게임에 로그인합니다. 메인 프로세스가 실행 중이 아닐 때만 가능하며, 호출 전에 반드시 input_on을 먼저 실행해야 합니다."""
    return str(_post(f"{subAction_URL}/weeing/login?id={game_id}&pw={game_pw}", timeout=30))

@tool
def get_macro_accounts() -> str:
    """저장된 매크로 계정 목록을 조회합니다."""
    return str(_get(f"{subAction_URL}/weeing/macros"))

@tool
def type_text(text: str) -> str:
    """한글/영문 문자열을 게임에 타이핑합니다. 2벌식 자모 자동 분해를 지원합니다."""
    return str(_post(f"{subAction_URL}/input/sequence/{text}"))

@tool
def toggle_input_mode() -> str:
    """입력 모드를 전환합니다 (한/영 전환)."""
    return str(_post(f"{subAction_URL}/input/convert_mode"))

@tool
def register_fcm_token(token: str) -> str:
    """FCM 푸시 알림 토큰을 등록합니다."""
    return str(_post(f"{subAction_URL}/status/addFCM?token={token}"))


# ══════════════════════════════════════════════
# 8. objectDetector (화면 탐지) 관련 Tools
# ══════════════════════════════════════════════

@tool
def find_object_on_screen(target: str, xywh: Optional[str] = None, conf: Optional[str] = None) -> str:
    """화면에서 특정 오브젝트(이미지)를 탐지합니다. target은 탐지할 대상 이름, xywh는 '탐색 영역(x,y,w,h)', conf는 '신뢰도 임계값'입니다."""
    url = f"{objectDetector_URL}/detect/img_multiple?req_imgs={target}"
    if xywh:
        url += f"&xywh={xywh}"
    if conf:
        url += f"&confs={conf}"
    return str(_get(url))

@tool
def detect_with_yolo(model_name: str) -> str:
    """YOLO 모델을 사용하여 화면에서 오브젝트를 탐지합니다. model_name은 사용할 YOLO 모델 이름입니다."""
    return str(_get(f"{objectDetector_URL}/detect/yolo?req_model={model_name}"))


# ══════════════════════════════════════════════
# 9. runeSolver (룬 풀이) 관련 Tools
# ══════════════════════════════════════════════

@tool
def awake_rune_solver_model() -> str:
    """룬 풀이 AI 모델을 깨웁니다. 룬 풀이 전에 모델을 미리 로드하기 위해 사용합니다."""
    return str(_post(f"{runeSolver_URL}/awake_model"))

@tool
def find_in_screen(screen_name: str) -> str:
    """화면에서 특정 화면 상태(예: 'login_screen', 'main_lobby_screen', 'loading_screen')를 확인합니다."""
    # 실제로는 find_object_on_screen과 유사하게 동작하거나 특정 좌표의 픽셀/이미지를 매칭할 수 있습니다.
    # 여기서는 범용적인 탐지용으로 find_object_on_screen을 호출하는 브릿지 역할을 수행합니다.
    url = f"{objectDetector_URL}/detect/img_multiple?req_imgs={screen_name}"
    return str(_get(url))


@tool
def solve_rune() -> str:
    """현재 화면의 룬을 풀이합니다. 방향키 순서를 반환합니다."""
    return str(_get(f"{runeSolver_URL}/solve_rune"))


# ══════════════════════════════════════════════
# 10. streaming (스트리머) 관련 Tools
# ══════════════════════════════════════════════

@tool
def start_streamer(room_id: str = "default") -> str:
    """화면 스트리머를 시작합니다. WebRTC를 통해 원격으로 화면을 볼 수 있습니다."""
    return str(_post(f"{streaming_URL}/streamer/start?room_id={room_id}"))

@tool
def stop_streamer() -> str:
    """화면 스트리머를 중지합니다."""
    return str(_post(f"{streaming_URL}/streamer/stop"))

@tool
def get_streamer_status() -> str:
    """스트리머의 현재 상태를 조회합니다 (실행 여부, PID 등)."""
    return str(_get(f"{streaming_URL}/streamer/status"))


# ══════════════════════════════════════════════
# 11. 시스템 제어 Tools
# ══════════════════════════════════════════════

@tool
def reset_all_states() -> str:
    """모든 외부 서비스 상태를 초기화합니다. capture off, 키 해제, 입력 off, 상태 클리어 등을 일괄 수행합니다."""
    results = []
    for name, func in [
        ("capture_off", lambda: _post(f"{statusChecker_URL}/capture/off")),
        ("releaseAll",  lambda: _post(f"{inputHandler_URL}/releaseAll")),
        ("input_off",   lambda: _post(f"{inputHandler_URL}/off")),
        ("clear_status", lambda: _post(f"{statusChecker_URL}/status/clear")),
        ("clear_rune",  lambda: _post(f"{statusChecker_URL}/info/rune_clear")),
        ("clear_intr",  lambda: _post(f"{intrAction_URL}/reset")),
    ]:
        try:
            r = func()
            results.append(f"{name}: OK")
        except Exception as e:
            results.append(f"{name}: FAIL ({e})")
    return "\n".join(results)


@tool
def read_documentation(doc_name: str) -> str:
    """Docs 디렉토리의 특정 정책 문서(.txt)를 읽습니다. doc_name은 파일명(예: 'login', 'mouse')입니다."""
    try:
        docs_dir = os.path.join(os.path.dirname(__file__), "Docs")
        file_path = os.path.join(docs_dir, f"{doc_name.replace('.txt', '')}.txt")
        if not os.path.exists(file_path):
            return f"Error: {doc_name} 문서를 찾을 수 없습니다."
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading document: {e}"


# ─── 모든 Tool 리스트 (agent에서 import) ───

ALL_TOOLS = [
    get_current_hour_minute_tool,
    # Input
    input_on, input_off,
    press_key, release_key, release_all_keys,
    press_key_with_delay, press_two_keys, add_delay, wait_time,
    # Mouse
    mouse_move, mouse_relative_move, mouse_click,
    # Status
    get_game_status, clear_game_status,
    check_rune_status, clear_rune_status,
    check_my_position,
    get_exp_cycle, set_exp_cycle,
    capture_on, capture_off,
    # Alarm
    send_alarm_message,
    # Interrupt
    add_interrupt, continue_main_process,
    get_interrupt_status, clear_interrupt,
    # Weeing
    get_build_list, start_weeing, pause_weeing,
    get_main_process_pid, wait_weeing_process,
    # SubAction
    game_logout, game_login, get_macro_accounts,
    type_text, toggle_input_mode, register_fcm_token,
    # ObjectDetector
    find_object_on_screen, detect_with_yolo, find_in_screen,
    # RuneSolver
    awake_rune_solver_model, solve_rune,
    # Streaming
    start_streamer, stop_streamer, get_streamer_status,
    # Documentation
    read_documentation,
    # System
    reset_all_states,
]
