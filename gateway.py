from random import random
import time
from typing import Any, List
import dotenv
import os
import requests
import psutil

import os, dotenv

dotenv.load_dotenv()

inputHandler_API_URL    = f"http://127.0.0.1:{int(os.getenv('inputHandler_API_PORT'))}"
statusChecker_API_URL   = f"http://127.0.0.1:{int(os.getenv('statusChecker_API_PORT'))}"
alarmHandler_API_URL    = f"http://127.0.0.1:{int(os.getenv('alarmHandler_API_PORT'))}"
intrAction_API_URL      = f"http://127.0.0.1:{int(os.getenv('intrAction_API_PORT'))}"
mainAction_API_URL      = f"http://127.0.0.1:{int(os.getenv('mainAction_API_PORT'))}"
subaction_API_URL       = f"http://127.0.0.1:{int(os.getenv('subaction_API_PORT'))}"
streaning_API_URL       = f"http://127.0.0.1:{int(os.getenv('streaning_API_PORT'))}"
objectDetector_API_URL  = f"http://127.0.0.1:{int(os.getenv('objectDetector_API_PORT'))}"
runeSolver_API_URL      = f"http://127.0.0.1:{int(os.getenv('runeSolver_API_PORT'))}"
agentServer_API_URL     = f"http://127.0.0.1:{int(os.getenv('agentServer_API_PORT'))}"


# ─── helpers ───

def _safe_post(url: str, timeout: int = 10) -> Any | None:
    """POST 요청 후 JSON 반환. 실패 시 None."""
    try:
        r = requests.post(url, timeout=timeout).json()
        val = r.get('resp', None)
        if val is None :
            raise ValueError(f"POST {url} returned no 'resp' field")
        return val
    except Exception as e:
        # print(f"[gateway] POST {url} failed: {e}")
        return None

def _safe_get(url: str, timeout: int = 10) -> Any | None:
    """GET 요청 후 JSON 반환. 실패 시 None."""
    try:
        r = requests.get(url, timeout=timeout).json()
        val = r.get('resp', None)
        if val is None :
            raise ValueError(f"GET {url} returned no 'resp' field")
        return val
    except Exception as e:
        # print(f"[gateway] GET {url} failed: {e}")
        return None

def __precise_wait(duration: float, start_time: float) -> None:
    wait_time = duration - (time.perf_counter() - start_time)
    if wait_time <= 0:
        return
    if wait_time > 0.015:
        time.sleep(wait_time - 0.015)
    while (time.perf_counter() - start_time) < duration:
        pass

def _post_and_wait(url: str) -> None:
    """POST → resp(ms) 파싱 → precise_wait. resp 파싱 실패 시 대기 생략."""
    start_t = time.perf_counter()
    resp = _safe_post(url)

    try :
        if resp is None :
            raise ValueError(f"POST {_post_and_wait.__name__} got no 'resp' value")
        __precise_wait(int(resp) / 1000, start_t)
    except Exception as e:
        print(f"[gateway] _post_and_wait failed: {e}")

# ─── inputHandler ───

def on() -> None:
    _safe_post(f"{inputHandler_API_URL}/on")

def off() -> None:
    _safe_post(f"{inputHandler_API_URL}/off")

def press_key(key_name: str) -> None:
    _post_and_wait(f"{inputHandler_API_URL}/press_key?key_name={key_name}")

def release_key(key_name: str) -> None:
    _post_and_wait(f"{inputHandler_API_URL}/release_key?key_name={key_name}")

def releaseAll() -> None:
    _post_and_wait(f"{inputHandler_API_URL}/releaseAll")

def press_key_with_delay(key_name: str, delay: int) -> None:
    _post_and_wait(f"{inputHandler_API_URL}/press_key_with_delay?key_name={key_name}&delay={delay}")

def press_two_key(key1: str, key2: str) -> None:
    _post_and_wait(f"{inputHandler_API_URL}/press_two_key?key1={key1}&key2={key2}")

def mouse_move(x: int, y: int) -> None:
    _post_and_wait(f"{inputHandler_API_URL}/mouse/move?x={x}&y={y}")

def mouse_click(click_mode: str, delay: int, x: int | None = None, y: int | None = None) -> None:
    url = f"{inputHandler_API_URL}/mouse/click?click_mode={click_mode}&delay={delay}"
    if x is not None and y is not None:
        url += f"&x={x}&y={y}"
    _post_and_wait(url)

def Rdelay(delay: int) -> None:
    _post_and_wait(f"{inputHandler_API_URL}/delay?delay={delay}")

def Rdelay_2(delay: int) -> None:
    Rdelay(delay)

# ─── statusChecker ───

def get_status(mode : str | None = None) -> float | dict:
    resp = _safe_get(f"{statusChecker_API_URL}/status/get")

    if resp is None:
        return 0.0 if mode else {}
    if mode:
        return resp.get(mode, 0.0)
    return resp

def clear_status() -> None:
    _safe_post(f"{statusChecker_API_URL}/status/clear")

def check_rune() -> List[tuple[int, int]] | None:
    return _safe_get(f"{statusChecker_API_URL}/info/rune")

def clear_rune() -> None:
    _safe_post(f"{statusChecker_API_URL}/info/rune_clear")

def check_pos() -> List[tuple[int, int]] | None:
    return _safe_get(f"{statusChecker_API_URL}/info/mypos")

def get_exp_cycle() -> int:
    res = _safe_get(f"{statusChecker_API_URL}/cycle/get")
    return res if res != None else -1

def set_exp_cycle(cycle: int) -> None:
    _safe_post(f"{statusChecker_API_URL}/cycle/set?cycle={cycle}")

def capture_on() -> None:
    _safe_post(f"{statusChecker_API_URL}/capture/on")

def capture_off() -> None:
    _safe_post(f"{statusChecker_API_URL}/capture/off")

# ─── alarmHandler ───

def send_message(message: str, token: str | None = None):
    url = f"{alarmHandler_API_URL}/send_message?message={message}"
    if token is not None:
        url += f"&token={token}"
    try:
        response = requests.post(url, timeout=5)
        return response.status_code == 200
    except Exception as e:
        return False

def clear_alarm():
    pass  # alarmHandler에 reset_cooldown 엔드포인트 없음 — 필요 시 추가

# ─── intrAction ───

def continue_main() -> None:
    _safe_post(f"{intrAction_API_URL}/continue")

def clear_intr() -> None:
    _safe_post(f"{intrAction_API_URL}/reset")

def add_intr(intr: str) -> None:
    _safe_post(f"{intrAction_API_URL}/add_intr/{intr}")

def get_intr_status() -> dict:
    data = _safe_get(f"{intrAction_API_URL}/status")
    return data if data else {}

# ─── runeSolver ───

def awake_rune_solver():
    _safe_post(f"{runeSolver_API_URL}/awake_model")

def solve_rune():
    return _safe_get(f"{runeSolver_API_URL}/solve_rune")

# ─── objectDetector ───

def find_in_screen(target: str, xywh: str | None = None, conf: str | None = None) -> dict[str, list[int]] | None:
    ''' 
    returns {"center" : [x, y], "xywh" : [x, y, w, h]}
    '''
    res = find_in_screen_multiple(target, xywh, conf).get(target)
    return res[0] if res else None

def find_in_screen_yolo(model: str) -> List[dict[str, list[int]]]:
    res = _safe_get(f"{objectDetector_API_URL}/detect/yolo?req_model={model}")
    return res if res else []

def find_in_screen_multiple(targets: str, xywh: str | None = None, confs: str | None = None) -> dict:
    url = f"{objectDetector_API_URL}/detect/img_multiple?req_imgs={targets}"
    if xywh is not None:
        url += f"&xywh={xywh}"
    if confs is not None:
        url += f"&confs={confs}"
    return _safe_get(url)

# ─── mainAction / process ───

def get_running_build():
    resp = _safe_get(f"{mainAction_API_URL}/weeing/running_build")
    if resp :
        return resp
    return None

def get_main_pid():
    val = _safe_get(f"{mainAction_API_URL}/pid")
    return int(val) if val is not None else -1

def get_main_process():
    try:
        pid = get_main_pid()
        if pid <= 0:
            return None
        proc = psutil.Process(pid)
        if proc.is_running():
            return proc
    except psutil.NoSuchProcess:
        pass
    return None

def is_waiting_for_continue():
    proc = get_main_process()
    if not proc:
        return False
    
    if proc.status() == "stopped":
        return True
    return False

def suspend_main():
    proc = get_main_process()
    if proc:
        proc.suspend()
        print(f"[process] Suspended PID {proc.pid}")
        return True
    print("[process] No main process to suspend")
    return False

def resume_main():
    proc = get_main_process()
    if proc:
        proc.resume()
        print(f"[process] Resumed PID {proc.pid}")
        return True
    print("[process] No main process to resume")
    return False

def kill_main():
    proc = get_main_process()
    if proc:
        proc.kill()
        print(f"[process] Killed PID {proc.pid}")
        _safe_post(f"{mainAction_API_URL}/weeing/stop")
        return True
    print("[process] No main process to kill")
    return False

def _goto_point(x, y, tolerance = 1):
    resp = _safe_post(f"{mainAction_API_URL}/goto_point?x={x}&y={y}&tolerance={tolerance}")
    assert resp == -1, "Failed goto_point"

# ─── agentServer ───

def stop_agent_jobs():
    """agentServer에서 실행 중인 모든 백그라운드 작업을 조회하여 중단 요청을 보냅니다."""
    jobs = _safe_get(f"{agentServer_API_URL}/chat/background/jobs")
    if not jobs:
        return

    for job in jobs:
        if job.get("status") == "running":
            job_id = job.get("job_id")
            print(f"[gateway] Stopping agent job: {job_id}")
            # DELETE /chat/background/stop/{job_id}
            try:
                requests.delete(f"{agentServer_API_URL}/chat/background/stop/{job_id}", timeout=5)
            except Exception as e:
                print(f"[gateway] Failed to stop agent job {job_id}: {e}")


# ─── misc ───

def prob(percent):
    return random() < percent / 100

def reset_external_states():
    """외부 서비스 상태를 모두 초기화 (시작/종료 시 호출)"""
    Error_list = []
    pipeline = [
        capture_off, 
        releaseAll,
        clear_status,
        clear_rune,
        clear_alarm,
        clear_intr
    ]

    for func in pipeline:
        try:
            func()
        except Exception:
            Error_list.append(func.__name__)

    if Error_list:
        print(f"[gateway] WARNING: reset_external_states had errors in: {', '.join(Error_list)}")
