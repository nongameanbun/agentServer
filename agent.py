import datetime
from utils import get_current_hour_minute, _load_docs, _stringify_content, _history_to_messages
"""
WEEING Agent — LangGraph 기반 실행/계획 오케스트레이션

- Agent (실행): 자연어 → Tool 선택 → 실행 → 결과 반환
- Planner (계획만): 자연어 → 실행 계획(JSON) 반환 (Tool 실행 없음)
"""

import os
import dotenv
from types import SimpleNamespace
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from tools import ALL_TOOLS

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ─── 공통 키/Tool 규칙 ───

# ─── 동적 규칙 로드 ───

# BASE_TOOL_RULES was here

BASE_TOOL_RULES = """\
## 핵심 규칙

1. **입력 제어 시작/종료**: 
   - 키보드나 마우스 입력을 하려면 반드시 먼저 `input_on`을 호출해야 합니다. 작업이 끝나면 `input_off`로 입력 제어를 종료하세요.
   - **예외**: `start_weeing` 호출 시에는 내부적으로 `input_on/off`가 자동으로 처리되므로 별도로 호출할 필요가 없습니다.

2. **쇼트커트 및 요청 변환 우선순위**:
   - `### [필독/최우선] 문서: shortcuts.txt`에 정의된 요청 형식과 변환 규칙을 **가장 먼저** 확인하고 준수하세요.
   - 사용자의 입력이 `shortcuts.txt`의 예시와 유사할 경우, 문서에 명시된 템플릿에 따라 쿼리를 변환하여 처리하세요.

3. **키 입력 패턴**:
   - 단순 키 입력: `press_key_with_delay` 사용 (키를 눌렀다 떼는 것을 한번에 처리)
   - 키 조합: `press_two_keys` 사용 (예: Alt+키)
   - 키를 누른 채로 유지해야 할 때만: `press_key` → (다른 동작) → `release_key` 사용

3. **마우스 입력 패턴**:
   - 특정 좌표 클릭: `mouse_click`에 x, y를 함께 전달 (이동+클릭을 한번에)
   - 이동만: `mouse_move` 사용

4. **사용 가능한 키 이름**:
   - 알파벳: a ~ z, 숫자: num0 ~ num9, 펑션키: f1 ~ f12
   - 특수키: space, enter, esc, tab, backspace, capslock
   - 제어키: left_ctrl, left_shift, left_alt, right_ctrl, right_shift, right_alt
   - 화살표: up, down, left, right
   - 편집키: insert, delete, home, end, pageup, pagedown
   - 기호: backtick, left_bracket, right_bracket, backslash, semicolon, quote, comma, period, slash

5. **자동 사냥 제어**:
   - `wait_weeing_process`: 사냥 종료까지 대기할 때 사용하세요. (`get_main_process_pid` 반복 호출 금지)

6. **정책 준수 및 검증**: 
   - 아래 [정책 문서 분석 결과]에 명시된 상세 프로세스(로그인, 로그아웃, 이미지 검색 재시도, 마우스 클릭 범위 등)를 엄격히 준수하세요.
   - **중요**: 문서에서 언급된 모든 '확인', '검증', '감지' 단계는 반드시 `find_in_screen`, `find_object_on_screen` 등의 명시적인 툴 호출을 통해 수행되어야 합니다. 임의로 생략하지 마세요.
   - 불분명한 동작이 필요한 경우 `read_documentation` 툴을 사용해 최신 정책을 확인하세요.

## 정책 문서 분석 결과 (Docs/*.txt)

{dynamic_docs}

## 안전 및 기타
- 모든 시간 지연은 ms 단위(100ms = 0.1초)를 기본으로 하되, 문서에 명시된 초 단위 대기는 반드시 준수하세요.
- 위험한 작업은 계획에 항상 경고를 표기하세요.
"""

TOOL_RULES = BASE_TOOL_RULES.format(dynamic_docs=_load_docs())

# ─── Agent System Prompt (실행용) ───

SYSTEM_PROMPT = f"""\
당신은 WEEING 게임 자동화 시스템의 AI 어시스턴트입니다.
사용자의 자연어 명령을 이해하고, 적절한 Tool을 호출하여 게임을 제어합니다.

{TOOL_RULES}

## 추가 규칙 (실행 모드)
- 복합 작업: 여러 단계가 필요한 경우, 순서대로 Tool을 호출하세요. 각 단계의 결과를 확인한 후 다음 단계로 진행하세요.
- 상태 확인: 작업 전 필요한 경우 상태를 먼저 확인하세요 (예: 위잉 시작 전 빌드 목록 확인).
- Tool 호출 결과에 error가 포함되어 있으면 사용자에게 알려주세요.
- Timeout/네트워크 오류가 발생한 동일 Tool은 연속 재시도하지 마세요. 기본은 1회 시도 후 실패 원인을 보고하고, 사용자가 명시적으로 요청한 경우에만 재시도하세요.
- 응답 언어: 사용자의 언어에 맞춰 한국어 또는 영어로 응답하세요.
"""


def _build_tool_descriptions() -> str:
    """모든 Tool의 이름 + 설명 + 파라미터를 텍스트로 정리."""
    lines = []
    for t in ALL_TOOLS:
        params = ""
        if hasattr(t, "args_schema") and t.args_schema:
            schema_fields = t.args_schema.model_fields
            if schema_fields:
                params_list = []
                for fname, finfo in schema_fields.items():
                    annotation = finfo.annotation.__name__ if hasattr(finfo.annotation, '__name__') else str(finfo.annotation)
                    required = "필수" if finfo.is_required() else "선택"
                    params_list.append(f"{fname}: {annotation} ({required})")
                params = ", ".join(params_list)
        lines.append(f"- **{t.name}**({params}): {t.description or ''}")
    return "\n".join(lines)


# ─── Planning System Prompt (계획 전용) ───

PLANNING_PROMPT_TEMPLATE = """\
당신은 WEEING 게임 자동화 시스템의 **플래너**입니다.
사용자의 자연어 명령을 분석하고, 실행에 필요한 Tool 호출 계획을 JSON으로 출력합니다.
**Tool을 직접 실행하지 마세요.** 계획만 세우세요.

{tool_rules}

## 사용 가능한 Tool 목록

{tool_descriptions}

## 출력 형식

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.

```json
{{
  "summary": "계획에 대한 한줄 요약",
  "warning": "위험 요소가 있으면 경고 메시지, 없으면 null",
  "steps": [
    {{
      "order": 1,
      "tool": "tool_name",
      "args": {{{{ "param1": "value1" }}}},
      "description": "이 단계에서 수행하는 작업 설명",
      "depends_on": null
    }},
    {{
      "order": 2,
      "tool": "tool_name",
      "args": {{{{ "param1": "value1" }}}},
      "description": "이 단계에서 수행하는 작업 설명",
      "depends_on": 1
    }}
  ],
  "estimated_duration_ms": 1500
}}
```

## 규칙
- `order`는 실행 순서 (1부터 시작)
- `depends_on`은 이 단계가 의존하는 이전 단계 번호 (의존 없으면 null)
- `args`는 해당 Tool에 전달할 파라미터. 파라미터가 없으면 빈 객체 `{{{{}}}}`
- `estimated_duration_ms`는 전체 계획의 예상 소요 시간 (ms)
- 조건부 분기가 필요한 경우, `description`에 조건을 명시하세요
- 사냥 지시가 포함될 경우, `get_macro_accounts` -> `input_on` -> `game_login` -> `input_off` -> `wait_time` -> `start_weeing` -> `wait_weeing_process` -> `input_on` -> `game_logout` -> `input_off` 의 연속된 스텝 패턴을 엄격히 준수하세요.
"""


def _get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# Utility functions moved to utils.py


def _normalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return plan

    normalized_steps = []
    for step in steps:
        if not isinstance(step, dict):
            normalized_steps.append(step)
            continue
            
        # Magic string "CURRENT" replacement
        if step.get("tool") == "start_weeing":
            args = step.get("args")
            if isinstance(args, dict):
                hour, minute = get_current_hour_minute()
                if args.get("start_hour") in ("CURRENT", None) or "start_hour" not in args:
                    args["start_hour"] = hour
                if args.get("start_minute") in ("CURRENT", None) or "start_minute" not in args:
                    args["start_minute"] = minute
                    
        normalized_steps.append(step)

    plan["steps"] = normalized_steps
    return plan


class WeeingLangGraphExecutor:
    def __init__(self, model_name: str = "gpt-5.4", temperature: float = 0.0):
        llm = _get_llm(model_name, temperature)
        self.graph = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            prompt=SYSTEM_PROMPT,
        )
        self.max_iterations = 100

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        input_text = payload.get("input", "")
        chat_history = payload.get("chat_history", [])
        messages = [*_history_to_messages(chat_history), HumanMessage(content=input_text)]

        result = self.graph.invoke(
            {"messages": messages},
            config={"recursion_limit": self.max_iterations * 3},
        )

        all_messages = result.get("messages", [])
        return {
            "output": self._extract_output(all_messages),
            "intermediate_steps": self._extract_intermediate_steps(all_messages),
        }

    def stream_steps(self, payload: dict[str, Any]):
        """Agent의 실행 과정을 실시간으로 스트리밍합니다."""
        input_text = payload.get("input", "")
        chat_history = payload.get("chat_history", [])
        messages = [*_history_to_messages(chat_history), HumanMessage(content=input_text)]

        # graph.stream을 사용하여 노드 실행 결과를 실시간으로 확인
        for event in self.graph.stream({"messages": messages}, config={"recursion_limit": self.max_iterations * 3}):
            # event는 노드 이름(예: 'agent', 'tools')을 키로 가지는 딕셔너리
            for node_name, node_data in event.items():
                if node_name == "agent":
                    # AIMessage에서 tool_calls 추출
                    msgs = node_data.get("messages", [])
                    if msgs:
                        msg = msgs[-1]
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield {"type": "call", "tool": tc["name"], "input": tc["args"]}
                elif node_name == "tools":
                    # ToolMessage에서 결과 추출
                    msgs = node_data.get("messages", [])
                    if msgs:
                        msg = msgs[-1]
                        if isinstance(msg, ToolMessage):
                            yield {"type": "result", "tool": "unknown", "output": _stringify_content(msg.content)}
                
                # 최종 응답 확인 (AIMessage 중 tool_calls가 없는 마지막 메시지)
                if node_name == "agent":
                    msgs = node_data.get("messages", [])
                    if msgs:
                        msg = msgs[-1]
                        if isinstance(msg, AIMessage) and not msg.tool_calls:
                            yield {"type": "final", "output": _stringify_content(msg.content)}

    def _extract_output(self, messages: list[Any]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                if getattr(message, "tool_calls", None):
                    continue
                return _stringify_content(message.content)
        return ""

    def _extract_intermediate_steps(self, messages: list[Any]) -> list[tuple[Any, str]]:
        steps: list[tuple[Any, str]] = []
        pending_actions: list[tuple[str | None, Any]] = []

        for message in messages:
            if isinstance(message, AIMessage):
                tool_calls = getattr(message, "tool_calls", None) or []
                for tool_call in tool_calls:
                    action = SimpleNamespace(
                        tool=tool_call.get("name", "unknown"),
                        tool_input=tool_call.get("args", {}),
                    )
                    pending_actions.append((tool_call.get("id"), action))

            elif isinstance(message, ToolMessage):
                action = None
                call_id = getattr(message, "tool_call_id", None)
                if call_id:
                    for idx, (pending_id, pending_action) in enumerate(pending_actions):
                        if pending_id == call_id:
                            action = pending_action
                            pending_actions.pop(idx)
                            break

                if action is None and pending_actions:
                    _, action = pending_actions.pop(0)

                if action is None:
                    action = SimpleNamespace(tool="unknown", tool_input={})

                steps.append((action, _stringify_content(message.content)))

        return steps


class PlannerState(TypedDict, total=False):
    input: str
    chat_history: list[Any]
    plan: dict[str, Any]


class WeeingPlannerGraph:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = _get_llm(model_name, temperature)
        self.parser = JsonOutputParser()

        tool_descriptions = _build_tool_descriptions()
        self.planning_system_prompt = PLANNING_PROMPT_TEMPLATE.format(
            tool_rules=TOOL_RULES,
            tool_descriptions=tool_descriptions,
        )

        builder = StateGraph(PlannerState)
        builder.add_node("make_plan", self._make_plan)
        builder.add_edge(START, "make_plan")
        builder.add_edge("make_plan", END)
        self.graph = builder.compile()

    def _make_plan(self, state: PlannerState) -> PlannerState:
        input_text = state.get("input", "")
        history = state.get("chat_history", [])
        messages = [
            SystemMessage(content=self.planning_system_prompt),
            *_history_to_messages(history),
            HumanMessage(content=input_text),
        ]

        llm_output = self.llm.invoke(messages)
        plan = self.parser.parse(_stringify_content(llm_output.content))
        plan = _normalize_plan(plan)
        return {"plan": plan}

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        state: PlannerState = {
            "input": payload.get("input", ""),
            "chat_history": payload.get("chat_history", []),
        }
        result = self.graph.invoke(state)
        return result.get("plan", {})


# ─── 생성 함수 (main.py API 호환) ───

def create_weeing_agent(
    model_name: str = "gpt-5.4",
    temperature: float = 0.0,
):
    """LangGraph 기반 WEEING 실행 Agent를 생성합니다."""
    return WeeingLangGraphExecutor(model_name=model_name, temperature=temperature)


def create_weeing_planner(
    model_name: str = "gpt-5.4",
    temperature: float = 0.0,
):
    """LangGraph 기반 WEEING Planner를 생성합니다."""
    return WeeingPlannerGraph(model_name=model_name, temperature=temperature)
