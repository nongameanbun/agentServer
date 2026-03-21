import os
import dotenv
from typing import Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from tools import ALL_TOOLS
from utils import _load_docs, _stringify_content, _history_to_messages

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ─── 동적 규칙 로드 (agent.py 재사용) ───
dynamic_docs = _load_docs()

PLANNING_FORMAT_RULES = """\
당신은 WEEING 게임 자동화 시스템의 **플래너**입니다.
사용자의 자연어 명령을 분석하고, 아래 명시된 **엄격한 텍스트 형식**으로 실행 계획을 수립하세요.

## 핵심 규칙
1. **출력 형식**: 반드시 아래와 같은 번호가 매겨진 행 단위 형식을 사용하고, **각 단계에서 사용할 Tool(도구)의 이름을 대괄호 `[Tool: 이름]` 형태로 명시**하세요.
   - `1. [Tool: 도구이름] {{동작 설명 및 사용할 인자}}`
   - `2. [Tool: 도구이름] {{동작 설명}}`
   - `3. [Tool: 도구이름] {{동작 A}} if ({{조건}}) else {{동작 B}}`
   - ...
   - `N. endofplan` (계획의 마지막 줄은 반드시 `endofplan`이어야 합니다.)

2. **도구 및 정책 적용 기준 (중요)**:
   - 사용자의 요청이 `Docs` 문서에 있는 복잡한 절차(예: 로그인, 빌드 실행 등)라면 해당 문서를 엄격히 따르세요.
   - **문서에 없는 단순 요청이라도, 아래 '사용 가능한 도구'의 기능으로 직접 해결할 수 있다면 해당 도구를 사용하여 즉시 계획을 세우세요.** (예: "포탈/me 위치 찾아줘" → `find_in_screen` 사용, "클릭해줘" → `mouse_click` 사용)

3. **필수 정보 누락 시 처리**:
   - 대상 계정 등 도구 실행에 필수적인 파라미터가 누락된 경우, 계획을 세우지 말고 사용자에게 질문 메시지를 평문으로 반환하세요.
   - 단, "portal 찾아줘", "me 위치 찾아줘" 처럼 대상 이미지 이름이 명확히 주어진 경우는 정상적으로 계획을 세웁니다.

4. **지원하지 않는 요청 처리**:
   - 정책 문서에도 없고, 어떤 도구로도 해결할 수 없는 완전한 예외 상황에만 오류 메시지를 반환하세요.

## 정책 문서 분석 결과 (Docs/*.txt)
{dynamic_docs}

## 사용 가능한 도구 (참고용)
{tool_descriptions}
"""

def _build_tool_descriptions() -> str:
    lines = []
    for t in ALL_TOOLS:
        lines.append(f"- {t.name}: {t.description or ''}")
    return "\n".join(lines)

class PlannerState(TypedDict):
    input: str
    chat_history: list[Any]
    plan: str

class WeeingPlannerSequentialGraph:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        tool_descriptions = _build_tool_descriptions()
        self.system_prompt = PLANNING_FORMAT_RULES.format(
            dynamic_docs=dynamic_docs,
            tool_descriptions=tool_descriptions
        )

        builder = StateGraph(PlannerState)
        builder.add_node("planner", self._planner_node)
        builder.add_edge(START, "planner")
        builder.add_edge("planner", END)
        self.graph = builder.compile()

    def _planner_node(self, state: PlannerState) -> dict:
        messages = [
            SystemMessage(content=self.system_prompt),
            *_history_to_messages(state.get("chat_history", [])),
            HumanMessage(content=state["input"])
        ]
        response = self.llm.invoke(messages)
        return {"plan": _stringify_content(response.content)}

    def invoke(self, payload: dict[str, Any]) -> str:
        state = {
            "input": payload.get("input", ""),
            "chat_history": payload.get("chat_history", []),
            "plan": ""
        }
        result = self.graph.invoke(state)
        return result.get("plan", "Error: Failed to generate plan.")

    def save_mermaid_png(self, filename: str = "planner_graph.png") -> None:
        """
        LangGraph 구조를 PNG 이미지 파일로 저장합니다.
        (주의: 내부적으로 Mermaid 렌더링 API를 호출하므로 타임아웃이 발생할 수 있습니다.)
        """
        try:
            print("PNG 이미지를 생성하는 중입니다. (API 호출 대기...)")
            # draw_mermaid_png()는 렌더링된 PNG의 바이트(bytes) 데이터를 반환합니다.
            png_bytes = self.graph.get_graph().draw_mermaid_png()
            
            # 바이너리 쓰기 모드("wb")로 파일 저장
            with open(filename, "wb") as f:
                f.write(png_bytes)
                
            print(f"그래프가 '{filename}'에 성공적으로 저장되었습니다.")
            
        except Exception as e:
            print(f"PNG 저장 중 오류가 발생했습니다: {e}")
            print("💡 Tip: 외부 API 타임아웃이 원인일 수 있습니다. 기존 마크다운 방식을 사용하거나 로컬 렌더링(Graphviz)이 필요할 수 있습니다.")

def create_sequential_planner(model_name: str = "gpt-5.4", temperature: float = 0.0):
    return WeeingPlannerSequentialGraph(model_name=model_name, temperature=temperature)