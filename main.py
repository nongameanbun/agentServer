"""WEEING Agent Server — FastAPI 기반 AI Agent 서버

엔드포인트:
  POST /chat         — 자연어 명령 처리 (Tool 실행)
  POST /chat/plan    — 실행 계획만 생성 (Tool 실행 안 함)
  POST /chat/stream  — 자연어 명령 처리 (SSE 스트리밍)
  GET  /tools        — 사용 가능한 Tool 목록 조회
  GET  /health       — 서버 상태 확인
"""

import os
import json
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import create_weeing_agent, create_weeing_planner
from tools import ALL_TOOLS


# ─── 앱 생명주기 ───

agent_executor = None
planner_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor, planner_chain
    print("[AgentServer] Initializing WEEING Agent...")
    agent_executor = create_weeing_agent()
    print(f"[AgentServer] Agent initialized with {len(ALL_TOOLS)} tools.")
    print("[AgentServer] Initializing WEEING Planner...")
    planner_chain = create_weeing_planner()
    print("[AgentServer] Planner initialized.")
    yield
    print("[AgentServer] Shutting down...")


app = FastAPI(
    title="WEEING Agent Server",
    description="자연어 → Tool 호출 AI Agent 서버",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response  모델 ───

class ToolInfo(BaseModel):
    name: str
    description: str

class ChatResponse(BaseModel):
    response: str
    steps: list
    tool_calls: list

class PlanResponse(BaseModel):
    summary: str
    warning: Optional[str] = None
    steps: list
    estimated_duration_ms: Optional[int] = None


# ─── 엔드포인트 ───

@app.get("/health", summary="서버 상태 확인")
async def health():
    return {
        "status": "healthy",
        "agent_ready": agent_executor is not None,
        "tool_count": len(ALL_TOOLS),
    }


@app.get("/tools", summary="사용 가능한 Tool 목록 조회", response_model=list[ToolInfo])
async def list_tools():
    return [
        ToolInfo(name=t.name, description=t.description or "")
        for t in ALL_TOOLS
    ]


@app.post("/chat", summary="자연어 명령 처리", response_model=ChatResponse)
async def chat(message: str = Query(..., description="사용자 메시지")):
    if agent_executor is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = await asyncio.to_thread(
            agent_executor.invoke,
            {
                "input": message,
                "chat_history": [],
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # intermediate_steps에서 tool 호출 정보 추출
    tool_calls = []
    steps = []
    for step in result.get("intermediate_steps", []):
        action, observation = step
        tool_call = {
            "tool": action.tool,
            "input": action.tool_input if isinstance(action.tool_input, dict) else str(action.tool_input),
            "output": str(observation)[:500],  # 출력 길이 제한
        }
        tool_calls.append(tool_call)
        steps.append(f"[{action.tool}] → {str(observation)[:200]}")

    return ChatResponse(
        response=result.get("output", ""),
        steps=steps,
        tool_calls=tool_calls,
    )


# 백그라운드 작업 상태를 추적하기 위한 간단한 딕셔너리
background_jobs = {}

def _run_agent_background(job_id: str, message: str):
    """실제 에이전트를 스레드에서 돌리고 결과를 실시간으로 저장합니다."""
    try:
        if agent_executor is None:
            background_jobs[job_id] = {"status": "error", "message": "Agent not initialized"}
            return
            
        background_jobs[job_id] = {
            "status": "running", 
            "message": "Agent is working...",
            "steps": [],
            "current_step": None,
            "output": None,
            "stop_requested": False
        }
        
        # stream_steps를 사용하여 실시간 업데이트
        for event in agent_executor.stream_steps({"input": message, "chat_history": []}):
            # 중단 요청 확인
            if background_jobs[job_id].get("stop_requested"):
                background_jobs[job_id]["status"] = "stopped"
                background_jobs[job_id]["message"] = "Job was stopped by user request."
                print(f"[AgentServer] Background Job {job_id} Stopped by user.")
                return

            job = background_jobs[job_id]
            if event["type"] == "call":
                step_info = {"tool": event["tool"], "input": event["input"], "status": "running"}
                job["steps"].append(step_info)
                job["current_step"] = step_info
            elif event["type"] == "result":
                if job["current_step"]:
                    job["current_step"]["status"] = "completed"
                    job["current_step"]["output"] = event["output"][:500]
            elif event["type"] == "final":
                job["output"] = event["output"]

        background_jobs[job_id]["status"] = "completed"
        print(f"[AgentServer] Background Job {job_id} Completed.")
    except Exception as e:
        background_jobs[job_id] = {"status": "error", "message": str(e)}
        print(f"[AgentServer] Background Job {job_id} Error: {e}")


@app.post("/chat/background", summary="자연어 명령 (백그라운드 비동기 실행)")
async def chat_background(background_tasks: BackgroundTasks, message: str = Query(..., description="사용자 메시지")):
    """
    명령을 백그라운드 스레드에서 실행하도록 위임하고, 즉시 Job ID를 반환합니다.
    클라이언트는 무한 대기할 필요가 없습니다.
    """
    import uuid
    job_id = str(uuid.uuid4())
    
    # 작업 추가
    background_tasks.add_task(_run_agent_background, job_id, message)
    
    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Task has been added to the background queue."
    }


@app.get("/chat/background/jobs", summary="백그라운드 작업 목록 조회")
async def chat_background_jobs():
    """
    현재 관리되고 있는 모든 백그라운드 작업의 ID와 상태를 조회합니다.
    """
    return [
        {"job_id": job_id, "status": job["status"], "message": job.get("message")}
        for job_id, job in background_jobs.items()
    ]


@app.delete("/chat/background/stop/{job_id}", summary="백그라운드 작업 중단")
async def chat_background_stop(job_id: str):
    """
    진행 중인 백그라운드 작업을 중단시킵니다.
    """
    job = background_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] == "running":
        job["stop_requested"] = True
        return {"status": "success", "message": "Stop request sent to the agent."}
    else:
        return {"status": "error", "message": f"Job is already in {job['status']} state."}


@app.get("/chat/background/status/{job_id}", summary="백그라운드 작업 상태 조회")
async def chat_background_status(job_id: str):
    """
    chat_background 에서 받은 job_id 의 현재 상태를 확인합니다.
    """
    job = background_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job


@app.post("/chat/plan", summary="실행 계획만 생성 (Tool 실행 안 함)")
async def chat_plan(message: str = Query(..., description="사용자 메시지")):
    """자연어 명령을 분석하여 Tool 호출 계획만 생성합니다. 실제 Tool은 실행하지 않습니다."""
    if planner_chain is None:
        raise HTTPException(status_code=503, detail="Planner not initialized")

    try:
        plan = await asyncio.to_thread(
            planner_chain.invoke,
            {
                "input": message,
                "chat_history": [],
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Planner error: {str(e)}")

    return PlanResponse(
        summary=plan.get("summary", ""),
        warning=plan.get("warning"),
        steps=plan.get("steps", []),
        estimated_duration_ms=plan.get("estimated_duration_ms"),
    )


@app.post("/chat/stream", summary="자연어 명령 처리 (SSE 스트리밍)")
async def chat_stream(message: str = Query(..., description="사용자 메시지")):
    """Server-Sent Events로 Agent 실행 과정을 실시간 스트리밍합니다."""
    if agent_executor is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    async def event_generator():
        try:
            # Agent 실행을 별도 스레드에서 수행
            result = await asyncio.to_thread(
                agent_executor.invoke,
                {
                    "input": message,
                    "chat_history": [],
                }
            )

            # Tool 호출 단계 스트리밍
            for step in result.get("intermediate_steps", []):
                action, observation = step
                event_data = json.dumps({
                    "type": "tool_call",
                    "tool": action.tool,
                    "input": action.tool_input if isinstance(action.tool_input, dict) else str(action.tool_input),
                    "output": str(observation)[:500],
                }, ensure_ascii=False)
                yield f"data: {event_data}\n\n"

            # 최종 응답
            final_data = json.dumps({
                "type": "final",
                "response": result.get("output", ""),
            }, ensure_ascii=False)
            yield f"data: {final_data}\n\n"

        except Exception as e:
            error_data = json.dumps({
                "type": "error",
                "message": str(e),
            }, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ─── 서버 실행 ───

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100, log_level="warning")
