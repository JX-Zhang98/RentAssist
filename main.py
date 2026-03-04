#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import time
import re
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI

from typedef import *
from logger import SessionFileLogger
from agent import RentAssistAgent, _extract_eval_id


agent = RentAssistAgent()
EVAL_TARGET_CASE_FILE = Path(__file__).parent / "cache" / "eval_target_case.txt"


def _normalize_eval_id(eval_id: str | None) -> str | None:
    """将 EV-1 或 EV-01 格式规范化为 EV-01"""
    if not eval_id:
        return None
    match = re.match(r"EV-(\d+)$", eval_id.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return f"EV-{int(match.group(1)):03d}"


def _read_target_eval_cases() -> set[str] | None:
    if not EVAL_TARGET_CASE_FILE.exists():
        return None
    try:
        raw = EVAL_TARGET_CASE_FILE.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[警告] 读取标记文件失败: {e}")
        return None

    cases: set[str] = set()
    for line in raw.splitlines():
        normalized = _normalize_eval_id(line.strip())
        if normalized:
            cases.add(normalized)

    return cases


@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent.start_mcp()
    yield
    await agent.close()


app = FastAPI(title="Chat Service", version="1.0.0", lifespan=lifespan)
session_logger = SessionFileLogger(Path("log"))


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    session_logger.log_request(req)

    target_eval_cases = _read_target_eval_cases()
    if target_eval_cases is not None:
        req_eval_case = _normalize_eval_id(_extract_eval_id(req.session_id))
        if req_eval_case not in target_eval_cases:
            print(
                f"[过滤] session_id={req.session_id}, 当前用例={req_eval_case}, "
                f"目标用例列表={sorted(target_eval_cases)}"
            )
            response = ChatResponse(
                session_id=req.session_id,
                response="{}",
                status="success",
                tool_results=[],
                timestamp=int(time.time()),
                duration_ms=max(int((time.perf_counter() - start) * 1000), 1),
            )
            session_logger.log_response(response)
            return response

    try:
        result = await agent.chat(
            model_ip=req.model_ip,
            session_id=req.session_id,
            message=req.message,
        )
        status = "success"
        response_text = result["response"]
        tool_results = result["tool_results"]
    except Exception as exc:
        status = "failure"
        response_text = f"Agent 调用失败: {exc}"
        tool_results = [
            ToolResult(tool_name="agent", status="failure", result=str(exc) or "unknown error")
        ]

    response = ChatResponse(
        session_id=req.session_id,
        response=response_text,
        status=status,
        tool_results=tool_results,
        timestamp=int(time.time()),
        duration_ms=max(int((time.perf_counter() - start) * 1000), 1),
    )

    session_logger.log_response(response)
    return response


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8080, reload=False)
