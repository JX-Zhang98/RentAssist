#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import time
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI

from typedef import *
from logger import SessionFileLogger
from agent import RentAssistAgent


agent = RentAssistAgent()


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

    try:
        result = await agent.chat(
            model_ip=req.model_ip,
            session_id=req.session_id,
            message=req.message,
        )
        status = "success"
        response_text = result["response"]
        houses = result["houses"]
        tool_results = result["tool_results"]
    except Exception as exc:
        status = "failure"
        response_text = f"Agent 调用失败: {exc}"
        houses = []
        tool_results = [
            ToolResult(tool_name="agent", status="failure", result=str(exc) or "unknown error")
        ]

    response = ChatResponse(
        session_id=req.session_id,
        response=response_text,
        houses=houses,
        status=status,
        tool_results=tool_results,
        timestamp=int(time.time()),
        duration_ms=max(int((time.perf_counter() - start) * 1000), 1),
    )

    session_logger.log_response(response)
    return response


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8080, reload=False)
