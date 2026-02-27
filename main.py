#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 

import time
import uvicorn  
from pathlib import Path
from fastapi import FastAPI


from typedef import *
from logger import SessionFileLogger
from model import ChatModelService



app = FastAPI(title="Chat Service", version="1.0.0")
session_logger = SessionFileLogger(Path("log"))
model_service = ChatModelService()


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    session_logger.log_request(req)

    try:
        model_result = model_service.chat(model_ip=req.model_ip, message=req.message)
        status = "success"
        response_text = model_result.response
        houses = model_result.houses
        tool_status = "success"
        tool_result = f"matched={len(houses)}"
    except Exception as exc:
        status = "failure"
        response_text = f"模型调用失败: {exc}"
        houses = []
        tool_status = "failure"
        tool_result = str(exc)

    response = ChatResponse(
        session_id=req.session_id,
        response=response_text,
        houses=houses,
        status=status,
        tool_results=[
            ToolResult(
                tool_name="house_search",
                status=tool_status,
                result=tool_result if tool_result else "unknown error",
            )
        ],
        timestamp=int(time.time()),
        duration_ms=max(int((time.perf_counter() - start) * 1000), 1),
    )

    session_logger.log_response(response)
    return response



if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8080, reload=False)
