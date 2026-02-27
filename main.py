#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 

import time
import uvicorn  
from pathlib import Path
from typing import  List
from fastapi import FastAPI


from typedef import *
from logger import SessionFileLogger



app = FastAPI(title="Chat Service", version="1.0.0")
session_logger = SessionFileLogger(Path("log"))

# 模拟查询房源，后续使用agent响应替代
def _pick_houses(message: str) -> List[str]:
    if "海淀" in message:
        return ["HF_4", "HF_6", "HF_277"]
    if "朝阳" in message:
        return ["CY_11", "CY_25"]
    return ["HF_4"]


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    session_logger.log_request(req)

    houses = _pick_houses(req.message)
    response = ChatResponse(
        session_id=req.session_id,
        response="已查询到匹配房源。",
        houses=houses,
        status="success",
        tool_results=[
            ToolResult(
                tool_name="house_search",
                status="success",
                result=f"matched={len(houses)}",
            )
        ],
        timestamp=int(time.time()),
        duration_ms=max(int((time.perf_counter() - start) * 1000), 1),
    )

    session_logger.log_response(response)
    return response



if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8080, reload=False)