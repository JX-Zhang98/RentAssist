import json
import re
import time
import uvicorn  
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Literal, Union, List
from typing_extensions import Annotated
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model_ip: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tool_name: str = Field(min_length=1)
    status: str = Field(min_length=1)
    result: str = Field(min_length=1)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    session_id: str = Field(min_length=1)
    response: str = Field(min_length=1)
    houses: List[str]
    status: Literal["success", "failure"]
    tool_results: List[ToolResult]
    timestamp: int
    duration_ms: int


class RequestLogEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: Literal["request"]
    timestamp: int
    payload: ChatRequest


class ResponseLogEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: Literal["response"]
    timestamp: int
    payload: ChatResponse


LogEvent = Annotated[Union[RequestLogEvent , ResponseLogEvent], Field(discriminator="event_type")]


class SessionLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    created_at: int
    events: List[LogEvent] = Field(default_factory=list)


class SessionFileLogger:
    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._session_file_by_id: dict[str, Path] = {}
        self._lock = Lock()

    def log_request(self, req: ChatRequest) -> None:
        event = RequestLogEvent(
            event_type="request",
            timestamp=int(time.time()),
            payload=req,
        )
        self._append_event(req.session_id, event)

    def log_response(self, resp: ChatResponse) -> None:
        event = ResponseLogEvent(
            event_type="response",
            timestamp=int(time.time()),
            payload=resp,
        )
        self._append_event(resp.session_id, event)

    def _append_event(self, session_id: str, event: LogEvent) -> None:
        with self._lock:
            log_path = self._resolve_log_path(session_id)
            if log_path.exists():
                session_log = SessionLog.model_validate_json(log_path.read_text(encoding="utf-8"))
            else:
                session_log = SessionLog(session_id=session_id, created_at=event.timestamp)

            session_log.events.append(event)
            serialized = json.dumps(
                session_log.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            temp_path = log_path.with_suffix(".tmp")
            temp_path.write_text(serialized, encoding="utf-8")
            temp_path.replace(log_path)

    def _resolve_log_path(self, session_id: str) -> Path:
        path = self._session_file_by_id.get(session_id)
        if path is not None:
            return path

        safe_session_id = _safe_session_id(session_id)
        existing = sorted(self._log_dir.glob(f"*-{safe_session_id}.json"))
        if existing:
            chosen = existing[-1]
        else:
            filename = f"{datetime.now().strftime('%m%d-%H-%M')}-{safe_session_id}.json"
            chosen = self._log_dir / filename

        self._session_file_by_id[session_id] = chosen
        return chosen


def _safe_session_id(raw_session_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", raw_session_id)
    return safe or "session"


def _pick_houses(message: str) -> List[str]:
    if "海淀" in message:
        return ["HF_4", "HF_6", "HF_277"]
    if "朝阳" in message:
        return ["CY_11", "CY_25"]
    return ["HF_4"]


app = FastAPI(title="Chat Service", version="1.0.0")
session_logger = SessionFileLogger(Path("log"))


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