#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
import json
import time

from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import  List
from threading import Lock
from datetime import datetime
from copy import deepcopy
from typedef import *

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
            
        )
        copy_resp = deepcopy(resp)
        try:
            respjson = json.load(copy_resp.response)
            copy_resp.response = json.dumps(respjson, indent=2, ensure_ascii=False)

            for i in range(len(copy_resp.tool_results)):
                res = copy_resp.tool_results[i]
                resobj = json.loads(res.result)
                res.result = json.dumps(resobj, indent=2, ensure_ascii=False)
                copy_resp.tool_results[i] = res

        except:
            pass
        event.payload = copy_resp
        self._append_event(copy_resp.session_id, event)

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
        existing = sorted(self._log_dir.glob(f"*-{session_id}.json"))
        if existing:
            chosen = existing[-1]
        else:
            filename = f"{datetime.now().strftime('%m%d-%H-%M')}-{session_id}.json"
            chosen = self._log_dir / filename

        self._session_file_by_id[session_id] = chosen
        return chosen
