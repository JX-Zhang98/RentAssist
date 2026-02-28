#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Union, List
from typing_extensions import Annotated


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
    message: str = Field(min_length=1)
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
