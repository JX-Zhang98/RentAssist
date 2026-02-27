#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", re.IGNORECASE)
_HOUSE_ID_RE = re.compile(r"[A-Za-z]{1,6}_\d+")


@dataclass
class ModelChatResult:
    response: str
    houses: List[str]


class ChatModelService:
    def __init__(self, model_name: str | None = None, api_key: str | None = None) -> None:
        self._model_name = model_name or os.getenv("CHAT_MODEL_NAME", "Qwen/Qwen3-32B")
        self._api_key = api_key or os.getenv("siliconflow_API_KEY", "EMPTY")

    def chat(self, *, model_ip: str, message: str) -> ModelChatResult:
        client = self._build_client(model_ip=model_ip)
        completion = client.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是租房助手。请只返回 JSON："
                        '{"response":"给用户的话术","houses":["房源ID1","房源ID2"]}。'
                        "houses 为空时返回空数组。"
                    ),
                },
                {"role": "user", "content": message},
            ],
            temperature=0.2,
        )
        content = completion.choices[0].message.content or ""
        return self._parse_model_output(content)

    def _build_client(self, *, model_ip: str):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少 openai 依赖，请先安装 openai 包") from exc

        return OpenAI(api_key=self._api_key, base_url=self._build_base_url(model_ip))

    @staticmethod
    def _build_base_url(model_ip: str) -> str:
        addr = model_ip.strip()
        if not addr:
            raise ValueError("model_ip 不能为空")
        
        # 处理xx.xx.xx.xx仅IP样式输入
        if len(addr.split(".")) == 4:
            addr = f"http://{addr}:8888/v1"

        if addr.startswith("http://") or addr.startswith("https://"):
            base = addr.rstrip("/")
        else:
            base = f"http://{addr.strip('/')}"

        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    @staticmethod
    def _parse_model_output(raw: str) -> ModelChatResult:
        text = raw.strip()
        if not text:
            raise RuntimeError("模型返回内容为空")

        payload = ChatModelService._parse_json_payload(text)
        if payload is not None:
            response = str(payload.get("response", "")).strip()
            houses = payload.get("houses", [])
            if not response:
                raise RuntimeError("模型返回缺少 response 字段")
            if not isinstance(houses, list):
                raise RuntimeError("模型返回的 houses 不是数组")
            normalized = [str(item).strip() for item in houses if str(item).strip()]
            return ModelChatResult(response=response, houses=normalized)

        # 兼容模型未按 JSON 返回的场景：response 用原文，houses 尝试从文本中提取。
        houses = list(dict.fromkeys(_HOUSE_ID_RE.findall(text)))
        return ModelChatResult(response=text, houses=houses)

    @staticmethod
    def _parse_json_payload(text: str):
        candidates = [text]
        block = _JSON_BLOCK_RE.search(text)
        if block:
            candidates.insert(0, block.group(1))

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidates.append(text[first : last + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None
