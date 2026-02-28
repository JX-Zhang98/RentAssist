#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 

"""
基于 LangGraph 的租房助手 Agent
- 通过 MCP (stdio) 连接租房仿真 API 工具
- 支持多轮对话（按 session_id 隔离）
- 支持闲聊 + 租房需求澄清 + 自主工具调用
"""

import json
import sys
import logging
import time
from pathlib import Path
from typing import List, Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from typedef import ToolResult

# Agent 内部日志
_log_dir = Path(__file__).parent / "log"
_log_dir.mkdir(parents=True, exist_ok=True)

_agent_logger = logging.getLogger("agent_trace")
_agent_logger.setLevel(logging.DEBUG)
_agent_logger.propagate = False
if not _agent_logger.handlers:
    _fh = logging.FileHandler(_log_dir / "agent_trace.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    _agent_logger.addHandler(_fh)


class AgentTraceCallback(BaseCallbackHandler):
    """记录 Agent 内部每次 LLM 调用和工具调用的详细日志"""

    def __init__(self, session_id: str):
        self.session_id = session_id

    def _log(self, event: str, data: dict):
        record = {"session_id": self.session_id, "event": event, "ts": time.time(), **data}
        _agent_logger.debug(json.dumps(record, ensure_ascii=False, default=str))

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
        self._log("llm_start", {"model": serialized.get("id", [""]), "prompts": prompts})

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs):
        msgs = []
        for batch in messages:
            for m in batch:
                msgs.append({"role": m.type, "content": str(m.content)[:500]})
        self._log("chat_model_start", {"model": serialized.get("id", [""]), "messages": msgs})

    def on_llm_end(self, response: LLMResult, **kwargs):
        generations = []
        for gen_list in response.generations:
            for gen in gen_list:
                generations.append(str(gen.text)[:1000] if gen.text else str(gen.message.content)[:1000])
        self._log("llm_end", {"output": generations})

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        self._log("tool_start", {"tool": serialized.get("name", ""), "input": input_str[:1000]})

    def on_tool_end(self, output: str, **kwargs):
        self._log("tool_end", {"output": str(output)[:2000]})

    def on_tool_error(self, error: BaseException, **kwargs):
        self._log("tool_error", {"error": str(error)[:1000]})

# 加载配置
_config_path = Path(__file__).parent / "config.json"
with open(_config_path, "r", encoding="utf-8") as f:
    _config = json.load(f)

USER_ID = _config.get("userid", "")

# MCP Server 路径
MCP_SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")

# System Prompt
SYSTEM_PROMPT = f"""你是一个专业的北京租房助手。你的用户工号是 {USER_ID}。

## 你的能力
- 帮助用户搜索北京地区的租房信息，包括按区域、价格、户型、地铁距离、通勤时间等多维度筛选
- 查询地标信息（地铁站、公司、商圈）
- 查询小区周边配套（商超、公园）
- 执行租房、退租、下架操作

## 工作原则
1. 当用户描述租房需求时，先理解需求，如果信息不足（如缺少区域、预算、户型偏好等关键信息），主动追问澄清
2. 需求明确后，调用合适的工具搜索房源，将结果以友好的方式呈现给用户
3. 调用房源相关工具时，user_id 参数始终传 "{USER_ID}"
4. 如果用户只是闲聊，正常友好回复即可，不需要调用工具
5. 推荐房源时，给出房源ID、价格、位置、户型等关键信息的简洁摘要
6. 近地铁：地铁距离 800 米以内；地铁可达：1000 米以内
7. 如果用户确认要租某套房，必须调用 rent_house 工具完成操作
8. 如果用户要退租或下架，必须调用对应的 terminate_rental 或 take_offline 工具

## 注意事项
- 默认使用安居客平台，除非用户指定其他平台
- 搜索结果较多时，优先展示最匹配用户需求的前5套
- 回复使用中文，简洁明了
"""


class RentAssistAgent:
    """租房助手 Agent，管理 MCP 客户端生命周期和多 session 对话"""

    def __init__(self):
        self._mcp_client: MultiServerMCPClient | None = None
        self._tools = None
        self._llm: ChatOpenAI | None = None
        self._agent = None
        self._checkpointer = MemorySaver()

    async def start_mcp(self):
        """启动 MCP 客户端，应在应用启动时调用"""
        self._mcp_client = MultiServerMCPClient(
            {
                "rent_api": {
                    "command": sys.executable,
                    "args": [MCP_SERVER_PATH],
                    "transport": "stdio",
                }
            }
        )
        self._tools = await self._mcp_client.get_tools()

    def _ensure_agent(self, model_ip: str):
        """首次请求时根据 model_ip 构建 Agent"""
        if self._llm is not None:
            return

        self._llm = ChatOpenAI(
            base_url=self._build_base_url(model_ip),
            api_key=_config.get("api_key", "EMPTY"),
            model=_config.get("model_name", "Qwen/Qwen3-32B"),
            temperature=0.2,
        )

    def _build_agent(self, session_id: str):
        """每次请求时用带 Session-ID 头的 LLM 构建 Agent"""
        llm_with_headers = self._llm.bind(extra_headers={"Session-ID": session_id})
        return create_agent(
            model=llm_with_headers,
            tools=self._tools,
            checkpointer=self._checkpointer,
            system_prompt=SYSTEM_PROMPT,
        )

    async def chat(self, *, model_ip: str, session_id: str, message: str) -> dict:
        """
        处理一轮对话

        Returns:
            dict with keys: response (str), houses (list[str]), tool_results (list[ToolResult])
        """
        if self._tools is None:
            raise RuntimeError("MCP 客户端未启动，请先调用 start_mcp()")

        self._ensure_agent(model_ip)
        agent = self._build_agent(session_id)

        callback = AgentTraceCallback(session_id)
        callback._log("user_input", {"message": message})

        config = {"configurable": {"thread_id": session_id}, "callbacks": [callback]}
        input_msg = {"messages": [HumanMessage(content=message)]}

        result = await agent.ainvoke(input_msg, config=config)

        messages = result["messages"]
        response = self._extract_response(messages)
        callback._log("agent_response", {
            "response": response["response"][:500],
            "tool_count": len(response["tool_results"]),
        })
        return response

    async def close(self):
        """关闭 MCP 客户端"""
        if self._mcp_client is not None:
            self._mcp_client = None
            self._tools = None
            self._llm = None

    @staticmethod
    def _extract_response(messages: list) -> dict:
        """从 Agent 输出的消息列表中提取最终回复、房源ID和工具调用结果

        - 无房源时 response 为纯文本
        - 有房源时 response 为 JSON 字符串: {"message":"...", "houses":["ID1","ID2"]}
        """
        response_text = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                response_text = msg.content
                break

        tool_results: List[ToolResult] = []
        houses = set()

        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_name = msg.name or "unknown"
                try:
                    data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    status = "success"
                    result_str = msg.content if isinstance(msg.content, str) else json.dumps(data, ensure_ascii=False)
                    _extract_house_ids(data, houses)
                except Exception:
                    status = "success"
                    result_str = str(msg.content)

                tool_results.append(ToolResult(
                    tool_name=tool_name,
                    status=status,
                    result=result_str[:2000] if len(result_str) > 2000 else result_str,
                ))

        final_text = response_text or "抱歉，我暂时无法回答这个问题。"

        # 有房源时，response 序列化为 JSON 字符串
        if houses:
            response = json.dumps(
                {"message": final_text, "houses": list(houses)},
                ensure_ascii=False,
            )
        else:
            response = final_text

        return {
            "response": response,
            "tool_results": tool_results,
        }

    @staticmethod
    def _build_base_url(model_ip: str) -> str:
        addr = model_ip.strip()
        if not addr:
            raise ValueError("model_ip 不能为空")
        
        # 处理xx.xx.xx.xx仅IP样式输入
        _config_path = Path(__file__).parent / "config.json"
        with open(_config_path, "r", encoding="utf-8") as f:
            _config = json.load(f)
        debugMode = _config.get("debug", False)
        if len(addr.split(".")) == 4:
            addr = f"http://{addr}:8888/v2" if debugMode else f"http://{addr}:8888/v1"

        if addr.startswith("http://") or addr.startswith("https://"):
            base = addr.rstrip("/")
        else:
            base = f"http://{addr.strip('/')}"

        return base


def _extract_house_ids(data, houses: set):
    """递归从 API 返回数据中提取房源 ID"""
    if isinstance(data, dict):
        if "house_id" in data:
            houses.add(str(data["house_id"]))
        if "data" in data:
            _extract_house_ids(data["data"], houses)
        if "items" in data:
            _extract_house_ids(data["items"], houses)
    elif isinstance(data, list):
        for item in data:
            _extract_house_ids(item, houses)
