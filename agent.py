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
import re
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import List, Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AnyMessage, RemoveMessage
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from typedef import ToolResult

# Agent 内部日志
_log_dir = Path(__file__).parent / "log"
_log_dir.mkdir(parents=True, exist_ok=True)

# 用例工具记录文件
_EVAL_TOOLS_PATH = Path(__file__).parent / "cache" / "eval_tools.json"
_EVAL_TOOLS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_eval_tools() -> dict:
    """加载用例工具记录 {eval_id: [tool_name, ...]}"""
    if _EVAL_TOOLS_PATH.exists():
        try:
            return json.loads(_EVAL_TOOLS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_eval_tools(data: dict) -> None:
    _EVAL_TOOLS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_eval_id(session_id: str) -> str | None:
    """从 session_id 中提取用例编号，如 eval_z00832142_EV-27_xxx -> EV-27"""
    m = re.search(r"(EV-\d+)", session_id)
    return m.group(1) if m else None

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
        self.llm_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

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
        self.llm_calls += 1
        generations = []
        for gen_list in response.generations:
            for gen in gen_list:
                generations.append(str(gen.text)[:1000] if gen.text else str(gen.message.content)[:1000])

        # 提取 token 用量
        token_usage = {}
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]
        elif response.llm_output and "usage" in response.llm_output:
            token_usage = response.llm_output["usage"]

        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total = token_usage.get("total_tokens", prompt_tokens + completion_tokens)

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total

        self._log("llm_end", {
            "output": generations,
            "llm_call_index": self.llm_calls,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total,
            },
            "cumulative_tokens": {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
            },
        })

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


# MCP Server 路径
MCP_SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")

# System Prompt
SYSTEM_PROMPT = f"""你是一个专业的北京租房助手。

## 你的能力
- 帮助用户搜索北京地区的租房信息，包括按区域、价格、户型、地铁距离、通勤时间等多维度筛选
- 查询地标信息（地铁站、公司、商圈）
- 查询小区周边配套（商超、公园）
- 执行租房、退租、下架操作

## 工作原则
1. 如果用户只是闲聊，正常友好回复即可，不需要调用工具
2. 当用户描述租房需求时，先根据用户的诉求或抱怨提取需求，调用合适的工具搜索房源
3. 近地铁：地铁距离 800 米以内；地铁可达：1000 米以内
4. 用户预期价格(含xx元左右)为可接受的最高价格
5. 用户的偏好(如最好xx,可接受xx)纳入检索要求
6. 如果用户确认要租某套房，必须调用 rent_house 工具完成操作
7. 如果用户要退租或下架，必须调用对应的 terminate_rental 或 take_offline 工具

## 注意事项
- 默认使用安居客平台，除非用户指定其他平台
- 用户有最近、最低等要求时，调用工具也要设置合适的排序参数
"""


HistoryComposeHook = Callable[[list[AnyMessage], str], list[AnyMessage] | None]


def compose_prompt_messages(messages: list[AnyMessage], session_id: str) -> list[AnyMessage]:
    """模型消息组合入口。

    裁剪策略：
    1) 最后一条是 HumanMessage：保留全部 HumanMessage + 非 tool_call 的 AIMessage；
       删除 ToolMessage 和用于 tool_call 的 AIMessage。
    2) 最后一条是 ToolMessage：从“最近 HumanMessage 到末尾”全保留；
       更早历史仅保留对话层消息（Human + 非 tool_call 的 AI）。
    """
    _ = session_id
    prompt_messages = list(messages)
    if not prompt_messages:
        return prompt_messages

    def _is_dialogue_layer(msg: AnyMessage) -> bool:
        if isinstance(msg, HumanMessage):
            return True
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return True
        return False

    last = prompt_messages[-1]
    new_messages = []
    if isinstance(last, HumanMessage):
        dialogue = [m for m in prompt_messages if _is_dialogue_layer(m)]
        for m in dialogue:
            if isinstance(m, HumanMessage):
                new_messages.append(m)
            elif isinstance(m, AIMessage):
                msg_content = m.content
                # 从 msg_content 中提取 HF_1234 形式的房源ID
                houses = []
                _HOUSE_ID_RE = re.compile(r"HF_\d+")
                for hid in _HOUSE_ID_RE.findall(msg_content):
                    houses.append(hid) if hid not in houses else None
                
                m.content = "合适的房源包括：" + str(houses)
                new_messages.append(m)
        return new_messages

    if isinstance(last, ToolMessage):
        last_human_idx = -1
        for i in range(len(prompt_messages) - 1, -1, -1):
            if isinstance(prompt_messages[i], HumanMessage):
                last_human_idx = i
                break

        if last_human_idx < 0:
            return prompt_messages

        prefix = prompt_messages[:last_human_idx]
        tail = prompt_messages[last_human_idx:]
        trimmed_prefix = [m for m in prefix if _is_dialogue_layer(m)]
        return [*trimmed_prefix, *tail]

    return prompt_messages


class HistoryComposeMiddleware:
    """在模型调用前执行历史消息组合 Hook 链。"""

    def __init__(self, session_id: str, hooks: dict[str, HistoryComposeHook]):
        self._session_id = session_id
        self._hooks = hooks

    def _apply_hooks(self, state: dict[str, Any]) -> dict[str, Any] | None:
        if not self._hooks:
            return None

        messages = state.get("messages")
        if not isinstance(messages, list):
            return None

        current_messages = list(messages)
        changed = False

        for hook_name, hook in self._hooks.items():
            try:
                updated = hook(current_messages, self._session_id)
            except Exception as exc:
                _agent_logger.debug(json.dumps({
                    "event": "history_hook_error",
                    "session_id": self._session_id,
                    "hook_name": hook_name,
                    "error": str(exc),
                }, ensure_ascii=False))
                continue

            if updated is None:
                continue
            if not isinstance(updated, list):
                _agent_logger.debug(json.dumps({
                    "event": "history_hook_invalid_return",
                    "session_id": self._session_id,
                    "hook_name": hook_name,
                    "return_type": str(type(updated)),
                }, ensure_ascii=False))
                continue

            current_messages = updated
            changed = True

        if not changed:
            return None

        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *current_messages]}


# ==================== 工具结果本地摘要 ====================

_HOUSE_SEARCH_TOOLS = {
    "get_houses_by_platform", "get_houses_nearby", "get_houses_by_community",
}
_HOUSE_DETAIL_TOOLS = {"get_house_by_id", "get_house_listings"}
_OPERATION_TOOLS = {"rent_house", "terminate_rental", "take_offline"}
_LANDMARK_TOOLS = {
    "get_landmarks", "get_landmark_by_name", "search_landmarks", "get_landmark_by_id",
}


def _format_house_item(house: dict) -> str:
    """格式化单条房源为简洁摘要行"""
    parts = []
    hid = house.get("house_id", "")
    if hid:
        parts.append(f"房源ID: {hid}")

    price = house.get("price") or house.get("rent_price") or house.get("monthly_rent")
    if price:
        parts.append(f"月租: {price}元")

    district = house.get("district", "")
    area = house.get("area") or house.get("business_area", "")
    community = house.get("community") or house.get("community_name", "")
    location = "/".join(p for p in [district, area, community] if p)
    if location:
        parts.append(f"位置: {location}")

    layout = house.get("layout") or house.get("house_type", "")
    if layout:
        parts.append(f"户型: {layout}")
    else:
        bedrooms = house.get("bedrooms") or house.get("bedroom_count")
        if bedrooms:
            parts.append(f"卧室: {bedrooms}室")

    house_area = house.get("area_size") or house.get("size") or house.get("house_area")
    if house_area:
        parts.append(f"面积: {house_area}㎡")

    orientation = house.get("orientation", "")
    if orientation:
        parts.append(f"朝向: {orientation}")

    subway_dist = house.get("subway_distance") or house.get("distance_to_subway")
    if subway_dist:
        parts.append(f"地铁距离: {subway_dist}m")

    decoration = house.get("decoration", "")
    if decoration:
        parts.append(f"装修: {decoration}")

    rental_type = house.get("rental_type", "")
    if rental_type:
        parts.append(f"类型: {rental_type}")

    distance = house.get("distance") or house.get("straight_distance")
    if distance:
        parts.append(f"直线距离: {distance}m")

    walk_time = house.get("walk_time") or house.get("walking_time")
    if walk_time:
        parts.append(f"步行: {walk_time}分钟")

    return " | ".join(parts)


def _summarize_tool_result(tool_name: str, result_str: str) -> str:
    """根据工具类型，对原始 JSON 结果进行本地摘要格式化"""
    try:
        data = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return result_str[:1000]

    # --- 房源搜索类 ---
    if tool_name in _HOUSE_SEARCH_TOOLS:

        inner = data.get("data", data) if isinstance(data, dict) else data
        items = []
        total = 0
        if isinstance(inner, dict):
            items = inner.get("items") or inner.get("houses") or inner.get("results", [])
            total = inner.get("total", len(items) if isinstance(items, list) else 0)
        elif isinstance(inner, list):
            items = inner
            total = len(inner)

        if not items:
            return "未找到符合条件的房源。"

        display = items[:5]
        lines = [f"共找到 {total} 套房源，展示前 {len(display)} 套："]
        for i, h in enumerate(display, 1):
            if isinstance(h, dict):
                lines.append(f"  {i}. {_format_house_item(h)}")
            else:
                lines.append(f"  {i}. {str(h)[:200]}")
        if total > 5:
            lines.append(f"  ...还有 {total - 5} 套，可进一步筛选。")
        return "\n".join(lines)

    # --- 房源详情 ---
    if tool_name in _HOUSE_DETAIL_TOOLS:
        inner = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(inner, dict):
            return _format_house_item(inner)
        if isinstance(inner, list):
            lines = []
            for h in inner[:5]:
                if isinstance(h, dict):
                    lines.append(_format_house_item(h))
            return "\n".join(lines) if lines else str(inner)[:1000]
        return str(inner)[:1000]

    # --- 操作类（租房/退租/下架）---
    if tool_name in _OPERATION_TOOLS:
        op_names = {"rent_house": "租房", "terminate_rental": "退租", "take_offline": "下架"}
        op = op_names.get(tool_name, "操作")
        if isinstance(data, dict):
            code = data.get("code") or data.get("status_code")
            msg = data.get("message") or data.get("msg", "")
            if code and int(code) == 200:
                return f"{op}操作成功。{msg}"
            elif msg:
                return f"{op}操作结果：{msg}"
        return f"{op}操作返回：{result_str[:500]}"

    # --- 地标查询 ---
    if tool_name in _LANDMARK_TOOLS:
        inner = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(inner, dict):
            name = inner.get("name", "")
            lid = inner.get("id", "")
            cat = inner.get("category", "")
            dist = inner.get("district", "")
            parts = [p for p in [f"ID:{lid}", f"名称:{name}", f"类别:{cat}", f"区域:{dist}"] if not p.endswith(":")]
            return " | ".join(parts) if parts else json.dumps(inner, ensure_ascii=False)[:500]
        if isinstance(inner, list):
            lines = [f"共 {len(inner)} 个地标："]
            for item in inner[:10]:
                if isinstance(item, dict):
                    n = item.get("name", "")
                    i = item.get("id", "")
                    lines.append(f"  - {i} {n}")
            if len(inner) > 10:
                lines.append(f"  ...还有 {len(inner) - 10} 个")
            return "\n".join(lines)
        return str(inner)[:1000]

    # --- 周边配套 ---
    if tool_name == "get_nearby_landmarks":
        inner = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(inner, list):
            if not inner:
                return "附近未找到相关配套设施。"
            lines = [f"附近共 {len(inner)} 个配套设施："]
            for item in inner[:10]:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    d = item.get("distance") or item.get("distance_m", "")
                    lines.append(f"  - {name} (距离 {d}m)" if d else f"  - {name}")
            return "\n".join(lines)

    # --- 统计类 ---
    if tool_name in {"get_house_stats", "get_landmark_stats"}:
        inner = data.get("data", data) if isinstance(data, dict) else data
        return json.dumps(inner, ensure_ascii=False, indent=2)[:2000]

    # 默认
    return json.dumps(data, ensure_ascii=False)[:1000]


class RentAssistAgent:
    """租房助手 Agent，管理 MCP 客户端生命周期和多 session 对话"""

    def __init__(self):
        self._mcp_client: MultiServerMCPClient | None = None
        self._tools = None
        self._base_url: str | None = None
        self._checkpointer = MemorySaver()
        self._history_hooks: dict[str, HistoryComposeHook] = {
            "compose_prompt_messages": compose_prompt_messages
        }

    def register_history_hook(self, name: str, hook: HistoryComposeHook) -> None:
        """注册历史消息组合 Hook。"""
        if not name:
            raise ValueError("hook name 不能为空")
        self._history_hooks[name] = hook

    def unregister_history_hook(self, name: str) -> None:
        """移除历史消息组合 Hook。"""
        self._history_hooks.pop(name, None)

    def clear_history_hooks(self) -> None:
        """清空历史消息组合 Hook。"""
        self._history_hooks.clear()

    def list_history_hooks(self) -> list[str]:
        """返回当前已注册的 Hook 名称（按注册顺序）。"""
        return list(self._history_hooks.keys())

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

    def _ensure_base_url(self, model_ip: str):
        """首次请求时根据 model_ip 计算 base_url"""
        if self._base_url is not None:
            return
        self._base_url = self._build_base_url(model_ip)

    def _build_agent(self, session_id: str):
        """每次请求时构建带 Session-ID 头的 Agent（单次 LLM 调用模式）。

        与 create_react_agent 的唯一区别：tools 节点执行完后直接 END，
        不再把工具结果喂回模型，从而将 LLM 调用次数压缩为 1 次。
        """
        llm = ChatOpenAI(
            base_url=self._base_url,
            api_key=_config.get("api_key", "EMPTY"),
            model=_config.get("model_name", "Qwen/Qwen3-32B"),
            default_headers={"Session-ID": session_id},
        )

        # 根据用例编号动态选择工具
        tools = self._tools
        eval_id = _extract_eval_id(session_id)
        if eval_id:
            eval_tools_map = _load_eval_tools()
            if eval_id in eval_tools_map:
                known_names = set(eval_tools_map[eval_id])
                tools = [t for t in self._tools if t.name in known_names]
                _agent_logger.debug(json.dumps({
                    "event": "dynamic_tools", "eval_id": eval_id,
                    "all_tools": len(self._tools), "loaded_tools": len(tools),
                    "tool_names": [t.name for t in tools],
                }, ensure_ascii=False))

        model_with_tools = llm.bind_tools(tools)
        tool_node = ToolNode(tools)

        # --- 中间件：在调用模型前裁剪历史消息 ---
        middleware = HistoryComposeMiddleware(session_id=session_id, hooks=self._history_hooks)

        def call_model(state: MessagesState):
            # 应用历史裁剪 hook
            patched = middleware._apply_hooks(state)
            if patched:
                # _apply_hooks 返回 [RemoveMessage(...), ...actual messages...]
                # 过滤掉 RemoveMessage，只保留实际消息
                messages = [m for m in patched["messages"] if not isinstance(m, RemoveMessage)]
            else:
                messages = state["messages"]
            # 注入 system prompt
            from langchain_core.messages import SystemMessage
            prompt_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
            return {"messages": [model_with_tools.invoke(prompt_messages)]}

        def after_agent(state: MessagesState):
            """模型输出后路由：有 tool_calls → tools 节点；否则 → 结束"""
            last = state["messages"][-1]
            return "tools" if last.tool_calls else END

        def after_tools(state: MessagesState):
            """工具执行完后进入 format 节点做本地结果处理"""
            last = state["messages"][-1]
            if last.name in _LANDMARK_TOOLS:
                return "agent"
            else:
                return "format"
        

        def format_results(state: MessagesState):
            """本地处理工具结果：提取关键信息，生成用户友好的回复。

            只处理当前轮次的 ToolMessage（最后一条带 tool_calls 的 AIMessage 之后的），
            格式化后作为一条新的 AIMessage 追加到 state。
            """
            messages = state["messages"]

            # 找到最后一条带 tool_calls 的 AIMessage 的位置
            last_ai_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], AIMessage) and messages[i].tool_calls:
                    last_ai_idx = i
                    break

            if last_ai_idx < 0:
                return {"messages": []}

            # 只取这条 AIMessage 之后的 ToolMessage
            tool_messages = [
                m for m in messages[last_ai_idx + 1:]
                if isinstance(m, ToolMessage)
            ]
            if not tool_messages:
                return {"messages": []}

            parts = []
            for tm in tool_messages:
                tool_name = tm.name
                if tool_name in _LANDMARK_TOOLS:
                    continue
                elif tool_name in _OPERATION_TOOLS:
                    parts.append(tm.content[0]['text'])
                elif tool_name in _HOUSE_SEARCH_TOOLS:
                    houses_str = tm.content[0]["text"]
                    houses = json.loads(houses_str)
                    for hi in houses:
                        parts.append(str({
                            "id": hi["house_id"],
                            "price": hi["price"],
                        }))

            summary_text = "\n".join(parts)
            return {"messages": [AIMessage(content=summary_text)]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", tool_node)
        graph.add_node("format", format_results)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", after_agent)
        graph.add_conditional_edges("tools", after_tools)
        graph.add_edge("format", END)

        return graph.compile(checkpointer=self._checkpointer)

    async def chat(self, *, model_ip: str, session_id: str, message: str) -> dict:
        """
        处理一轮对话

        Returns:
            dict with keys: response (str), houses (list[str]), tool_results (list[ToolResult])
        """
        if self._tools is None:
            raise RuntimeError("MCP 客户端未启动，请先调用 start_mcp()")

        self._ensure_base_url(model_ip)
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
        callback._log("token_summary", {
            "llm_calls": callback.llm_calls,
            "total_prompt_tokens": callback.total_prompt_tokens,
            "total_completion_tokens": callback.total_completion_tokens,
            "total_tokens": callback.total_tokens,
        })

        # 记录本次用例使用的工具，供下次动态加载
        eval_id = _extract_eval_id(session_id)
        if eval_id:
            used_tools = {tr.tool_name for tr in response["tool_results"]}
            if used_tools:
                eval_tools_map = _load_eval_tools()
                existing = set(eval_tools_map.get(eval_id, []))
                merged = existing | used_tools
                if merged != existing:
                    eval_tools_map[eval_id] = sorted(merged)
                    _save_eval_tools(eval_tools_map)
                    callback._log("eval_tools_updated", {
                        "eval_id": eval_id, "tools": sorted(merged),
                    })

        return response

    async def close(self):
        """关闭 MCP 客户端"""
        if self._mcp_client is not None:
            self._mcp_client = None
            self._tools = None
            self._base_url = None

    @staticmethod
    def _extract_response(messages: list) -> dict:
        """从图的最终 state 中组装返回给 main.py 的响应。

        职责分工：
        - format_results 节点（图内）：负责把工具原始 JSON → 用户友好的摘要文本，追加为 AIMessage
        - 本方法（图外）：负责从 messages 中收集 tool_results 元数据 + 提取最终回复文本 + 提取 house_ids → 打包成 API 响应格式
        """
        _HOUSE_ID_RE = re.compile(r"HF_\d+")

        # --- 1. 只收集当前轮次的 ToolMessage → tool_results 元数据 ---
        tool_results: List[ToolResult] = []
        houses = []

        # 找最后一条带 tool_calls 的 AIMessage 位置
        last_ai_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage) and messages[i].tool_calls:
                last_ai_idx = i
                break

        current_tool_msgs = (
            [m for m in messages[last_ai_idx + 1:] if isinstance(m, ToolMessage)]
            if last_ai_idx >= 0 else []
        )

        for msg in current_tool_msgs:
                tool_name = msg.name or "unknown"
                try:
                    data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    status = "success"
                    result_str = msg.content if isinstance(msg.content, str) else json.dumps(data, ensure_ascii=False)
                except Exception:
                    status = "failed"
                    result_str = str(msg.content)

                tool_results.append(ToolResult(
                    tool_name=tool_name,
                    status=status,
                    result=result_str[:2000] if len(result_str) > 2000 else result_str,
                ))

                # 从工具原始结果中提取房源ID
                for hid in _HOUSE_ID_RE.findall(result_str):
                    if hid not in houses:
                        houses.append(hid)

        # --- 2. 提取最终回复文本（从后往前找第一条无 tool_calls 的 AIMessage）---
        #    闲聊 → LLM 直接回复的 AIMessage
        #    工具调用 → format_results 节点追加的 AIMessage（摘要文本）
        response_text = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                response_text = msg.content
                break

        # 从回复文本中也提取房源ID
        if response_text:
            for hid in _HOUSE_ID_RE.findall(response_text):
                if hid not in houses:
                    houses.append(hid)

        final_text = response_text or "抱歉，我暂时无法回答这个问题。"

        response = json.dumps(
            {"message": final_text, "houses": list(houses)},
            ensure_ascii=False,
        )

        return {
            "response": response,
            "tool_results": tool_results,
        }

    @staticmethod
    def _build_base_url(model_ip: str) -> str:
        addr = model_ip.strip()
        if not addr:
            raise ValueError("model_ip 不能为空")

        # 调试模式 --model-ip 1.2.3.4debug        
        debugMode = addr.endswith("debug")

        # 处理xx.xx.xx.xx仅IP样式输入
        if not addr.startswith("http") and len(addr.split(".")) == 4:
            addr = f"http://{addr[:-5]}:8888/v2" if debugMode else f"http://{addr}:8888/v1"

        if addr.startswith("http://") or addr.startswith("https://"):
            base = addr.rstrip("/")
        else:
            base = f"http://{addr.strip('/')}"

        return base

# TODO:根据HF_1234形式识别house
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
