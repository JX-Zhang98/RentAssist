"""
基于 LangGraph 的租房助手 Agent
- 通过 MCP (stdio) 连接租房仿真 API 工具
- 支持多轮对话（按 session_id 隔离）
- 支持闲聊 + 租房需求澄清 + 自主工具调用
"""

import json
import sys
from pathlib import Path
from typing import List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from typedef import ToolResult

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
        self._agent = None
        self._checkpointer = MemorySaver()

    async def _ensure_initialized(self, model_ip: str):
        """懒初始化：首次调用时启动 MCP 客户端并构建 Agent"""
        if self._agent is not None:
            return

        # 启动 MCP 客户端（stdio 模式连接 mcp_server.py）
        self._mcp_client = MultiServerMCPClient(
            {
                "rent_api": {
                    "command": sys.executable,
                    "args": [MCP_SERVER_PATH],
                    "transport": "stdio",
                }
            }
        )
        await self._mcp_client.__aenter__()
        tools = self._mcp_client.get_tools()

        # 构建 LLM
        llm = ChatOpenAI(
            base_url=self._build_base_url(model_ip),
            api_key=_config.get("api_key", "EMPTY"),
            model=_config.get("model_name", "Qwen/Qwen3-32B"),
            temperature=0.2,
        )

        # 构建 ReAct Agent
        self._agent = create_react_agent(
            model=llm,
            tools=tools,
            checkpointer=self._checkpointer,
            prompt=SYSTEM_PROMPT,
        )

    async def chat(self, *, model_ip: str, session_id: str, message: str) -> dict:
        """
        处理一轮对话

        Returns:
            dict with keys: response (str), houses (list[str]), tool_results (list[ToolResult])
        """
        await self._ensure_initialized(model_ip)

        config = {"configurable": {"thread_id": session_id}}
        input_msg = {"messages": [HumanMessage(content=message)]}

        # 运行 Agent
        result = await self._agent.ainvoke(input_msg, config=config)

        # 从结果中提取信息
        messages = result["messages"]
        return self._extract_response(messages)

    async def close(self):
        """关闭 MCP 客户端"""
        if self._mcp_client is not None:
            await self._mcp_client.__aexit__(None, None, None)
            self._mcp_client = None
            self._agent = None

    @staticmethod
    def _extract_response(messages: list) -> dict:
        """从 Agent 输出的消息列表中提取最终回复、房源ID和工具调用结果"""
        # 最后一条 AI 消息作为回复
        response_text = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                response_text = msg.content
                break

        # 收集所有工具调用结果
        tool_results: List[ToolResult] = []
        houses = set()

        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_name = msg.name or "unknown"
                try:
                    data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    status = "success"
                    result_str = msg.content if isinstance(msg.content, str) else json.dumps(data, ensure_ascii=False)
                    # 从工具结果中提取房源 ID
                    _extract_house_ids(data, houses)
                except Exception:
                    status = "success"
                    result_str = str(msg.content)

                tool_results.append(ToolResult(
                    tool_name=tool_name,
                    status=status,
                    result=result_str[:2000] if len(result_str) > 2000 else result_str,
                ))

        return {
            "response": response_text or "抱歉，我暂时无法回答这个问题。",
            "houses": list(houses),
            "tool_results": tool_results,
        }

    @staticmethod
    def _build_base_url(model_ip: str) -> str:
        addr = model_ip.strip()
        if not addr:
            raise ValueError("model_ip 不能为空")
        if addr.startswith("http://") or addr.startswith("https://"):
            base = addr.rstrip("/")
        else:
            base = f"http://{addr.strip('/')}"
        if not base.endswith("/v1"):
            base = f"{base}/v1"
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
