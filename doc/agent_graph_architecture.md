# RentAssist Agent 图架构文档

## 1. 整体架构

使用 LangGraph 的 `StateGraph` 构建，包含 3 个节点和 1 个共享状态。

### 1.1 设计目标

**原始 React 模式（create_agent）：**
```
用户 → LLM → 工具 → LLM(读结果) → 工具 → LLM(总结) → ... → 回复
              多次 LLM 调用，工具结果反复喂回模型，浪费大量 token
```

**当前单次调用模式：**
```
用户 → LLM(1次) → 工具 → 本地格式化 → 回复
              LLM 只调用 1 次，工具结果本地处理，不再喂回模型
```

### 1.2 图结构

```
               ┌────────────┐
               │   agent     │  call_model()
               │  (LLM节点)  │  调用大模型，返回 AIMessage
               └──────┬─────┘
                      │
                after_agent() 路由函数
               ┌──────┴──────────┐
               │                 │
          有 tool_calls      无 tool_calls
               │                 │
               ▼                 ▼
         ┌───────────┐         END
         │   tools    │        (闲聊直接返回)
         │ (ToolNode) │
         └─────┬─────┘
               │
          after_tools() → "format"
               │
               ▼
         ┌────────────┐
         │   format    │  format_results()
         │ (本地格式化) │  把工具 JSON → 用户友好摘要
         └──────┬─────┘
                │
                ▼
               END
```

### 1.3 代码对应

```python
graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)          # LLM 决策节点
graph.add_node("tools", tool_node)           # 工具执行节点
graph.add_node("format", format_results)     # 本地结果格式化节点
graph.set_entry_point("agent")               # 入口
graph.add_conditional_edges("agent", after_agent)   # agent → tools 或 END
graph.add_conditional_edges("tools", after_tools)   # tools → format
graph.add_edge("format", END)                       # format → END
```

---

## 2. 共享状态 (State)

**核心概念：所有节点共享同一个 state，没有"给不同节点传不同输入"的概念。**

```python
# MessagesState 本质上就是：
state = {
    "messages": [msg1, msg2, msg3, ...]
}
```

### 2.1 状态更新机制

每个节点返回 `{"messages": [...]}` 时，LangGraph **自动追加**（append）到现有 messages 列表：

```python
# 节点返回
return {"messages": [new_ai_message]}

# LangGraph 自动执行
state["messages"].append(new_ai_message)
```

> 如果想**替换**而非追加，需要先插入 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 清空旧消息，
> 再放入新消息列表。这就是 `_apply_hooks` 中裁剪历史时的做法。

### 2.2 每个节点怎么从 state 中取自己需要的数据

| 节点 | 从 state 中取什么 | 怎么取的 |
|------|-------------------|----------|
| `agent` (call_model) | 全部 messages | `state["messages"]`（经裁剪后注入 SystemPrompt） |
| `tools` (ToolNode) | 最后一条 AIMessage 的 `tool_calls` | ToolNode 内部自动提取 `state["messages"][-1].tool_calls` |
| `format` (format_results) | 最后一条带 tool_calls 的 AIMessage 之后的所有 ToolMessage | 手动遍历 `state["messages"]` 定位 |

---

## 3. 各节点详解

### 3.1 agent 节点 — `call_model(state)`

**职责**：调用 LLM，理解用户意图，决定是闲聊回复还是调用工具。

**输入**：`state["messages"]`（包含历史对话 + 当前用户消息）

**处理逻辑**：
1. 调用 `middleware._apply_hooks(state)` 执行历史消息裁剪
2. 过滤掉裁剪产生的 `RemoveMessage`
3. 在头部注入 `SystemMessage(content=SYSTEM_PROMPT)`
4. 调用 `model_with_tools.invoke(prompt_messages)`

**输出**：`{"messages": [AIMessage(...)]}`
- 闲聊时：`AIMessage(content="你好！有什么可以帮你的？", tool_calls=[])`
- 需要工具时：`AIMessage(content="", tool_calls=[{name: "get_houses_by_platform", args: {...}}])`

```python
def call_model(state: MessagesState):
    patched = middleware._apply_hooks(state)
    if patched:
        messages = [m for m in patched["messages"] if not isinstance(m, RemoveMessage)]
    else:
        messages = state["messages"]
    prompt_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    return {"messages": [model_with_tools.invoke(prompt_messages)]}
```

### 3.2 after_agent 路由函数

**职责**：检查 LLM 返回的 AIMessage 是否包含 tool_calls，决定走向。

```python
def after_agent(state: MessagesState):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END
```

- 返回 `"tools"` → 走到 tools 节点
- 返回 `END` → 流程结束

### 3.3 tools 节点 — `ToolNode(tools)`

**职责**：根据 AIMessage 中的 tool_calls 执行对应的 MCP 工具。

**这是 LangGraph 预构建的节点**，内部逻辑（伪代码）：

```python
class ToolNode:
    def __call__(self, state):
        last_msg = state["messages"][-1]          # 取最后一条 AIMessage
        results = []
        for tc in last_msg.tool_calls:            # 遍历每个 tool_call
            tool = self.tool_map[tc["name"]]      # 按 name 找到对应工具
            result = tool.invoke(tc["args"])       # 用 AI 给的参数执行
            results.append(ToolMessage(
                content=result,
                tool_call_id=tc["id"],
                name=tc["name"],
            ))
        return {"messages": results}              # 全部追加到 state
```

### 3.4 format 节点 — `format_results(state)`

**职责**：从当前轮的 ToolMessage 中提取关键信息，格式化为用户友好的摘要文本，追加为一条 AIMessage。

**为什么只取当前轮的 ToolMessage**：state 中保存着历史所有消息，如果直接遍历所有 ToolMessage 会把前几轮的结果也格式化进来。

**定位策略**：找最后一条带 `tool_calls` 的 AIMessage 的位置，只取它之后的 ToolMessage。

```python
def format_results(state: MessagesState):
    messages = state["messages"]

    # 找最后一条带 tool_calls 的 AIMessage
    last_ai_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage) and messages[i].tool_calls:
            last_ai_idx = i
            break

    # 只取这条 AIMessage 之后的 ToolMessage
    tool_messages = [
        m for m in messages[last_ai_idx + 1:]
        if isinstance(m, ToolMessage)
    ]

    parts = []
    for tm in tool_messages:
        parts.append(_summarize_tool_result(tm.name, tm.content))

    summary_text = "\n\n".join(parts)
    return {"messages": [AIMessage(content=summary_text)]}
```

**输出的 AIMessage 会被 `_extract_response` 识别为最终回复**（因为它是最后一条无 tool_calls 的 AIMessage）。

### 3.5 _extract_response — 图外的打包方法

**不是图的节点**，而是 `chat()` 方法在图执行完毕后调用的静态方法。

**职责**：从最终 state 的 messages 中打包出 API 响应格式。

| 提取内容 | 来源 | 用途 |
|----------|------|------|
| `response_text` | 最后一条无 tool_calls 的 AIMessage（即 format 追加的那条） | 返回给用户的文本 |
| `tool_results` | 当前轮的 ToolMessage（同样用 last_ai_idx 定位） | `ChatResponse.tool_results` 元数据 |
| `houses` | 从 ToolMessage 原始 JSON + 摘要文本中提取 HF_xxxx | `ChatResponse.houses` |

```
format_results（图内）              _extract_response（图外）
─────────────────────              ──────────────────────────
  "内容加工"                          "打包发货"
  ToolMessage JSON → 摘要文本          遍历 messages 收集元数据
  追加为 AIMessage 到 state            组装成 {"response": json, "tool_results": [...]}
                                      返回给 main.py → ChatResponse
```

---

## 4. 完整示例

### 示例 1：闲聊（不调工具）

**用户输入**：`"你好"`

```
步骤 1: chat() 构建输入
  input = {"messages": [HumanMessage("你好")]}

步骤 2: 进入 agent 节点 (call_model)
  state["messages"] = [HumanMessage("你好")]
  ↓ 裁剪 → 注入 SystemPrompt → 调 LLM
  ↓ LLM 返回: AIMessage(content="你好！我是北京租房助手...", tool_calls=[])

  state["messages"] 变为:
    [0] HumanMessage("你好")
    [1] AIMessage(content="你好！我是北京租房助手...", tool_calls=[])

步骤 3: after_agent 路由
  last = messages[1]  →  tool_calls=[]  →  返回 END

步骤 4: 图结束，返回 state

步骤 5: _extract_response 打包
  从后往前找无 tool_calls 的 AIMessage → messages[1]
  response_text = "你好！我是北京租房助手..."
  tool_results = []（没有 ToolMessage）
  houses = []
  → {"response": "{\"message\":\"你好！...\",\"houses\":[]}", "tool_results": []}
```

### 示例 2：单轮工具调用

**用户输入**：`"帮我找海淀区5000以内的两居室"`

```
步骤 1: chat() 构建输入
  input = {"messages": [HumanMessage("帮我找海淀区5000以内的两居室")]}

步骤 2: 进入 agent 节点 (call_model)
  state["messages"] = [HumanMessage("帮我找海淀区5000以内的两居室")]
  ↓ 裁剪 → 注入 SystemPrompt → 调 LLM
  ↓ LLM 返回: AIMessage(
  ↓   content="",
  ↓   tool_calls=[{
  ↓     name: "get_houses_by_platform",
  ↓     args: {district: "海淀", max_price: 5000, bedrooms: "2"}
  ↓   }]
  ↓ )

  state["messages"]:
    [0] HumanMessage("帮我找海淀区5000以内的两居室")
    [1] AIMessage(tool_calls=[{get_houses_by_platform, ...}])

步骤 3: after_agent 路由
  last = messages[1]  →  tool_calls 非空  →  返回 "tools"

步骤 4: 进入 tools 节点 (ToolNode)
  读取 messages[-1].tool_calls → [{get_houses_by_platform, args}]
  执行工具 → 返回原始 JSON 结果

  state["messages"]:
    [0] HumanMessage("帮我找海淀区5000以内的两居室")
    [1] AIMessage(tool_calls=[...])
    [2] ToolMessage(name="get_houses_by_platform", content='{"code":200,"data":{"items":[{"house_id":"HF_1001","price":4500,"district":"海淀",...},...],"total":15}}')

步骤 5: after_tools 路由 → "format"

步骤 6: 进入 format 节点 (format_results)
  找 last_ai_idx = 1（messages[1] 是带 tool_calls 的 AIMessage）
  取 messages[2:] 中的 ToolMessage → [messages[2]]
  调用 _summarize_tool_result("get_houses_by_platform", raw_json)
  → 生成摘要:
    "共找到 15 套房源，展示前 5 套：
      1. 房源ID: HF_1001 | 月租: 4500元 | 位置: 海淀/西二旗小区 | 户型: 2室1厅
      2. 房源ID: HF_1002 | 月租: 4800元 | 位置: 海淀/上地小区 | 户型: 2室1厅
      ..."
  追加为 AIMessage(content=摘要)

  state["messages"]:
    [0] HumanMessage("帮我找海淀区5000以内的两居室")
    [1] AIMessage(tool_calls=[...])
    [2] ToolMessage(name="get_houses_by_platform", content=原始JSON)
    [3] AIMessage(content="共找到 15 套房源，展示前 5 套：...")   ← format 追加

步骤 7: format → END，图结束

步骤 8: _extract_response 打包
  找 last_ai_idx = 1
  当前轮 ToolMessage = [messages[2]]  →  提取 tool_results 元数据 + house_ids
  从后往前找无 tool_calls 的 AIMessage → messages[3]（format 追加的摘要）
  response_text = "共找到 15 套房源..."
  → {"response": "{\"message\":\"共找到...\",\"houses\":[\"HF_1001\",\"HF_1002\",...]}",
     "tool_results": [ToolResult(tool_name="get_houses_by_platform", status="success", ...)]}
```

### 示例 3：多轮对话（第2轮）

**前提**：示例 2 已执行完，checkpointer 保存了 session 历史。

**用户输入**：`"再看看朝阳的"`

```
步骤 1: chat() 构建输入
  input = {"messages": [HumanMessage("再看看朝阳的")]}

  因为 checkpointer，state 会恢复历史：
  state["messages"]:
    [0] HumanMessage("帮我找海淀区5000以内的两居室")     ← 第1轮
    [1] AIMessage(tool_calls=[...])                      ← 第1轮
    [2] ToolMessage(content=原始JSON)                    ← 第1轮
    [3] AIMessage(content="共找到 15 套房源...")           ← 第1轮 format
    [4] HumanMessage("再看看朝阳的")                     ← 当前输入

步骤 2: 进入 agent 节点 (call_model)
  ↓ _apply_hooks 裁剪历史
  ↓ compose_prompt_messages 看到最后一条是 HumanMessage:
  ↓   保留所有 HumanMessage + 非 tool_call 的 AIMessage
  ↓   删除 ToolMessage 和带 tool_calls 的 AIMessage
  ↓
  ↓ 裁剪后:
  ↓   [0] HumanMessage("帮我找海淀区5000以内的两居室")
  ↓   [1] AIMessage("合适的房源包括：['HF_1001',...]")   ← 摘要被压缩
  ↓   [2] HumanMessage("再看看朝阳的")
  ↓
  ↓ 注入 SystemPrompt → 调 LLM
  ↓ LLM 看到上下文，理解"朝阳的"是在上一轮基础上换区域
  ↓ 返回: AIMessage(tool_calls=[{get_houses_by_platform, {district: "朝阳", max_price: 5000, bedrooms: "2"}}])

步骤 3 ~ 步骤 8: 与示例 2 相同流程
  agent → tools → format → END → _extract_response

  最终 state["messages"]:
    [0] HumanMessage("帮我找海淀区5000以内的两居室")     ← 第1轮
    [1] AIMessage(tool_calls=[...])                      ← 第1轮
    [2] ToolMessage(...)                                 ← 第1轮
    [3] AIMessage("共找到 15 套房源...")                   ← 第1轮 format
    [4] HumanMessage("再看看朝阳的")                     ← 第2轮
    [5] AIMessage(tool_calls=[{朝阳...}])                ← 第2轮 LLM
    [6] ToolMessage(...)                                 ← 第2轮工具结果
    [7] AIMessage("共找到 8 套房源...")                    ← 第2轮 format

  _extract_response:
    last_ai_idx = 5（最后带 tool_calls 的）
    当前轮 ToolMessage = messages[6]    ← 只取第2轮的，不含第1轮的 messages[2]
    response_text = messages[7].content  ← format 追加的第2轮摘要
```

---

## 5. 扩展点

### 5.1 工具结果为空时自动重试

```python
def after_tools(state: MessagesState):
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        try:
            data = json.loads(last.content)
            items = data.get("data", {}).get("items", [])
            if len(items) == 0:
                return "agent"   # 回到 LLM 让它放宽条件
        except:
            pass
    return "format"
```

此时图变为：
```
agent → tools → (结果为空?) → agent (重试)
                (有结果)   → format → END
```

注意：回到 agent 时，`call_model` 中的 `_apply_hooks` 裁剪会再次执行，历史策略依然生效。

### 5.2 添加本地重试节点（不消耗 LLM）

```python
def retry_search(state: MessagesState):
    """本地放宽条件重新搜索，不经过 LLM"""
    # 从上一次 AIMessage.tool_calls 提取参数，放宽后直接调工具
    ...
    return {"messages": [ToolMessage(...)]}

graph.add_node("retry", retry_search)
# tools → (空结果?) → retry → format
# tools → (有结果?) → format → END
```

### 5.3 条件路由的工作原理

`add_conditional_edges(source_node, routing_function)` 的机制：
- 当 `source_node` 执行完毕后，调用 `routing_function(state)`
- routing_function 返回一个**节点名字符串**（或 `END`）
- 图引擎跳转到对应节点继续执行

```python
def after_agent(state):
    return "tools"    # → 跳到 tools 节点
    return END        # → 流程结束
    return "agent"    # → 跳回 agent（形成循环）
    return "format"   # → 跳到 format 节点
```

routing_function 只决定"**下一步去哪**"，不决定"**带什么数据去**"——因为所有节点共享同一个 state。
