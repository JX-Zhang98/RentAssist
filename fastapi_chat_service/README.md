# FastAPI Chat Service

## 安装依赖

```bash
python3 -m pip install -r fastapi_chat_service/requirements.txt
```

## 启动服务

```bash
python3 -m uvicorn fastapi_chat_service.app:app --host 0.0.0.0 --port 8191
```

## 调用示例

```bash
curl -X POST "http://localhost:8191/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model_ip": "xxx.xxx.xx.x",
    "session_id": "abc123",
    "message": "查询海淀区的房源"
  }'
```

## 日志说明

- 日志目录：`log/`
- 每个 `session_id` 一个日志文件
- 文件名格式：`月日-时-分-sessionid.json`，例如 `0227-16-45-abc123.json`
- 每次请求会写入 `request` 事件，每次响应会写入 `response` 事件
