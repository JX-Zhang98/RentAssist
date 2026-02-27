#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 

import argparse
import json
import sys
import time
from urllib import error, request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a test request to /api/v1/chat")
    parser.add_argument(
        "--url",
        default="http://localhost:8080/api/v1/chat",
        help="Target chat endpoint URL",
    )
    parser.add_argument("--model-ip", default="https://api.siliconflow.cn/v1", help="model_ip in request payload")
    parser.add_argument(
        "--session-id",
        default=f"cli-{int(time.time())}",
        help="session_id in request payload",
    )
    parser.add_argument(
        "--message",
        default="查询海淀区的房源",
        help="message in request payload",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="request timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "model_ip": args.model_ip,
        "session_id": args.session_id,
        "message": args.message,
    }

    req = request.Request(
        url=args.url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.time()
    try:
        with request.urlopen(req, timeout=args.timeout) as resp:
            body = resp.read().decode("utf-8")
            elapsed_ms = int((time.time() - started) * 1000)
            print(f"HTTP {resp.status} ({elapsed_ms}ms)")
            try:
                parsed = json.loads(body)
                print(json.dumps(parsed, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                print(body)
            return 0
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}")
        print(body)
        return 1
    except error.URLError as exc:
        print(f"Request failed: {exc.reason}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
