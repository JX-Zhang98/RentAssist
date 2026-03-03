import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import requests
import urllib3

START_URL = "https://aienable.coreai.rnd.huawei.com/aimatch/expertStreamApi/api/judge/start"
RANK_URL = "https://aienable.coreai.rnd.huawei.com/aimatch/expertStreamApi/api/rank"

START_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

RANK_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "User-Agent": "Mozilla/5.0",
}


@dataclass
class RankSnapshot:
    user_id: str
    rank: int | None
    score: float | None
    is_evaluating: bool


@dataclass
class EvalRunResult:
    user_id: str
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    rank: int
    score: float
    start_response: dict[str, Any]


class EvalApiClient:
    def __init__(self, *, verify_ssl: bool, request_timeout: float) -> None:
        self.verify_ssl = verify_ssl
        self.request_timeout = request_timeout
        self.session = requests.Session()
        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def start_task(
        self,
        *,
        user_id: str,
        username: str,
        agent_ip: str,
        agent_port: int,
    ) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "username": username,
            "agent_ip": agent_ip,
            "agent_port": agent_port,
        }
        response = self.session.post(
            START_URL,
            headers=START_HEADERS,
            json=payload,
            verify=self.verify_ssl,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return parse_json_response(response)

    def query_rank_rows(self, *, limit: int) -> list[dict[str, Any]]:
        response = self.session.get(
            RANK_URL,
            headers=RANK_HEADERS,
            params={"limit": limit},
            verify=self.verify_ssl,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        payload = parse_json_response(response)
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise ValueError("rank接口返回结构异常：缺少 data 列表")
        return rows


def parse_json_response(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {"raw_text": response.text}

    if not isinstance(data, dict):
        return {"raw_data": data}
    return data


def normalize_is_evaluating(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "y", "t", "running", "evaluating"}
    return bool(value)


def parse_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_user_snapshot(rank_rows: list[dict[str, Any]], user_id: str) -> RankSnapshot | None:
    for item in rank_rows:
        if str(item.get("user_id")) != user_id:
            continue
        return RankSnapshot(
            user_id=user_id,
            rank=parse_int(item.get("rank")),
            score=parse_float(item.get("score")),
            is_evaluating=normalize_is_evaluating(item.get("is_evaluating")),
        )
    return None


def poll_until_finished(
    *,
    fetch_snapshot: Callable[[], RankSnapshot | None],
    interval_seconds: float,
    timeout_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.time,
    on_poll: Callable[[int, float, RankSnapshot | None], None] | None = None,
) -> RankSnapshot:
    started_at = now_fn()
    attempts = 0

    while True:
        attempts += 1
        elapsed = now_fn() - started_at
        snapshot = fetch_snapshot()
        if on_poll is not None:
            on_poll(attempts, elapsed, snapshot)

        if snapshot and not snapshot.is_evaluating:
            if snapshot.rank is not None and snapshot.score is not None:
                return snapshot

        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"等待测评结束超时，已等待 {elapsed:.1f}s，timeout={timeout_seconds:.1f}s"
            )
        sleep_fn(interval_seconds)


def run(
    *,
    client: EvalApiClient,
    user_id: str,
    username: str,
    agent_ip: str,
    agent_port: int,
    limit: int,
    interval_seconds: float,
    timeout_seconds: float,
    on_started: Callable[[dict[str, Any]], None] | None = None,
    on_poll: Callable[[int, float, RankSnapshot | None], None] | None = None,
    on_query_error: Callable[[Exception], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.time,
) -> EvalRunResult:
    started_at = datetime.now()
    started_perf = time.perf_counter()

    start_payload = client.start_task(
        user_id=user_id,
        username=username,
        agent_ip=agent_ip,
        agent_port=agent_port,
    )
    if on_started is not None:
        on_started(start_payload)

    def fetch_snapshot_once() -> RankSnapshot | None:
        try:
            rows = client.query_rank_rows(limit=limit)
        except (requests.RequestException, ValueError) as exc:
            if on_query_error is not None:
                on_query_error(exc)
            return None
        return extract_user_snapshot(rows, user_id)

    final_snapshot = poll_until_finished(
        fetch_snapshot=fetch_snapshot_once,
        interval_seconds=interval_seconds,
        timeout_seconds=timeout_seconds,
        sleep_fn=sleep_fn,
        now_fn=now_fn,
        on_poll=on_poll,
    )

    finished_at = datetime.now()
    duration_seconds = time.perf_counter() - started_perf
    if final_snapshot.rank is None or final_snapshot.score is None:
        raise ValueError("测评结束但未拿到完整 rank/score")

    return EvalRunResult(
        user_id=user_id,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        rank=final_snapshot.rank,
        score=final_snapshot.score,
        start_response=start_payload,
    )


def format_snapshot_log(attempt: int, elapsed: float, snapshot: RankSnapshot | None) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if snapshot is None:
        return (
            f"[{timestamp}] 第{attempt}次查询 | 已用时 {elapsed:.1f}s | "
            "未在榜单中找到该用户"
        )
    status = "测评中" if snapshot.is_evaluating else "已结束"
    return (
        f"[{timestamp}] 第{attempt}次查询 | 已用时 {elapsed:.1f}s | "
        f"状态={status} 排名={snapshot.rank} 分数={snapshot.score}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="自动启动测评任务并轮询状态，结束后输出最终分数和排名"
    )
    parser.add_argument("--user-id", default="z00832143", help="测评用户ID")
    parser.add_argument("--username", default="张嘉熙", help="测评用户名")
    parser.add_argument("--agent-ip", default="10.107.156.121", help="Agent服务IP")
    parser.add_argument("--agent-port", type=int, default=8080, help="Agent服务端口")
    parser.add_argument("--limit", type=int, default=2000, help="榜单查询上限")
    parser.add_argument("--interval", type=float, default=5.0, help="轮询间隔(秒)")
    parser.add_argument("--timeout", type=float, default=1800.0, help="总超时(秒)")
    parser.add_argument("--request-timeout", type=float, default=10.0, help="单次HTTP请求超时(秒)")
    parser.add_argument("--verify-ssl", action="store_true", help="开启SSL证书校验")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = EvalApiClient(
        verify_ssl=args.verify_ssl,
        request_timeout=args.request_timeout,
    )

    print(
        f"开始启动测评: user_id={args.user_id}, agent={args.agent_ip}:{args.agent_port}"
    )

    try:
        result = run(
            client=client,
            user_id=args.user_id,
            username=args.username,
            agent_ip=args.agent_ip,
            agent_port=args.agent_port,
            limit=args.limit,
            interval_seconds=args.interval,
            timeout_seconds=args.timeout,
            on_started=lambda payload: print(f"启动接口返回: {payload}"),
            on_poll=lambda attempt, elapsed, snapshot: print(
                format_snapshot_log(attempt, elapsed, snapshot)
            ),
            on_query_error=lambda exc: print(f"查询状态失败，稍后重试: {exc}"),
        )
    except requests.RequestException as exc:
        print(f"启动测评失败: {exc}")
        return 1
    except TimeoutError as exc:
        print(str(exc))
        return 1

    print("\n测评结束，最终结果如下：")
    print(f"开始时间: {result.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"耗时: {result.duration_seconds:.2f}s")
    print(f"最终排名: {result.rank}")
    print(f"最终分数: {result.score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
