import argparse
import csv
import time
from pathlib import Path
from typing import Any

try:
    from eval.run_eval import EvalApiClient, format_snapshot_log, run
except ModuleNotFoundError:
    from run_eval import EvalApiClient, format_snapshot_log, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="重复执行测评并记录每次分数/排名")
    parser.add_argument("--rounds", type=int, default=60, help="测评轮数")
    parser.add_argument("--gap", type=float, default=3.0, help="轮与轮之间等待秒数")
    parser.add_argument("--output", default="eval/eval_runs.csv", help="CSV输出路径")
    parser.add_argument("--append", action="store_true", help="追加写入CSV")

    parser.add_argument("--user-id", default="z00832143", help="测评用户ID")
    parser.add_argument("--username", default="张嘉熙", help="测评用户名")
    parser.add_argument("--agent-ip", default="10.107.156.121", help="Agent服务IP")
    parser.add_argument("--agent-port", type=int, default=8080, help="Agent服务端口")
    parser.add_argument("--limit", type=int, default=2000, help="榜单查询上限")
    parser.add_argument("--interval", type=float, default=5.0, help="轮询间隔(秒)")
    parser.add_argument("--timeout", type=float, default=1800.0, help="单轮总超时(秒)")
    parser.add_argument("--request-timeout", type=float, default=10.0, help="单次HTTP请求超时(秒)")
    parser.add_argument("--verify-ssl", action="store_true", help="开启SSL证书校验")
    return parser


def to_csv_record(
    round_idx: int,
    status: str,
    result: Any | None,
    duration: float,
    start_time: str,
) -> dict[str, Any]:
    if result is None:
        return {
            "round": round_idx,
            "status": status,
            "start_time": start_time,
            "duration_seconds": f"{duration:.2f}",
            "score": "",
            "rank": "",
        }

    return {
        "round": round_idx,
        "status": status,
        "start_time": result.started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": f"{result.duration_seconds:.2f}",
        "score": result.score,
        "rank": result.rank,
    }


def write_csv(records: list[dict[str, Any]], output_path: str, append: bool) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["round", "status", "start_time", "duration_seconds", "score", "rank"]
    mode = "a" if append else "w"
    needs_header = not append or not path.exists() or path.stat().st_size == 0

    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = build_parser().parse_args()
    if args.rounds <= 0:
        print("--rounds 必须大于 0")
        return 1

    client = EvalApiClient(verify_ssl=args.verify_ssl, request_timeout=args.request_timeout)
    records: list[dict[str, Any]] = []

    for idx in range(1, args.rounds + 1):
        print(f"\n===== 第 {idx}/{args.rounds} 轮测评 =====")
        round_started_perf = time.perf_counter()
        round_start_time = time.strftime("%Y-%m-%d %H:%M:%S")

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
            record = to_csv_record(
                idx,
                "success",
                result,
                result.duration_seconds,
                round_start_time,
            )
            print(
                f"第 {idx} 轮结束: 开始时间={record['start_time']} 耗时={record['duration_seconds']}s "
                f"分数={record['score']} 排名={record['rank']}"
            )
        except Exception as exc:
            duration = time.perf_counter() - round_started_perf
            record = to_csv_record(
                idx,
                f"failure: {exc}",
                None,
                duration,
                round_start_time,
            )
            print(f"第 {idx} 轮失败: {exc}")

        records.append(record)

        if idx < args.rounds and args.gap > 0:
            print(f"等待 {args.gap}s 后开始下一轮...")
            time.sleep(args.gap)

    write_csv(records, args.output, args.append)

    print("\n全部测评完成，汇总如下:")
    for r in records:
        print(
            f"轮次={r['round']} 状态={r['status']} 开始时间={r['start_time']} "
            f"耗时={r['duration_seconds']}s 分数={r['score']} 排名={r['rank']}"
        )
    print(f"记录已写入: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
