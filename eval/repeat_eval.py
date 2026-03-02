import argparse
import csv
import time
from pathlib import Path
from typing import Any

try:
    from eval.run_eval import EvalApiClient, format_snapshot_log, run
except ModuleNotFoundError:
    from run_eval import EvalApiClient, format_snapshot_log, run

EVAL_TARGET_CASE_FILE = Path(__file__).resolve().parents[1] / "cache" / "eval_target_case.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按用例逐个执行测评并记录每个用例的分数/排名")
    parser.add_argument("--case-start", type=int, default=1, help="起始用例编号(含)，如 1 表示 EV-01")
    parser.add_argument("--case-end", type=int, default=60, help="结束用例编号(含)，如 60 表示 EV-60")
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


def _format_case_name(case_no: int) -> str:
    return f"EV-{case_no:02d}"


def _write_target_case(case_name: str) -> None:
    EVAL_TARGET_CASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVAL_TARGET_CASE_FILE.write_text(case_name + "\n", encoding="utf-8")


def _clear_target_case_file() -> None:
    if EVAL_TARGET_CASE_FILE.exists():
        EVAL_TARGET_CASE_FILE.unlink()


def to_csv_record(
    round_idx: int,
    case_name: str,
    status: str,
    result: Any | None,
    duration: float,
    start_time: str,
) -> dict[str, Any]:
    if result is None:
        return {
            "round": round_idx,
            "eval_case": case_name,
            "status": status,
            "start_time": start_time,
            "duration_seconds": f"{duration:.2f}",
            "score": "",
            "rank": "",
        }

    return {
        "round": round_idx,
        "eval_case": case_name,
        "status": status,
        "start_time": result.started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": f"{result.duration_seconds:.2f}",
        "score": result.score,
        "rank": result.rank,
    }


def write_csv(records: list[dict[str, Any]], output_path: str, append: bool) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "round",
        "eval_case",
        "status",
        "start_time",
        "duration_seconds",
        "score",
        "rank",
    ]
    mode = "a" if append else "w"
    needs_header = not append or not path.exists() or path.stat().st_size == 0

    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = build_parser().parse_args()
    if args.case_start <= 0:
        print("--case-start 必须大于 0")
        return 1
    if args.case_end < args.case_start:
        print("--case-end 必须大于等于 --case-start")
        return 1

    cases = [_format_case_name(i) for i in range(args.case_start, args.case_end + 1)]
    client = EvalApiClient(verify_ssl=args.verify_ssl, request_timeout=args.request_timeout)
    records: list[dict[str, Any]] = []

    try:
        for idx, case_name in enumerate(cases, start=1):
            print(f"\n===== 第 {idx}/{len(cases)} 轮测评 | 用例 {case_name} =====")
            _write_target_case(case_name)
            print(f"已写入标记文件: {EVAL_TARGET_CASE_FILE} -> {case_name}")
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
                    case_name,
                    "success",
                    result,
                    result.duration_seconds,
                    round_start_time,
                )
                print(
                    f"用例 {case_name} 结束: 开始时间={record['start_time']} "
                    f"耗时={record['duration_seconds']}s 分数={record['score']} 排名={record['rank']}"
                )
            except Exception as exc:
                duration = time.perf_counter() - round_started_perf
                record = to_csv_record(
                    idx,
                    case_name,
                    f"failure: {exc}",
                    None,
                    duration,
                    round_start_time,
                )
                print(f"用例 {case_name} 失败: {exc}")

            records.append(record)

            if idx < len(cases) and args.gap > 0:
                print(f"等待 {args.gap}s 后开始下一轮...")
                time.sleep(args.gap)
    finally:
        _clear_target_case_file()
        print(f"已清理标记文件: {EVAL_TARGET_CASE_FILE}")

    write_csv(records, args.output, args.append)

    print("\n全部测评完成，汇总如下:")
    for r in records:
        print(
            f"轮次={r['round']} 用例={r['eval_case']} 状态={r['status']} 开始时间={r['start_time']} "
            f"耗时={r['duration_seconds']}s 分数={r['score']} 排名={r['rank']}"
        )
    print(f"记录已写入: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
