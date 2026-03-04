#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def collect_tag_counts(cache_dir: Path) -> dict[str, int]:
    counter: Counter[str] = Counter()

    if not cache_dir.exists():
        return {}

    for json_path in sorted(cache_dir.rglob("*.json")):
        if not json_path.is_file():
            continue

        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue

        data = payload.get("data") if isinstance(payload, dict) else None
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            tags = item.get("tags")
            if not isinstance(tags, list):
                continue
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    counter[tag.strip()] += 1

    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 cache 目录中 JSON 文件里的房源 tag 频次")
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="JSON 文件目录，默认 cache",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="仅显示前 N 个 tag，默认 0 表示显示全部",
    )
    args = parser.parse_args()

    counts = collect_tag_counts(Path(args.cache_dir))
    if not counts:
        print("没有统计到任何 tag。")
        return

    total_count = sum(counts.values())
    print(f"tag 种类数: {len(counts)}")
    print(f"tag 总出现次数: {total_count}")
    print("-" * 30)

    items = list(counts.items())
    if args.top > 0:
        items = items[: args.top]

    for tag, count in items:
        print(f"{tag}\t{count}")


if __name__ == "__main__":
    main()
