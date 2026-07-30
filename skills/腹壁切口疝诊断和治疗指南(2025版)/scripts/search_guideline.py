#!/usr/bin/env python3
"""在当前指南 skill 资源中搜索关键词。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "references"
DEFAULT_FILE = REFERENCE_DIR / "guideline-full-text.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="搜索指南 skill 参考文件")
    parser.add_argument("keywords", nargs="+", help="关键词或正则表达式")
    parser.add_argument("--file", default="guideline-full-text.md", help="references 下的文件名")
    parser.add_argument("--context", type=int, default=2, help="命中行前后的上下文行数")
    parser.add_argument("--regex", action="store_true", help="按正则表达式匹配关键词")
    return parser.parse_args()


def compile_patterns(keywords: list[str], regex: bool) -> list[re.Pattern[str]]:
    flags = re.IGNORECASE
    patterns = []
    for keyword in keywords:
        pattern = keyword if regex else re.escape(keyword)
        patterns.append(re.compile(pattern, flags))
    return patterns


def main() -> int:
    args = parse_args()
    target = REFERENCE_DIR / args.file
    if not target.exists():
        raise SystemExit(f"文件不存在：{target}")

    lines = target.read_text(encoding="utf-8").splitlines()
    patterns = compile_patterns(args.keywords, args.regex)
    hit_lines: set[int] = set()

    for index, line in enumerate(lines):
        if any(pattern.search(line) for pattern in patterns):
            start = max(0, index - args.context)
            end = min(len(lines), index + args.context + 1)
            hit_lines.update(range(start, end))

    if not hit_lines:
        print("未找到匹配内容")
        return 1

    previous = -2
    for index in sorted(hit_lines):
        if index != previous + 1:
            print("\n---")
        print(lines[index])
        previous = index
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
