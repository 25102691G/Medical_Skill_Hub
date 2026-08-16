from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = SKILL_DIR / "references" / "recommendations-index.md"
FULL_TEXT_PATH = SKILL_DIR / "references" / "guideline-full-text.md"
HEADING_ID_PATTERN = re.compile(r"H\d{4}")
SOURCE_ID_PATTERN = re.compile(r"L(\d{6})-L(\d{6})")


@dataclass
class IndexEntry:
    entry_id: str
    heading_id: str
    section: str
    subsection: str | None
    content: str
    source_ids: list[str]


def _parse_index() -> list[IndexEntry]:
    lines = INDEX_PATH.read_text(encoding="utf-8").splitlines()
    entries: list[IndexEntry] = []
    heading_ids: dict[tuple[str, str | None], str] = {}
    section = ""
    subsection: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("## "):
            section = line[3:].strip()
            subsection = None
        elif line.startswith("### "):
            subsection = line[4:].strip()
        elif line.startswith("- "):
            heading_key = (section, subsection)
            if heading_key not in heading_ids:
                heading_ids[heading_key] = f"H{len(heading_ids) + 1:04d}"
            content = line[2:].strip()
            source_ids: list[str] = []
            if index + 1 < len(lines) and lines[index + 1].startswith(
                "  - 原文位置："
            ):
                source_ids = SOURCE_ID_PATTERN.findall(lines[index + 1])
                source_ids = [f"L{start}-L{end}" for start, end in source_ids]
                index += 1
            entries.append(
                IndexEntry(
                    entry_id=f"E{len(entries) + 1:04d}",
                    heading_id=heading_ids[heading_key],
                    section=section,
                    subsection=subsection,
                    content=content,
                    source_ids=source_ids,
                )
            )
        index += 1
    return entries


def _print_catalog() -> None:
    entries = _parse_index()
    entry_counts = Counter(entry.heading_id for entry in entries)
    print("<INDEX_CATALOG>")
    seen: set[str] = set()
    for entry in entries:
        if entry.heading_id in seen:
            continue
        seen.add(entry.heading_id)
        heading = entry.section
        if entry.subsection:
            heading = f"{heading} > {entry.subsection}"
        print(f"{entry.heading_id} | entries={entry_counts[entry.heading_id]} | {heading}")
    print("</INDEX_CATALOG>")


def _print_entries(heading_ids: list[str]) -> None:
    requested_ids = list(dict.fromkeys(heading_ids))
    if any(not HEADING_ID_PATTERN.fullmatch(heading_id) for heading_id in requested_ids):
        raise SystemExit("Invalid heading ID. Expected an ID such as H0001.")

    entries = _parse_index()
    available_ids = {entry.heading_id for entry in entries}
    unknown_ids = [
        heading_id for heading_id in requested_ids if heading_id not in available_ids
    ]
    if unknown_ids:
        raise SystemExit(f"Unknown heading ID: {', '.join(unknown_ids)}")

    for entry in entries:
        if entry.heading_id not in requested_ids:
            continue
        print(
            f'<INDEX_ENTRY id="{entry.entry_id}" heading_id="{entry.heading_id}">'
        )
        print(f"section: {entry.section}")
        if entry.subsection:
            print(f"subsection: {entry.subsection}")
        print(entry.content)
        print(f"source_ids: {', '.join(entry.source_ids)}")
        print("</INDEX_ENTRY>")
        print()


def _read_sources(source_ids: list[str]) -> None:
    lines = FULL_TEXT_PATH.read_text(encoding="utf-8").splitlines()
    for source_id in dict.fromkeys(source_ids):
        match = SOURCE_ID_PATTERN.fullmatch(source_id)
        if not match:
            raise SystemExit(f"Invalid source ID: {source_id}")
        start_line, end_line = (int(value) for value in match.groups())
        if start_line < 1 or end_line < start_line or end_line > len(lines):
            raise SystemExit(f"Source ID is outside the guideline full text: {source_id}")
        print(f'<SOURCE_BLOCK id="{source_id}">')
        for line_number in range(start_line, end_line + 1):
            print(f"{line_number}: {lines[line_number - 1]}")
        print("</SOURCE_BLOCK>")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expose one guideline index for LLM semantic matching and read exact sources."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("catalog")
    entries_parser = subparsers.add_parser("entries")
    entries_parser.add_argument("--heading-id", action="append", required=True)
    sources_parser = subparsers.add_parser("sources")
    sources_parser.add_argument("--source-id", action="append", required=True)
    args = parser.parse_args()

    if args.command == "catalog":
        _print_catalog()
    elif args.command == "entries":
        _print_entries(args.heading_id)
    else:
        _read_sources(args.source_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
