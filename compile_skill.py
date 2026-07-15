from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diagnosis.agents.skill_compiler_agent import (
    SkillCompilerAbbreviation,
    SkillCompilerResult,
    compile_guideline_text,
)


ROOT_DIR = Path(__file__).resolve().parent
SKILLS_DIR = ROOT_DIR / "skills"
DEFAULT_MINERU_COMMAND = "mineru -p {input} -o {output} -b pipeline -m auto -l ch"


@dataclass
class GuidelineSource:
    full_text: str
    page_map: dict[str, Any] | None = None


SEARCH_GUIDELINE_SCRIPT = '''#!/usr/bin/env python3
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
        if all(pattern.search(line) for pattern in patterns):
            start = max(0, index - args.context)
            end = min(len(lines), index + args.context + 1)
            hit_lines.update(range(start, end))

    if not hit_lines:
        print("未找到匹配内容")
        return 1

    previous = -2
    for index in sorted(hit_lines):
        if index != previous + 1:
            print("\\n---")
        print(f"{index + 1}: {lines[index]}")
        previous = index
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile PDF guidelines into local skill directories.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pdf", type=Path, help="Source PDF file.")
    input_group.add_argument("--pdfs", type=Path, help="Directory containing source PDF files.")
    input_group.add_argument(
        "--full-text-md",
        type=Path,
        help="Existing MinerU Markdown output. When provided, PDF parsing is skipped.",
    )
    parser.add_argument("--skills-dir", type=Path, default=SKILLS_DIR, help="Directory containing local skills.")
    parser.add_argument(
        "--mineru-command",
        default=os.getenv("MINERU_COMMAND", DEFAULT_MINERU_COMMAND),
        help="MinerU command template. Use {input} and {output} placeholders.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing target skill directory.")
    return parser.parse_args()


def _validate_inputs(args: argparse.Namespace) -> None:
    if args.pdf and not args.pdf.exists():
        raise SystemExit(f"Error: PDF file does not exist: {args.pdf}")
    if args.pdfs and not args.pdfs.is_dir():
        raise SystemExit(f"Error: PDF directory does not exist: {args.pdfs}")
    if args.full_text_md and not args.full_text_md.exists():
        raise SystemExit(f"Error: Markdown file does not exist: {args.full_text_md}")


def _run_mineru(pdf_path: Path, command_template: str) -> GuidelineSource:
    output_dir = ROOT_DIR / "mineru"
    document_output_dir = output_dir / pdf_path.stem
    if document_output_dir.is_dir():
        markdown_path = _find_mineru_markdown(document_output_dir, pdf_path.stem)
        print(f"Using existing MinerU Markdown: {markdown_path}", flush=True)
        return _load_guideline_source(markdown_path, output_dir=document_output_dir)

    command_parts = shlex.split(command_template)
    if not command_parts:
        raise SystemExit("Error: MinerU command template cannot be empty.")

    mineru_bin = command_parts[0]
    if shutil.which(mineru_bin) is None:
        raise SystemExit(
            f"Error: MinerU command not found: {mineru_bin}. "
            "Install MinerU in the project virtual environment or pass --full-text-md."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"MinerU output root: {output_dir}", flush=True)
    command = [
        part.format(input=str(pdf_path), output=str(output_dir))
        for part in command_parts
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise SystemExit(
            "Error: MinerU parsing failed.\n"
            f"Output root: {output_dir}\n"
            f"Command: {' '.join(command)}\n"
            f"stderr:\n{completed.stderr.strip()}"
        )

    markdown_path = _find_mineru_markdown(document_output_dir, pdf_path.stem)
    return _load_guideline_source(markdown_path, output_dir=document_output_dir)


def _find_mineru_markdown(output_dir: Path, pdf_stem: str) -> Path:
    candidates = [path for path in output_dir.rglob("*.md") if path.is_file()]
    if not candidates:
        raise SystemExit(f"Error: MinerU did not produce a Markdown file under {output_dir}.")

    def score(path: Path) -> tuple[int, int]:
        name_score = 1 if pdf_stem.lower() in path.stem.lower() else 0
        return name_score, path.stat().st_size

    return max(candidates, key=score)


def _find_mineru_content_list(markdown_path: Path, search_dir: Path) -> Path | None:
    exact_path = markdown_path.with_name(f"{markdown_path.stem}_content_list.json")
    if exact_path.is_file():
        return exact_path

    candidates = [
        path
        for path in search_dir.rglob("*_content_list.json")
        if path.is_file()
        and not path.name.endswith("_content_list_v2.json")
        and markdown_path.stem.lower() in path.stem.lower()
    ]
    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_size)


def _normalize_source_text(value: str) -> str:
    markdown_prefixes = ("#", "-", "*", ">", "|")
    normalized = value.strip()
    while normalized.startswith(markdown_prefixes):
        normalized = normalized[1:].lstrip()
    return "".join(normalized.replace("\\", "").split())


def _block_texts(block: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    if isinstance(block.get("text"), str):
        texts.append(block["text"])
    if isinstance(block.get("list_items"), list):
        texts.extend(item for item in block["list_items"] if isinstance(item, str))
    for field in ("table_caption", "table_footnote", "image_caption", "image_footnote"):
        value = block.get(field)
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, list):
            texts.extend(item for item in value if isinstance(item, str))
    if isinstance(block.get("table_body"), str):
        texts.append(block["table_body"])
    return [text for text in texts if text.strip()]


def _find_markdown_line_range(
    markdown_lines: list[str],
    source_text: str,
    start_index: int,
) -> tuple[int, int] | None:
    target = _normalize_source_text(source_text)
    if not target:
        return None

    for line_index in range(start_index, len(markdown_lines)):
        combined = ""
        for end_index in range(line_index, min(len(markdown_lines), line_index + 12)):
            combined += _normalize_source_text(markdown_lines[end_index])
            if combined == target:
                return line_index + 1, end_index + 1
            if len(target) >= 20 and (
                target in combined or (combined in target and len(combined) >= 20)
            ):
                return line_index + 1, end_index + 1
            if len(combined) > len(target) * 2:
                break
    return None


def _build_page_map(
    full_text: str,
    content_list: list[dict[str, Any]],
    source_name: str,
) -> tuple[str, dict[str, Any]]:
    markdown_lines = full_text.splitlines()
    pages: dict[int, dict[str, Any]] = {}
    cursor = 0

    for block in content_list:
        page_idx = block.get("page_idx")
        if not isinstance(page_idx, int):
            continue
        page = pages.setdefault(
            page_idx,
            {
                "page_idx": page_idx,
                "pdf_page": page_idx + 1,
                "printed_page": None,
                "blocks": [],
            },
        )
        if block.get("type") == "page_number" and isinstance(block.get("text"), str):
            page["printed_page"] = block["text"].strip() or None
            continue
        if block.get("type") in {"header", "page_header", "page_footnote"}:
            continue

        line_ranges: list[tuple[int, int]] = []
        for text in _block_texts(block):
            matched_range = _find_markdown_line_range(markdown_lines, text, cursor)
            if matched_range is None:
                continue
            line_ranges.append(matched_range)
            cursor = matched_range[1]

        mapped_block: dict[str, Any] = {
            key: block[key]
            for key in ("type", "sub_type", "text_level", "bbox", "text", "list_items")
            if key in block
        }
        if line_ranges:
            mapped_block["original_markdown_line_start"] = min(item[0] for item in line_ranges)
            mapped_block["original_markdown_line_end"] = max(item[1] for item in line_ranges)
        page["blocks"].append(mapped_block)

    ordered_pages = [pages[index] for index in sorted(pages)]
    page_starts: dict[int, int] = {}
    for page in ordered_pages:
        starts = [
            block["original_markdown_line_start"]
            for block in page["blocks"]
            if "original_markdown_line_start" in block
        ]
        if starts:
            page_starts[min(starts)] = page["page_idx"]

    annotated_lines: list[str] = []
    original_to_annotated: dict[int, int] = {}
    page_marker_lines: dict[int, int] = {}
    for original_line, line in enumerate(markdown_lines, start=1):
        if original_line in page_starts:
            page = pages[page_starts[original_line]]
            marker = f"<!-- pdf_page: {page['pdf_page']}"
            if page["printed_page"]:
                marker += f", printed_page: {page['printed_page']}"
            marker += " -->"
            annotated_lines.append(marker)
            page_marker_lines[page["page_idx"]] = len(annotated_lines)
        annotated_lines.append(line)
        original_to_annotated[original_line] = len(annotated_lines)

    for page_index, page in enumerate(ordered_pages):
        marker_line = page_marker_lines.get(page["page_idx"])
        page["markdown_line_start"] = marker_line
        next_marker = None
        if page_index + 1 < len(ordered_pages):
            next_marker = page_marker_lines.get(ordered_pages[page_index + 1]["page_idx"])
        page["markdown_line_end"] = next_marker - 1 if next_marker else len(annotated_lines)
        for block in page["blocks"]:
            start = block.pop("original_markdown_line_start", None)
            end = block.pop("original_markdown_line_end", None)
            if start is not None:
                block["markdown_line_start"] = original_to_annotated[start]
                block["markdown_line_end"] = original_to_annotated[end]

    page_map = {
        "source_content_list": source_name,
        "page_idx_base": 0,
        "page_count": len(ordered_pages),
        "pages": ordered_pages,
    }
    return "\n".join(annotated_lines), page_map


def _load_guideline_source(markdown_path: Path, *, output_dir: Path | None = None) -> GuidelineSource:
    full_text = markdown_path.read_text(encoding="utf-8")
    content_list_path = _find_mineru_content_list(markdown_path, output_dir or markdown_path.parent)
    if content_list_path is None:
        return GuidelineSource(full_text=full_text)

    try:
        content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: cannot read MinerU content list: {content_list_path}: {exc}") from exc
    if not isinstance(content_list, list) or not all(isinstance(item, dict) for item in content_list):
        raise SystemExit(f"Error: MinerU content list must be a JSON array of objects: {content_list_path}")

    annotated_text, page_map = _build_page_map(full_text, content_list, content_list_path.name)
    print(f"MinerU page metadata loaded: {content_list_path}", flush=True)
    return GuidelineSource(full_text=annotated_text, page_map=page_map)


def _write_text(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _yaml_value(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _render_openai_yaml(result: SkillCompilerResult) -> str:
    return "\n".join(
        [
            f"display_name: {_yaml_value(result.display_name)}",
            f"short_description: {_yaml_value(result.short_description)}",
            f"default_prompt: {_yaml_value(result.default_prompt)}",
        ]
    )


def _render_skill_md(skill_name: str, result: SkillCompilerResult, *, has_page_map: bool) -> str:
    abbreviations = _render_abbreviations(result.common_abbreviations)
    page_map_workflow = (
        "3. 如需核对 PDF 物理页码、期刊印刷页码或版面位置，读取 "
        "`references/guideline-page-map.json`。\n"
        if has_page_map
        else ""
    )
    search_step = 4 if has_page_map else 3
    verification_step = 5 if has_page_map else 4
    page_map_resource = (
        "- `references/guideline-page-map.json`：从 MinerU `content_list.json` 提取的 PDF 页码、"
        "印刷页码、Markdown 行号和文本块坐标映射。\n"
        if has_page_map
        else ""
    )
    return f"""---
name: {skill_name}
description: {_yaml_value(result.skill_description)}
---

# {result.guideline_title}

## 工作流程

使用本 skill 回答与《{result.guideline_title}》相关的问题时，以 `references/guideline-full-text.md` 为原文依据。

1. 先读取 `references/recommendations-index.md`，定位相关{result.recommendations_label}、诊断标准、鉴别诊断、检查、治疗、监测、随访等重要信息和原文行号。
2. 再读取 `references/guideline-full-text.md` 中对应行号附近内容，补充适用人群、限制条件、解释依据和上下文。
{page_map_workflow}{search_step}. 如果问题没有明显对应{result.recommendations_label}，使用 `scripts/search_guideline.py` 进行关键词搜索。
{verification_step}. 如用户询问该文件之外的最新证据、药品获批状态、医保或现实可及性，应使用当前权威来源另行核实。

## 回答规则

- 明确说明回答依据《{result.guideline_title}》。
- 有{result.recommendations_label}编号时，列出对应编号。
- 有证据等级和推荐强度时，按索引或原文原样列出。
- 区分“指南/共识推荐、建议、可考虑、不推荐”和 Codex 自己的解释性总结。
- 不要编造原文没有给出的剂量、疗程、监测阈值、禁忌证或随访间隔。
- 对患者个体化决策，说明指南或共识不能替代临床医生评估；诊疗选择需结合疾病分期、活动度、并发症、既往治疗反应、感染风险、合并症和药物可及性。
- 如果原文和索引不一致，以 `guideline-full-text.md` 原文为准。

## 资源

- `references/recommendations-index.md`：LLM 根据全文自动生成的重要信息索引，用于定位{result.recommendations_label}、诊断标准、鉴别诊断、检查、治疗、监测和随访等关键内容。
- `references/guideline-full-text.md`：MinerU 解析得到的指南 Markdown 全文。
{page_map_resource}- `scripts/search_guideline.py`：关键词/正则搜索脚本。

{abbreviations}
"""


def _render_abbreviations(items: list[SkillCompilerAbbreviation]) -> str:
    if not items:
        return "## 常用缩写\n\n- 暂无自动提取的常用缩写。"
    lines = ["## 常用缩写", ""]
    for item in items:
        lines.append(f"- {item.abbreviation}：{item.meaning}")
    return "\n".join(lines)


def _write_skill_directory(
    skill_dir: Path,
    skill_name: str,
    full_text: str,
    result: SkillCompilerResult,
    page_map: dict[str, Any] | None,
) -> None:
    _write_text(skill_dir / "references" / "guideline-full-text.md", full_text)
    _write_text(skill_dir / "references" / "recommendations-index.md", result.recommendations_index_md)
    if page_map is not None:
        _write_text(
            skill_dir / "references" / "guideline-page-map.json",
            json.dumps(page_map, ensure_ascii=False, indent=2),
        )
    _write_text(
        skill_dir / "SKILL.md",
        _render_skill_md(skill_name, result, has_page_map=page_map is not None),
    )
    _write_text(skill_dir / "agents" / "openai.yaml", _render_openai_yaml(result))
    _write_text(skill_dir / "scripts" / "search_guideline.py", SEARCH_GUIDELINE_SCRIPT, executable=True)


def main() -> int:
    args = parse_args()
    _validate_inputs(args)

    if args.pdfs:
        input_paths = sorted(
            (
                path
                for path in args.pdfs.iterdir()
                if path.is_file() and path.suffix.lower() == ".pdf"
            ),
            key=lambda path: path.name,
        )
        if not input_paths:
            print(f"No PDF files found in directory: {args.pdfs}")
            return 0
    else:
        input_paths = [args.full_text_md or args.pdf]

    for input_path in input_paths:
        skill_name = input_path.stem
        skill_dir = args.skills_dir / skill_name
        if skill_dir.exists() and not args.force:
            print(f"Target skill directory already exists, skipping: {skill_dir}")
            continue

        print(f"Compiling guideline: {input_path}", flush=True)
        if args.full_text_md:
            source = _load_guideline_source(input_path)
        else:
            source = _run_mineru(input_path, args.mineru_command)

        result = compile_guideline_text(source.full_text)
        _write_skill_directory(
            skill_dir,
            skill_name,
            source.full_text,
            result,
            source.page_map,
        )

        print(f"Skill compiled: {skill_dir}")
        print("Recommendations index generated: references/recommendations-index.md")
        if source.page_map is not None:
            print("Page map generated: references/guideline-page-map.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
