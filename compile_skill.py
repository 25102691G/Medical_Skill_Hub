from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from diagnosis.agents.skill_compiler_agent import (
    SkillCompilerAbbreviation,
    SkillCompilerResult,
    compile_guideline_text,
)


ROOT_DIR = Path(__file__).resolve().parent
SKILLS_DIR = ROOT_DIR / "skills"
DEFAULT_MINERU_COMMAND = "mineru -p {input} -o {output} -b pipeline -m auto -l ch"


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
        print(lines[index])
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


def _run_mineru(pdf_path: Path, command_template: str) -> str:
    output_dir = ROOT_DIR / "mineru"
    document_output_dir = output_dir / pdf_path.stem
    if document_output_dir.is_dir():
        markdown_path = _find_mineru_markdown(document_output_dir, pdf_path.stem)
        print(f"Using existing MinerU Markdown: {markdown_path}", flush=True)
        return markdown_path.read_text(encoding="utf-8")

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
    return markdown_path.read_text(encoding="utf-8")


def _find_mineru_markdown(output_dir: Path, pdf_stem: str) -> Path:
    candidates = [path for path in output_dir.rglob("*.md") if path.is_file()]
    if not candidates:
        raise SystemExit(f"Error: MinerU did not produce a Markdown file under {output_dir}.")

    def score(path: Path) -> tuple[int, int]:
        name_score = 1 if pdf_stem.lower() in path.stem.lower() else 0
        return name_score, path.stat().st_size

    return max(candidates, key=score)


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
            "interface:",
            f"  display_name: {_yaml_value(result.display_name)}",
            f"  short_description: {_yaml_value(result.short_description)}",
            f"  default_prompt: {_yaml_value(result.default_prompt)}",
        ]
    )


def _render_skill_md(skill_name: str, result: SkillCompilerResult) -> str:
    abbreviations = _render_abbreviations(result.common_abbreviations)
    return f"""---
name: {skill_name}
description: {_yaml_value(result.skill_description)}
---

# {result.guideline_title}

## 工作流程

使用本 skill 回答与《{result.guideline_title}》相关的问题时，以 `references/guideline-full-text.md` 为原文依据。

1. 先读取 `references/recommendations-index.md`，定位相关{result.recommendations_label}、诊断标准、鉴别诊断、检查、治疗、监测、随访等重要信息。
2. 再读取 `references/guideline-full-text.md` 中的相关内容，补充适用人群、限制条件、解释依据和上下文。
3. 如果问题没有明显对应{result.recommendations_label}，使用 `scripts/search_guideline.py` 进行关键词搜索。
4. 如用户询问该文件之外的最新证据、药品获批状态、医保或现实可及性，应使用当前权威来源另行核实。

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
- `scripts/search_guideline.py`：关键词/正则搜索脚本。

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
) -> None:
    _write_text(skill_dir / "references" / "guideline-full-text.md", full_text)
    _write_text(skill_dir / "references" / "recommendations-index.md", result.recommendations_index_md)
    _write_text(
        skill_dir / "SKILL.md",
        _render_skill_md(skill_name, result),
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
            full_text = input_path.read_text(encoding="utf-8")
        else:
            full_text = _run_mineru(input_path, args.mineru_command)

        result = compile_guideline_text(full_text)
        _write_skill_directory(
            skill_dir,
            skill_name,
            full_text,
            result,
        )

        print(f"Skill compiled: {skill_dir}")
        print("Recommendations index generated: references/recommendations-index.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
