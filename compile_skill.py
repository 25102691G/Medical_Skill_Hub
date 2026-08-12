from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from diagnosis.agents.skill_compiler_agent import (
    SkillCompilerAbbreviation,
    SkillCompilerResult,
    compile_guideline_text,
)


ROOT_DIR = Path(__file__).resolve().parent
SKILLS_DIR = ROOT_DIR / "skills"
DEFAULT_MINERU_COMMAND = "mineru -p {input} -o {output} -b pipeline -m auto -l ch"


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
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of guidelines compiled concurrently. MinerU remains serialized.",
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
    if args.workers < 1:
        raise SystemExit("Error: --workers must be at least 1.")


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
    completed = subprocess.run(command, check=False, text=True)
    if completed.returncode != 0:
        raise SystemExit(
            "Error: MinerU parsing failed.\n"
            f"Output root: {output_dir}\n"
            f"Command: {' '.join(command)}"
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


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


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
    primary_scope = result.skill_description.strip().rstrip("。")
    differential_diseases = "、".join(result.explicit_differential_diseases) or "无"
    description = (
        f"主要疾病及适用范围：{primary_scope}。"
        f"明确鉴别疾病：{differential_diseases}。"
    )
    return f"""---
name: {skill_name}
description: {_yaml_value(description)}
---

# {result.guideline_title}

## 工作流程

使用本 skill 回答与《{result.guideline_title}》相关的问题时，以 `references/guideline-full-text.md` 为原文依据。

1. 完整读取 `references/recommendations-index.md`，根据问题或病例阳性特征语义匹配相关{result.recommendations_label}、诊断标准、鉴别诊断、检查、治疗、监测、随访等重要信息。
2. 按索引条目的“原文位置”直接读取 `references/guideline-full-text.md` 对应行，核实适用人群、限制条件、解释依据和上下文。
3. 如用户询问该文件之外的最新证据、药品获批状态、医保或现实可及性，应使用当前权威来源另行核实。

## 回答规则

- 明确说明回答依据《{result.guideline_title}》。
- 有{result.recommendations_label}编号时，列出对应编号。
- 有证据等级和推荐强度时，按索引或原文原样列出。
- 区分“指南/共识推荐、建议、可考虑、不推荐”和 Codex 自己的解释性总结。
- 不要编造原文没有给出的剂量、疗程、监测阈值、禁忌证或随访间隔。
- 对患者个体化决策，说明指南或共识不能替代临床医生评估；诊疗选择需结合疾病分期、活动度、并发症、既往治疗反应、感染风险、合并症和药物可及性。
- 如果原文和索引不一致，以 `guideline-full-text.md` 原文为准。

## 资源

- `references/recommendations-index.md`：LLM 根据全文自动生成的重要信息索引；每个条目带有确定性的全文行号范围，用于直接定位原文。
- `references/guideline-full-text.md`：MinerU 解析得到的指南 Markdown 全文。

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

    mineru_lock = Lock()

    def compile_input(input_path: Path) -> None:
        skill_name = input_path.stem
        skill_dir = args.skills_dir / skill_name
        if skill_dir.exists() and not args.force:
            print(f"Target skill directory already exists, skipping: {skill_dir}")
            return

        print(f"Compiling guideline: {input_path}", flush=True)
        if args.full_text_md:
            full_text = input_path.read_text(encoding="utf-8")
        else:
            with mineru_lock:
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

    print(f"Compiling with {args.workers} workers; MinerU concurrency is limited to 1.", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(compile_input, input_paths))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
