from __future__ import annotations

import json
import re
from typing import Literal

from agents import Agent, Runner
from openai import OpenAI
from pydantic import BaseModel, Field

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    OPENAI_MODEL,
    SKILL_COMPILER_MODEL,
    SKILL_COMPILER_PROVIDER,
)


class SkillCompilerAbbreviation(BaseModel):
    abbreviation: str = Field(description="Abbreviation, for example CD")
    meaning: str = Field(description="Meaning exactly supported by the guideline text")


class SkillCompilerResult(BaseModel):
    guideline_title: str = Field(description="Official guideline or consensus title")
    skill_name_suggestion: str = Field(description="Lowercase kebab-case skill directory name")
    skill_description: str = Field(description="Front matter description for SKILL.md")
    display_name: str = Field(description="Display name for agents/openai.yaml")
    short_description: str = Field(description="Short description for agents/openai.yaml")
    default_prompt: str = Field(description="Default prompt for agents/openai.yaml")
    source_type: Literal["guideline", "consensus", "expert_opinion", "other"] = Field(
        description="Document type inferred from the source text"
    )
    recommendations_label: str = Field(
        description="Chinese label used by the document, for example 推荐意见 or 共识意见"
    )
    recommendations_index_md: str = Field(
        description=(
            "Complete Markdown content for references/recommendations-index.md, "
            "generated from important source text with line numbers"
        )
    )
    common_abbreviations: list[SkillCompilerAbbreviation] = Field(
        default_factory=list,
        description="Common abbreviations useful for this skill",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Known extraction limitations caused by OCR, line breaks, or incomplete evidence fields",
    )


SKILL_COMPILER_INSTRUCTIONS = """
You are a clinical guideline skill compiler.

Input:
- A Markdown full text extracted from a clinical PDF.
- Every source line is prefixed with a stable line number.

Task:
1. Identify the official title of the guideline, consensus, or expert document.
2. Propose a lowercase kebab-case skill directory name.
3. Generate the complete Markdown content for references/recommendations-index.md.
4. The index must be based on the full guideline text, not on a fixed table template.
5. Automatically decide which source-backed information is important for later skill use, including
   applicable recommendation or consensus items, diagnostic criteria, disease classification or
   activity assessment, differential diagnosis, examination suggestions, treatment principles,
   monitoring, follow-up, contraindications, cautions, and other clinically important guidance.
6. Organize the Markdown with useful headings and tables or bullet lists as appropriate for the
   source document. Every important item should include source line numbers whenever possible,
   so users can verify it in guideline-full-text.md.
7. Do not invent recommendation numbers, evidence levels, recommendation strengths, diseases,
   drugs, doses, thresholds, or follow-up intervals.
8. If OCR line breaks or missing context make an item unclear, explicitly mark that uncertainty in
   the Markdown index instead of filling unsupported fields.
9. Generate concise metadata for SKILL.md and agents/openai.yaml.
10. Prefer Chinese output for titles and user-facing skill metadata when the source document is Chinese.

The output must be valid structured data matching the requested schema.
""".strip()


def _number_guideline_text(full_text: str) -> str:
    return "\n".join(f"{index}: {line}" for index, line in enumerate(full_text.splitlines(), start=1))


def build_skill_compiler_agent() -> Agent:
    return Agent(
        name="Skill Compiler Agent",
        model=SKILL_COMPILER_MODEL or OPENAI_MODEL,
        instructions=SKILL_COMPILER_INSTRUCTIONS,
        output_type=SkillCompilerResult,
    )


def _build_compile_prompt(full_text: str) -> str:
    return (
        "Compile the following clinical document into a guideline skill metadata and "
        "recommendation index. Use only the numbered source text.\n\n"
        f"{_number_guideline_text(full_text)}"
    )


def _build_deepseek_system_prompt() -> str:
    schema = {
        "guideline_title": "string",
        "skill_name_suggestion": "lowercase kebab-case string",
        "skill_description": "string",
        "display_name": "string",
        "short_description": "string",
        "default_prompt": "string",
        "source_type": "guideline | consensus | expert_opinion | other",
        "recommendations_label": "string",
        "recommendations_index_md": "complete Markdown string",
        "common_abbreviations": [{"abbreviation": "string", "meaning": "string"}],
        "limitations": ["string"],
    }
    return (
        f"{SKILL_COMPILER_INSTRUCTIONS}\n\n"
        "Return only one valid JSON object. Do not wrap it in Markdown code fences. "
        "Do not include explanations before or after the JSON. The JSON object must match this shape:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return stripped[start : end + 1]


def _compile_guideline_text_with_deepseek(full_text: str) -> SkillCompilerResult:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is required when SKILL_COMPILER_PROVIDER=deepseek.")

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": _build_deepseek_system_prompt()},
            {"role": "user", "content": _build_compile_prompt(full_text)},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    try:
        return SkillCompilerResult.model_validate_json(content)
    except Exception:
        try:
            json_text = _extract_json_object(content)
            return SkillCompilerResult.model_validate_json(json_text)
        except Exception as exc:
            preview = content[:1000].replace("\n", "\\n")
            raise RuntimeError(f"DeepSeek did not return valid SkillCompilerResult JSON. Preview: {preview}") from exc


def compile_guideline_text(full_text: str) -> SkillCompilerResult:
    if SKILL_COMPILER_PROVIDER == "deepseek":
        return _compile_guideline_text_with_deepseek(full_text)

    return Runner.run_sync(build_skill_compiler_agent(), _build_compile_prompt(full_text)).final_output
