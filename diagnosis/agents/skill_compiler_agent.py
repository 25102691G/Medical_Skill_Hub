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


class SkillCompilerMetadata(BaseModel):
    guideline_title: str = Field(description="Official guideline or consensus title")
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
    common_abbreviations: list[SkillCompilerAbbreviation] = Field(
        default_factory=list,
        description="Common abbreviations useful for this skill",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Known extraction limitations caused by OCR, line breaks, or incomplete evidence fields",
    )


class SkillCompilerResult(SkillCompilerMetadata):
    recommendations_index_md: str = Field(
        description=(
            "Complete Markdown content for references/recommendations-index.md, "
            "generated from important source text"
        )
    )


DEEPSEEK_INDEX_CHUNK_CHARS = 30_000


SKILL_COMPILER_INSTRUCTIONS = """
You are a clinical guideline skill compiler.

Input:
- A Markdown full text extracted from a clinical PDF.

Task:
1. Identify the official title of the guideline, consensus, or expert document.
2. Generate the complete Markdown content for references/recommendations-index.md.
3. The index must be based on the full guideline text, not on a fixed table template.
4. Automatically decide which source-backed information is important for later skill use, including
   applicable recommendation or consensus items, diagnostic criteria, disease classification or
   activity assessment, differential diagnosis, examination suggestions, treatment principles,
   monitoring, follow-up, contraindications, cautions, and other clinically important guidance.
5. Organize the Markdown with useful headings and tables or bullet lists as appropriate for the
   source document.
6. Do not invent recommendation numbers, evidence levels, recommendation strengths, diseases,
   drugs, doses, thresholds, or follow-up intervals.
7. If OCR line breaks or missing context make an item unclear, explicitly mark that uncertainty in
   the Markdown index instead of filling unsupported fields.
8. Generate concise metadata for SKILL.md and agents/openai.yaml.

The output must be valid structured data matching the requested schema.
""".strip()


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
        "recommendation index. Use only the supplied source text.\n\n"
        f"{full_text}"
    )


def _build_deepseek_metadata_system_prompt() -> str:
    schema = {
        "guideline_title": "string",
        "skill_description": "string",
        "display_name": "string",
        "short_description": "string",
        "default_prompt": "string",
        "source_type": "guideline | consensus | expert_opinion | other",
        "recommendations_label": "string",
        "common_abbreviations": [{"abbreviation": "string", "meaning": "string"}],
        "limitations": ["string"],
    }
    return (
        "You generate concise metadata for a clinical guideline skill. Use only the supplied source text. "
        "Do not generate the recommendation index in this response. Prefer Chinese user-facing metadata "
        "when the source document is Chinese.\n\n"
        "Return only one valid JSON object. Do not wrap it in Markdown code fences. "
        "Do not include explanations before or after the JSON. The JSON object must match this shape:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def _build_deepseek_index_system_prompt() -> str:
    return """
You generate one source-backed Markdown fragment for a clinical guideline index.

Rules:
1. Use only the supplied source chunk.
2. Extract clinically important recommendations, diagnostic criteria, classifications, differential
   diagnoses, examinations, treatments, monitoring, follow-up, contraindications, and cautions.
3. Preserve recommendation numbers, evidence levels, strengths, drugs, doses, thresholds, and intervals
   exactly when present. Never invent missing information.
4. Be concise while retaining the important source-backed information in this chunk.
5. Return Markdown only, without a document-level H1 heading, JSON, code fences, or commentary.
6. Use useful H2/H3 headings and keep the source order.
""".strip()


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


def _chunk_guideline_text(
    full_text: str,
    *,
    max_chars: int = DEEPSEEK_INDEX_CHUNK_CHARS,
) -> list[str]:
    chunks: list[str] = []
    current_lines: list[str] = []
    current_chars = 0

    for line in full_text.splitlines():
        added_chars = len(line) + (1 if current_lines else 0)
        if current_lines and current_chars + added_chars > max_chars:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_chars = 0

        current_lines.append(line)
        current_chars += len(line) + (1 if len(current_lines) > 1 else 0)

    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


def _request_deepseek_text(
    client: OpenAI,
    *,
    system_prompt: str,
    user_prompt: str,
    purpose: str,
) -> str:
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    choice = response.choices[0]
    content = choice.message.content or ""
    if choice.finish_reason == "length":
        raise RuntimeError(f"DeepSeek output was truncated while generating {purpose}.")
    if not content.strip():
        raise RuntimeError(f"DeepSeek returned empty output while generating {purpose}.")
    return content


def _parse_deepseek_metadata(content: str) -> SkillCompilerMetadata:
    try:
        return SkillCompilerMetadata.model_validate_json(content)
    except Exception:
        try:
            json_text = _extract_json_object(content)
            return SkillCompilerMetadata.model_validate_json(json_text)
        except Exception as exc:
            preview = content[:1000].replace("\n", "\\n")
            raise RuntimeError(f"DeepSeek did not return valid skill metadata JSON. Preview: {preview}") from exc


def _strip_markdown_fence(content: str) -> str:
    stripped = content.strip()
    fenced_match = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return fenced_match.group(1).strip() if fenced_match else stripped


def _compile_guideline_text_with_deepseek(full_text: str) -> SkillCompilerResult:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is required when SKILL_COMPILER_PROVIDER=deepseek.")

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    metadata_content = _request_deepseek_text(
        client,
        system_prompt=_build_deepseek_metadata_system_prompt(),
        user_prompt=(
            "Generate skill metadata from the following complete clinical document.\n\n"
            f"{full_text}"
        ),
        purpose="skill metadata",
    )
    metadata = _parse_deepseek_metadata(metadata_content)

    chunks = _chunk_guideline_text(full_text)
    index_fragments: list[str] = []
    for chunk_number, chunk in enumerate(chunks, start=1):
        print(f"Generating DeepSeek index chunk {chunk_number}/{len(chunks)}", flush=True)
        fragment = _request_deepseek_text(
            client,
            system_prompt=_build_deepseek_index_system_prompt(),
            user_prompt=(
                f"Generate the Markdown index fragment for source chunk {chunk_number} of {len(chunks)}.\n\n"
                f"{chunk}"
            ),
            purpose=f"recommendation index chunk {chunk_number}/{len(chunks)}",
        )
        index_fragments.append(_strip_markdown_fence(fragment))

    recommendations_index_md = (
        f"# {metadata.guideline_title}重要信息索引\n\n"
        + "\n\n---\n\n".join(index_fragments)
    )
    return SkillCompilerResult(
        **metadata.model_dump(),
        recommendations_index_md=recommendations_index_md,
    )


def compile_guideline_text(full_text: str) -> SkillCompilerResult:
    if SKILL_COMPILER_PROVIDER == "deepseek":
        return _compile_guideline_text_with_deepseek(full_text)

    return Runner.run_sync(build_skill_compiler_agent(), _build_compile_prompt(full_text)).final_output
