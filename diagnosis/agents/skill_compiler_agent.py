from __future__ import annotations

import json
import re

from agents import Agent, Runner
from openai import OpenAI
from pydantic import BaseModel, Field

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_THINKING,
    OPENAI_MODEL,
    SKILL_COMPILER_MODEL,
    SKILL_COMPILER_PROVIDER,
)


class SkillCompilerAbbreviation(BaseModel):
    abbreviation: str = Field(description="Abbreviation, for example CD")
    meaning: str = Field(description="Meaning exactly supported by the guideline text")


class SkillCompilerMetadata(BaseModel):
    guideline_title: str = Field(description="Official guideline or consensus title")
    skill_description: str = Field(
        description=(
            "Primary-disease front matter description for SKILL.md. Include the Chinese and English "
            "primary disease names, common abbreviations, and applicable clinical scope. Do not "
            "include a broad disease category."
        )
    )
    display_name: str = Field(description="Display name for agents/openai.yaml")
    short_description: str = Field(description="Short description for agents/openai.yaml")
    default_prompt: str = Field(description="Default prompt for agents/openai.yaml")
    recommendations_label: str = Field(
        description="Chinese label used by the document, for example 推荐意见 or 共识意见"
    )
    common_abbreviations: list[SkillCompilerAbbreviation] = Field(
        default_factory=list,
        description="Common abbreviations useful for this skill",
    )
    explicit_differential_diseases: list[str] = Field(
        description=(
            "Diseases explicitly covered by substantive differential diagnosis, exclusion, or "
            "disease-comparison content in the guideline. Exclude incidental mentions, references, "
            "comorbidities, and treatment-only mentions."
        ),
    )


class SkillCompilerIndexEntry(BaseModel):
    section: str = Field(description="H2 section name for this index entry")
    subsection: str | None = Field(
        default=None,
        description="Optional H3 subsection name for this index entry",
    )
    content: str = Field(
        description=(
            "One concise source-backed clinical index item without a Markdown list marker"
        )
    )
    source_block_ids: list[str] = Field(
        description="One or more supplied source block IDs that directly support this item"
    )


class SkillCompilerDraft(SkillCompilerMetadata):
    recommendations_index_entries: list[SkillCompilerIndexEntry] = Field(
        description="Clinically important index entries kept in source order"
    )


class SkillCompilerIndexChunk(BaseModel):
    entries: list[SkillCompilerIndexEntry]


class SkillCompilerResult(SkillCompilerMetadata):
    recommendations_index_md: str


class GuidelineSourceBlock(BaseModel):
    block_id: str
    start_line: int
    end_line: int
    text: str


DEEPSEEK_INDEX_CHUNK_CHARS = 12_000


SKILL_COMPILER_INSTRUCTIONS = """
## SKILL COMPILER INSTRUCTIONS

You are a clinical guideline skill compiler.

### 1. Input

You will receive a Markdown full text extracted from a clinical PDF.

### 2. Objective

1. Identify the official title of the guideline, consensus, or expert document.
2. Generate structured entries for references/recommendations-index.md.
3. The index must be based on the full guideline text, not on a fixed table template.
4. Automatically decide which source-backed information is important for later skill use, including
   applicable recommendation or consensus items, diagnostic criteria, disease classification or
   activity assessment, differential diagnosis, examination suggestions, treatment principles,
   monitoring, follow-up, contraindications, cautions, and other clinically important guidance.
5. Organize the entries with useful section and optional subsection names appropriate for the source
   document.
6. Do not invent recommendation numbers, evidence levels, recommendation strengths, diseases,
   drugs, doses, thresholds, or follow-up intervals.
7. If OCR line breaks or missing context make an item unclear, explicitly mark that uncertainty in
   the entry content instead of filling unsupported fields.
8. Generate concise metadata for SKILL.md and agents/openai.yaml. The skill description must identify
   the guideline's primary disease in Chinese and English, common abbreviations when supported, and
   applicable clinical scope. Do not assign or mention a broad disease category.
9. Populate explicit_differential_diseases only with diseases for which the guideline contains
   substantive differential diagnosis, explicit exclusion, or disease-comparison content. Do not
   include diseases mentioned only in background text, references, comorbidity discussions, treatment
   indications, adverse effects, or other non-diagnostic contexts. Return an empty list when the
   guideline contains no explicit differential disease.
10. The source document is divided into SOURCE_BLOCK elements with stable IDs derived from the exact
   Markdown line ranges. For every index entry, return all and only the supplied source_block_ids that
   directly support it. Never create, alter, or infer a source block ID.

### 3. Output Requirements

Return structured index entries rather than pre-rendered Markdown. Keep entries in source order. Use a
concise single paragraph for each content value and do not include a Markdown list marker in it. The
output must be valid structured data matching the requested schema.
""".strip()


def build_skill_compiler_agent() -> Agent:
    return Agent(
        name="Skill Compiler Agent",
        model=SKILL_COMPILER_MODEL or OPENAI_MODEL,
        instructions=SKILL_COMPILER_INSTRUCTIONS,
        output_type=SkillCompilerDraft,
    )


def _build_source_blocks(
    full_text: str,
    *,
    max_chars: int | None = None,
) -> list[GuidelineSourceBlock]:
    lines = full_text.splitlines()
    blocks: list[GuidelineSourceBlock] = []
    start_line: int | None = None
    block_lines: list[str] = []

    for index, line in enumerate(lines, start=1):
        if not line.strip():
            if start_line is not None:
                end_line = index - 1
                block_id = f"L{start_line:06d}-L{end_line:06d}"
                blocks.append(
                    GuidelineSourceBlock(
                        block_id=block_id,
                        start_line=start_line,
                        end_line=end_line,
                        text="\n".join(block_lines),
                    )
                )
                start_line = None
                block_lines = []
            continue

        if start_line is None:
            start_line = index

        if max_chars and block_lines:
            candidate = GuidelineSourceBlock(
                block_id=f"L{start_line:06d}-L{index:06d}",
                start_line=start_line,
                end_line=index,
                text="\n".join([*block_lines, line]),
            )
            if len(_render_source_blocks([candidate])) > max_chars:
                end_line = index - 1
                block_id = f"L{start_line:06d}-L{end_line:06d}"
                blocks.append(
                    GuidelineSourceBlock(
                        block_id=block_id,
                        start_line=start_line,
                        end_line=end_line,
                        text="\n".join(block_lines),
                    )
                )
                start_line = index
                block_lines = []
        block_lines.append(line)

    if start_line is not None:
        end_line = len(lines)
        block_id = f"L{start_line:06d}-L{end_line:06d}"
        blocks.append(
            GuidelineSourceBlock(
                block_id=block_id,
                start_line=start_line,
                end_line=end_line,
                text="\n".join(block_lines),
            )
        )
    return blocks


def _render_source_blocks(blocks: list[GuidelineSourceBlock]) -> str:
    return "\n\n".join(
        f'<SOURCE_BLOCK id="{block.block_id}">\n{block.text}\n</SOURCE_BLOCK>'
        for block in blocks
    )


def _render_recommendations_index(
    guideline_title: str,
    entries: list[SkillCompilerIndexEntry],
    source_blocks: list[GuidelineSourceBlock],
) -> str:
    valid_block_ids = {block.block_id for block in source_blocks}
    lines = [f"# {guideline_title}重要信息索引"]
    current_section: str | None = None
    current_subsection: str | None = None

    for entry in entries:
        source_block_ids = []
        for block_id in entry.source_block_ids:
            if block_id not in valid_block_ids and len(block_id) == 7 and block_id.startswith("L"):
                single_line_block_id = f"{block_id}-{block_id}"
                if single_line_block_id in valid_block_ids:
                    block_id = single_line_block_id
            if block_id not in valid_block_ids:
                range_match = re.fullmatch(r"L(\d{6})-L(\d{6})", block_id)
                if range_match:
                    range_start, range_end = map(int, range_match.groups())
                    covered_blocks = [
                        block
                        for block in source_blocks
                        if block.start_line >= range_start and block.end_line <= range_end
                    ]
                    if (
                        covered_blocks
                        and covered_blocks[0].start_line == range_start
                        and covered_blocks[-1].end_line == range_end
                    ):
                        source_block_ids.extend(block.block_id for block in covered_blocks)
                        continue
            source_block_ids.append(block_id)
        invalid_block_ids = [
            block_id
            for block_id in source_block_ids
            if block_id not in valid_block_ids
        ]
        if not source_block_ids or invalid_block_ids:
            raise RuntimeError(
                "Skill compiler returned an index entry without valid source blocks: "
                f"{entry.content}; invalid IDs: {invalid_block_ids}"
            )

        section = entry.section.strip()
        subsection = entry.subsection.strip() if entry.subsection else None
        if section != current_section:
            lines.extend(["", f"## {section}"])
            current_section = section
            current_subsection = None
        if subsection != current_subsection:
            if subsection:
                lines.extend(["", f"### {subsection}"])
            current_subsection = subsection

        content = " ".join(entry.content.splitlines()).strip()
        source_ids = "、".join(f"`{block_id}`" for block_id in source_block_ids)
        lines.extend(["", f"- {content}", f"  - 原文位置：{source_ids}"])

    return "\n".join(lines)


def _build_compiler_result(
    draft: SkillCompilerDraft,
    source_blocks: list[GuidelineSourceBlock],
) -> SkillCompilerResult:
    return SkillCompilerResult(
        **draft.model_dump(exclude={"recommendations_index_entries"}),
        recommendations_index_md=_render_recommendations_index(
            draft.guideline_title,
            draft.recommendations_index_entries,
            source_blocks,
        ),
    )


def _build_compile_prompt(full_text: str) -> str:
    source_blocks = _build_source_blocks(full_text)
    return (
        "## Task\n\n"
        "Compile the following clinical document into a guideline skill metadata and "
        "recommendation index. Use only the supplied source text.\n\n"
        "<SOURCE_DOCUMENT>\n"
        f"{_render_source_blocks(source_blocks)}\n"
        "</SOURCE_DOCUMENT>"
    )


def _build_deepseek_metadata_system_prompt() -> str:
    schema = {
        "guideline_title": "示例指南",
        "skill_description": "用于示例疾病（Example disease，ED）的诊断、鉴别诊断、治疗和随访。",
        "display_name": "示例指南",
        "short_description": "查询示例指南中的临床建议",
        "default_prompt": "请使用示例指南回答临床问题",
        "recommendations_label": "推荐意见",
        "common_abbreviations": [{"abbreviation": "CD", "meaning": "Crohn disease"}],
        "explicit_differential_diseases": ["示例鉴别疾病（Example differential disease）"],
    }
    return f"""
## DEEPSEEK METADATA INSTRUCTIONS

You generate concise metadata for a clinical guideline skill.

### 1. Source Boundaries

Use only the supplied source text. Do not generate the recommendation index in this response.

### 2. Metadata Requirements

Prefer Chinese user-facing metadata when the source document is Chinese.

The skill_description must identify the guideline's primary disease in Chinese and English, include
common abbreviations when supported, describe the applicable clinical scope, and must not assign or
mention a broad disease category.

Populate explicit_differential_diseases only with diseases for which the guideline contains substantive
differential diagnosis, explicit exclusion, or disease-comparison content. Do not include incidental
mentions, references, comorbidities, treatment-only mentions, or adverse-effect mentions. Use an empty
list when no disease meets this requirement.

### 3. Output Requirements

Return only one valid JSON object. Do not wrap it in Markdown code fences. Do not include explanations
before or after the JSON.

The following JSON object is the required output format example:

<OUTPUT_FORMAT_EXAMPLE>
{json.dumps(schema, ensure_ascii=False, indent=2)}
</OUTPUT_FORMAT_EXAMPLE>
""".strip()


def _build_deepseek_index_system_prompt() -> str:
    return """
## DEEPSEEK INDEX INSTRUCTIONS

You generate structured source-backed entries for a clinical guideline index.

### 1. Source Boundaries

Use only the supplied SOURCE_BLOCK elements. Each element's ID is derived from the exact Markdown line
range in the saved guideline full text.

### 2. Content Requirements

1. Extract clinically important recommendations, diagnostic criteria, classifications, differential
   diagnoses, examinations, treatments, monitoring, follow-up, contraindications, and cautions.
2. Preserve recommendation numbers, evidence levels, strengths, drugs, doses, thresholds, and intervals
   exactly when present. Never invent missing information.
3. For every entry, return all and only the supplied source_block_ids that directly support it. Never
   create, alter, or infer an ID.
4. Be concise while retaining the important source-backed information in this chunk.
5. Use useful section and optional subsection names and keep the source order.

### 3. Output Requirements

Put the entries in one valid JSON object. Each content value must be one concise paragraph without a
Markdown list marker. Do not add Markdown code fences, commentary, or text outside the JSON object.

#### Example JSON Output

<OUTPUT_FORMAT_EXAMPLE>
{"entries":[{"section":"诊断","subsection":null,"content":"源文支持的诊断标准","source_block_ids":["L000035-L000039"]}]}
</OUTPUT_FORMAT_EXAMPLE>
""".strip()


def _chunk_source_blocks(
    source_blocks: list[GuidelineSourceBlock],
    *,
    max_chars: int = DEEPSEEK_INDEX_CHUNK_CHARS,
) -> list[list[GuidelineSourceBlock]]:
    chunks: list[list[GuidelineSourceBlock]] = []
    current_blocks: list[GuidelineSourceBlock] = []
    current_chars = 0

    for block in source_blocks:
        rendered_block = _render_source_blocks([block])
        added_chars = len(rendered_block) + (2 if current_blocks else 0)
        if current_blocks and current_chars + added_chars > max_chars:
            chunks.append(current_blocks)
            current_blocks = []
            current_chars = 0

        current_blocks.append(block)
        current_chars += added_chars

    if current_blocks:
        chunks.append(current_blocks)
    return chunks


def _request_deepseek_text(
    client: OpenAI,
    *,
    system_prompt: str,
    user_prompt: str,
    purpose: str,
    max_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_body={
            "thinking": {
                "type": "enabled" if DEEPSEEK_THINKING else "disabled",
            }
        },
    )
    choice = response.choices[0]
    content = choice.message.content or ""
    if choice.finish_reason == "length":
        raise RuntimeError(f"DeepSeek output was truncated while generating {purpose}.")
    if not content.strip():
        raise RuntimeError(f"DeepSeek returned empty output while generating {purpose}.")
    return content.strip()


def _parse_deepseek_metadata(content: str) -> SkillCompilerMetadata:
    try:
        return SkillCompilerMetadata.model_validate_json(content)
    except Exception as exc:
        preview = content[:1000].replace("\n", "\\n")
        raise RuntimeError(
            f"DeepSeek did not return valid skill metadata JSON. Preview: {preview}"
        ) from exc


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
            "## Task\n\n"
            "Generate skill metadata from the following complete clinical document.\n\n"
            "<SOURCE_DOCUMENT>\n"
            f"{full_text}\n"
            "</SOURCE_DOCUMENT>"
        ),
        purpose="skill metadata",
        max_tokens=8192,
    )
    metadata = _parse_deepseek_metadata(metadata_content)

    source_blocks = _build_source_blocks(
        full_text,
        max_chars=DEEPSEEK_INDEX_CHUNK_CHARS,
    )
    chunks = _chunk_source_blocks(source_blocks)
    index_entries: list[SkillCompilerIndexEntry] = []
    for chunk_number, chunk in enumerate(chunks, start=1):
        print(f"Generating DeepSeek index chunk {chunk_number}/{len(chunks)}", flush=True)
        fragment = _request_deepseek_text(
            client,
            system_prompt=_build_deepseek_index_system_prompt(),
            user_prompt=(
                "## Task\n\n"
                f"Generate the structured index entries for source chunk {chunk_number} of "
                f"{len(chunks)}.\n\n"
                f'<SOURCE_CHUNK index="{chunk_number}" total="{len(chunks)}">\n'
                f"{_render_source_blocks(chunk)}\n"
                "</SOURCE_CHUNK>"
            ),
            purpose=f"recommendation index chunk {chunk_number}/{len(chunks)}",
            max_tokens=32768,
        )
        index_entries.extend(
            SkillCompilerIndexChunk.model_validate(
                json.loads(fragment, strict=False)
            ).entries
        )

    return _build_compiler_result(
        SkillCompilerDraft(
            **metadata.model_dump(),
            recommendations_index_entries=index_entries,
        ),
        source_blocks,
    )


def compile_guideline_text(full_text: str) -> SkillCompilerResult:
    if SKILL_COMPILER_PROVIDER == "deepseek":
        return _compile_guideline_text_with_deepseek(full_text)

    source_blocks = _build_source_blocks(full_text)
    draft = Runner.run_sync(
        build_skill_compiler_agent(),
        _build_compile_prompt(full_text),
    ).final_output
    return _build_compiler_result(draft, source_blocks)
