from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Type

from agents import Agent, Model, RunContextWrapper, function_tool
from agents.sandbox import Manifest, SandboxAgent, SandboxPathGrant
from agents.sandbox.capabilities import Capabilities, LocalDirLazySkillSource, Skills
from agents.sandbox.entries import LocalDir
from pydantic import BaseModel

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"
MAX_GUIDELINE_SEARCH_RESULT_LINES = 120


@dataclass
class GuidelineToolState:
    lock: Any = field(default_factory=Lock, repr=False)
    listed_skills: bool = False
    skill_md_reads: set[str] = field(default_factory=set)
    index_searches: set[str] = field(default_factory=set)
    full_text_searches: set[str] = field(default_factory=set)
    full_text_read_counts: dict[str, int] = field(default_factory=dict)
    accessed_skills: set[str] = field(default_factory=set)

GUIDELINE_SEARCHER_INSTRUCTIONS = """
## GUIDELINE SEARCHER INSTRUCTIONS

You are a Guideline Searcher Agent for gastroenterology diagnosis.

### 1. Objective

You will receive diagnostic hypotheses, positive patient features, and local disease guideline skills.

### 2. Skill Selection

Select skills only by direct correspondence between a skill description and at least one diagnostic
hypothesis. The specific disease name, abbreviation, or ICD-10-CM code must clearly match, and the
verified disease category must be compatible. A shared broad disease category alone is not sufficient.
Every disease modifier required by a skill, such as early, hereditary, metastatic site, complication,
procedure, or pregnancy, must also be explicit in the diagnostic hypothesis set. A general disease
hypothesis alone does not match a skill for a narrower subtype or clinical condition. Do not infer
these modifiers from positive_features.
Do not select a skill only because positive patient features, symptoms, examinations, or broad
gastroenterology terms overlap. There is no target or maximum number of selected skills: use every
directly relevant skill and no unrelated skill.

### 3. Guideline Retrieval

After selecting skills, do not use the hypotheses to search within a skill or to assess the patient.
For each selected skill, use only the supplied positive_features to locate guideline content relevant
to diagnostic criteria, differential diagnosis, confirmation or exclusion tests, and recommended next
steps. Use a recommendations index only to locate relevant content, then verify the supporting context
against the skill's guideline full text. The guideline full text is the authoritative source.

### 4. Source Boundaries

Treat positive_features as findings observed in the current patient, not as confirmed diagnoses. Do not
invent negative findings from missing information.

If the skill materials do not provide clear evidence, do not invent recommendation numbers,
evidence levels, recommendation strengths, or guideline statements.

### 5. Output Requirements

Return one skill_results item for every selected and searched skill. Keep all information from one skill
inside that item:

* skill_name: the exact original local skill identifier;
* disease_name: the disease evaluated by the skill;
* guideline_evidence: relevant evidence verified against that skill's guideline full text, without a
  skill-name prefix;
* guideline_diagnosis: a concise natural-language conclusion comparing the positive_features with the
  verified guideline information. State whether the patient may have the disease, which positive
  features support that conclusion, and which essential diagnostic information is still missing.

Do not use a fixed assessment scale or a Boolean diagnostic label. If the positive_features and
retrieved guideline information are insufficient, state that limitation in guideline_diagnosis and
return an empty guideline_evidence list for that skill.

Set used_skill to true when at least one skill was selected and searched, and set unused_reason to null.
If no skill directly matches the hypotheses, return used_skill as false, explain the specific reason in
unused_reason, and return an empty skill_results list.

Return only used_skill, unused_reason, skill_results, and reason in the structured output. Set reason
to null when the guideline search completes normally.
""".strip()

SANDBOX_SKILL_INSTRUCTIONS = """
## SANDBOX SKILL WORKFLOW

Use the sandbox Skills capability:

1. Inspect the available skills.
2. Select all and only skills whose descriptions directly correspond to at least one supplied
   diagnostic hypothesis. A shared broad category without a specific disease match is insufficient.
   Do not select skills from unrelated disease categories.
3. Call load_skill for every selected disease guideline skill.
4. Read .agents/{skill_name}/SKILL.md and follow its workflow to read references or run scripts.
5. After skill selection, use only positive_features for content retrieval and patient comparison.
   Do not use hypothesis disease names as within-skill search terms.
6. Use the recommendations index for location and verify relevant evidence against the guideline full
   text before returning it or using it in guideline_diagnosis.
""".strip()

FUNCTION_TOOL_SKILL_INSTRUCTIONS = """
## FUNCTION TOOL SKILL WORKFLOW

Use the local guideline function tools:

1. Call list_guideline_skills exactly once and select all and only skills whose descriptions directly
   correspond to at least one supplied diagnostic hypothesis. A shared broad category without a
   specific disease match is insufficient. Do not target a fixed skill count and do not select skills
   from unrelated disease categories. All narrower conditions required by a skill must be explicit in
   the diagnostic hypothesis set; a general disease hypothesis alone is insufficient.
2. For each selected skill, call read_guideline_file with file_name "SKILL.md" exactly once.
3. For each selected skill, call search_guideline on "recommendations-index.md" at most once, using
   only the most discriminative positive_features as alternative keywords in one call. Do not use
   hypothesis disease names as search keywords.
4. For each selected skill, read at most 2 relevant line ranges from "guideline-full-text.md" to verify
   the original context. If the index has no clear match, search "guideline-full-text.md" at most once.
5. Never repeat a tool call with the same arguments. Do not try to run scripts or use tools that are
   not provided.
6. Return one skill_results item for every selected and searched skill. If a search tool returns
   "No matching guideline content found.", treat that text only as an intermediate tool result and
   never copy it into the submitted guideline result. Keep the selected skill item, return an empty
   guideline_evidence list, and explain the insufficiency only inside guideline_diagnosis.
7. If no skill is selected, set used_skill to false, provide a specific explanation in unused_reason,
   and return an empty skill_results list in the final JSON object.
8. If at least one skill is selected, set used_skill to true and unused_reason to the JSON null value.
   After completing this bounded search, do not call another tool. Return the final result immediately
   as exactly one JSON object without Markdown fences or explanatory text.
9. A tool rejection means its call limit has already been reached. Do not attempt the call again with
   different arguments. Stop calling tools and return the final JSON using the evidence already read.
""".strip()

GUIDELINE_FILES = {
    "SKILL.md": "SKILL.md",
    "recommendations-index.md": "references/recommendations-index.md",
    "guideline-full-text.md": "references/guideline-full-text.md",
}


def _resolve_guideline_skill(skill_name: str) -> Path:
    skills = {
        path.name: path
        for path in SKILLS_DIR.iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    }
    if skill_name not in skills:
        raise ValueError(f"Unknown guideline skill: {skill_name}")
    return skills[skill_name]


@function_tool
def list_guideline_skills(
    ctx: RunContextWrapper[GuidelineToolState],
) -> list[dict[str, str]] | str:
    """List available local guideline skills with their descriptions."""
    with ctx.context.lock:
        if ctx.context.listed_skills:
            return (
                "Tool call rejected: guideline skills were already listed. "
                "Do not call more tools; return the final JSON now."
            )
        ctx.context.listed_skills = True
    skills: list[dict[str, str]] = []
    for path in sorted(SKILLS_DIR.iterdir(), key=lambda item: item.name):
        skill_md = path / "SKILL.md"
        if not path.is_dir() or not skill_md.is_file():
            continue
        description = ""
        for line in skill_md.read_text(encoding="utf-8").splitlines():
            if line.startswith("description:"):
                description = line.split(":", 1)[1].strip().strip("\"'")
                break
        skills.append({"name": path.name, "description": description})
    return skills


@function_tool
def search_guideline(
    ctx: RunContextWrapper[GuidelineToolState],
    skill_name: str,
    keywords: list[str],
    file_name: str = "recommendations-index.md",
    context_lines: int = 2,
) -> str:
    """Search any keyword in one allowed guideline reference file and return numbered context lines."""
    if file_name not in {"recommendations-index.md", "guideline-full-text.md"}:
        raise ValueError(f"Unsupported guideline search file: {file_name}")
    with ctx.context.lock:
        searched_skills = (
            ctx.context.index_searches
            if file_name == "recommendations-index.md"
            else ctx.context.full_text_searches
        )
        if skill_name in searched_skills:
            return (
                f"Tool call rejected: {file_name} was already searched for this skill. "
                "Do not call more tools; return the final JSON now."
            )
        searched_skills.add(skill_name)
        ctx.context.accessed_skills.add(skill_name)
    target = _resolve_guideline_skill(skill_name) / GUIDELINE_FILES[file_name]
    lines = target.read_text(encoding="utf-8").splitlines()
    normalized_keywords = [
        keyword.strip().casefold()
        for keyword in keywords
        if keyword.strip()
    ]
    hit_lines: set[int] = set()
    for index, line in enumerate(lines):
        normalized_line = line.casefold()
        if any(keyword in normalized_line for keyword in normalized_keywords):
            start = max(0, index - context_lines)
            end = min(len(lines), index + context_lines + 1)
            hit_lines.update(range(start, end))
    if not hit_lines:
        return "No matching guideline content found."
    output: list[str] = []
    previous = -2
    for index in sorted(hit_lines):
        if index != previous + 1:
            output.append("---")
        output.append(f"{index + 1}: {lines[index]}")
        previous = index
    if len(output) > MAX_GUIDELINE_SEARCH_RESULT_LINES:
        omitted = len(output) - MAX_GUIDELINE_SEARCH_RESULT_LINES
        output = output[:MAX_GUIDELINE_SEARCH_RESULT_LINES]
        output.append(f"--- {omitted} additional matching lines omitted ---")
    return "\n".join(output)


@function_tool
def read_guideline_file(
    ctx: RunContextWrapper[GuidelineToolState],
    skill_name: str,
    file_name: str,
    start_line: int = 1,
    end_line: int = 200,
) -> str:
    """Read a numbered line range from one allowed file in a local guideline skill."""
    if file_name not in GUIDELINE_FILES:
        raise ValueError(f"Unsupported guideline file: {file_name}")
    if file_name == "recommendations-index.md":
        return (
            "Tool call rejected: use search_guideline once for recommendations-index.md. "
            "Do not call more tools; return the final JSON now."
        )
    with ctx.context.lock:
        if file_name == "SKILL.md":
            if skill_name in ctx.context.skill_md_reads:
                return (
                    "Tool call rejected: SKILL.md was already read for this skill. "
                    "Do not call more tools; return the final JSON now."
                )
            ctx.context.skill_md_reads.add(skill_name)
        else:
            read_count = ctx.context.full_text_read_counts.get(skill_name, 0)
            if read_count >= 2:
                return (
                    "Tool call rejected: guideline-full-text.md has already been read twice for this "
                    "skill. Do not call more tools; return the final JSON now."
                )
            ctx.context.full_text_read_counts[skill_name] = read_count + 1
        ctx.context.accessed_skills.add(skill_name)
    target = _resolve_guideline_skill(skill_name) / GUIDELINE_FILES[file_name]
    lines = target.read_text(encoding="utf-8").splitlines()
    start = max(1, start_line)
    end = min(len(lines), end_line)
    return "\n".join(
        f"{index}: {lines[index - 1]}"
        for index in range(start, end + 1)
    )


def _build_guideline_skill_capability() -> Skills:
    return Skills(
        lazy_from=LocalDirLazySkillSource(
            source=LocalDir(src=SKILLS_DIR),
        ),
    )


def _build_guideline_skill_manifest() -> Manifest:
    return Manifest(
        extra_path_grants=(
            SandboxPathGrant(
                path=str(SKILLS_DIR),
                read_only=True,
                description="Disease guideline skill source directory",
            ),
        ),
    )


def build_guideline_searcher_agent(
    output_type: Type[BaseModel],
    model: str | Model,
    *,
    native_structured_output: bool = True,
) -> Agent | SandboxAgent:
    if native_structured_output:
        return SandboxAgent(
            name="Guideline Searcher Agent",
            model=model,
            instructions=f"{GUIDELINE_SEARCHER_INSTRUCTIONS}\n\n{SANDBOX_SKILL_INSTRUCTIONS}",
            output_type=output_type,
            capabilities=[
                *Capabilities.default(),
                _build_guideline_skill_capability(),
            ],
            default_manifest=_build_guideline_skill_manifest(),
        )
    return Agent(
        name="Guideline Searcher Agent",
        model=model,
        instructions=f"{GUIDELINE_SEARCHER_INSTRUCTIONS}\n\n{FUNCTION_TOOL_SKILL_INSTRUCTIONS}",
        tools=[
            list_guideline_skills,
            search_guideline,
            read_guideline_file,
        ],
    )
