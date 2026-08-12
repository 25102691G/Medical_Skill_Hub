from __future__ import annotations

from pathlib import Path
from typing import Sequence, Type

from agents import Agent, Model, Tool
from agents.sandbox import Manifest, SandboxAgent, SandboxPathGrant
from agents.sandbox.capabilities import (
    Capabilities,
    LocalDirLazySkillSource,
    Shell,
    Skills,
)
from agents.sandbox.entries import LocalDir
from pydantic import BaseModel

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"


GUIDELINE_ORCHESTRATOR_INSTRUCTIONS = """
## GUIDELINE ORCHESTRATOR INSTRUCTIONS

You select local gastroenterology guideline skills from the supplied diagnostic hypotheses and skill
catalog. The catalog contains each skill's exact name and its SKILL.md front-matter description.

First select every skill whose primary disease directly corresponds to a diagnostic hypothesis. The
specific disease must match, and every narrower condition required by the skill, such as subtype,
stage, hereditary status, metastatic site, complication, procedure, or pregnancy, must be explicit in
the hypothesis. A broad shared disease category is insufficient.

Then perform exactly one forward differential expansion step. For every directly matched skill, read
the explicit differential diseases in its description and select every available skill whose primary
disease directly corresponds to one of those differential diseases. Do not select a skill merely
because a hypothesis appears in that skill's differential-disease list. Do not expand from an expanded
skill and do not recurse. Deduplicate the selected skill names.

Always call run_selected_guideline_skills exactly once. Put each direct match in direct_matches with
the exact matching hypothesis. Put each forward expansion in expanded_matches with its directly
matched source skill and the explicit differential disease that links them. Use exact skill names from
the catalog. When no skill is selected, pass empty direct_matches and expanded_matches plus a specific
unused_reason. The tool result is the final guideline search result; never construct that result
yourself.
""".strip()


GUIDELINE_SKILL_EXECUTOR_INSTRUCTIONS = """
## GUIDELINE SKILL EXECUTOR INSTRUCTIONS

You execute exactly one explicitly selected local disease guideline skill for a gastroenterology
diagnosis task.

1. Call the native load_skill tool for the exact selected skill name and do not load any other skill.
2. Completely read that skill's SKILL.md before retrieving evidence.
3. Follow the workflow and resource instructions defined by that SKILL.md exactly. The SKILL.md, not
   an external generic retrieval procedure, determines which references or scripts to use and how to
   verify evidence.
4. Use only the supplied positive_features as findings observed in the current patient. The selected
   skill name and its disease scope are external knowledge, not patient facts. Do not invent negative
   findings from missing information.
5. If the skill materials do not provide clear relevant evidence, return an empty guideline_evidence
   list and explain the insufficiency in guideline_diagnosis. Do not invent recommendation numbers,
   evidence levels, recommendation strengths, or guideline statements.
6. Return exactly one GuidelineSkillResult for the selected skill. Use the exact original skill name.
   Include medical evidence only; do not include source block IDs, full-text line ranges, or locator
   text in guideline_evidence.
""".strip()


def guideline_skill_catalog() -> list[dict[str, str]]:
    catalog: list[dict[str, str]] = []
    for path in sorted(SKILLS_DIR.iterdir(), key=lambda item: item.name):
        skill_md = path / "SKILL.md"
        if not path.is_dir() or not skill_md.is_file():
            continue
        description = ""
        for line in skill_md.read_text(encoding="utf-8").splitlines():
            if line.startswith("description:"):
                description = line.split(":", 1)[1].strip().strip("\"'")
                break
        catalog.append({"name": path.name, "description": description})
    return catalog


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


def build_guideline_orchestrator_agent(
    output_type: Type[BaseModel],
    model: str | Model,
    tools: Sequence[Tool],
    *,
    native_structured_output: bool = True,
) -> Agent:
    return Agent(
        name="Guideline Orchestrator Agent",
        model=model,
        instructions=GUIDELINE_ORCHESTRATOR_INSTRUCTIONS,
        output_type=output_type if native_structured_output else None,
        tools=list(tools),
        tool_use_behavior="stop_on_first_tool",
    )


def build_guideline_skill_executor_agent(
    output_type: Type[BaseModel],
    model: str | Model,
    *,
    native_structured_output: bool = True,
) -> SandboxAgent:
    capabilities = (
        [*Capabilities.default(), _build_guideline_skill_capability()]
        if native_structured_output
        else [Shell(), _build_guideline_skill_capability()]
    )
    return SandboxAgent(
        name="Guideline Skill Executor Agent",
        model=model,
        instructions=GUIDELINE_SKILL_EXECUTOR_INSTRUCTIONS,
        output_type=output_type if native_structured_output else None,
        capabilities=capabilities,
        default_manifest=_build_guideline_skill_manifest(),
    )
