from __future__ import annotations

from pathlib import Path
from typing import Type

from agents import Agent, Model
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

Return only direct matches. Return each directly matched skill at most once in direct_matches. Do not
perform differential expansion and do not select a skill merely because a hypothesis appears in its
differential-disease list. Use exact skill names from the catalog. When at least one skill is selected,
set unused_reason to null. When no skill is selected, return an empty direct_matches list plus a
specific unused_reason.
""".strip()


GUIDELINE_EXPANSION_INSTRUCTIONS = """
## GUIDELINE EXPANSION INSTRUCTIONS

You perform exactly one forward differential expansion for the supplied directly matched guideline
skills.

For every explicit differential disease supplied under each source skill, select every available target
skill whose primary disease directly corresponds to that differential disease. The target catalog
contains only each target skill's primary disease scope. The specific disease must match, and every
narrower condition required by the target skill, such as subtype, stage, hereditary status, metastatic
site, complication, procedure, or pregnancy, must be explicit in the differential disease. A broad
shared disease category or symptom similarity is insufficient.

Return results first grouped by source skill and then by differential disease. Copy every returned
source skill and differential disease exactly from the supplied input and use exact target skill names
from the supplied catalog. Return each target skill at most once across the complete response. Do not
select a source skill, do not perform reverse matching, and do not expand from a target skill. Omit
differential diseases and source skills that have no directly corresponding target skill.
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
4. Use only the supplied five-field patient_features as findings observed in the current patient. Keep
   present illness, preadmission history, physical examination, family history, and pertinent results
   distinct. The selected skill name and its disease scope are external knowledge, not patient facts.
   Do not invent negative findings from missing information.
5. If the skill materials do not provide clear relevant evidence, return an empty guideline_evidence
   list and explain the insufficiency in guideline_diagnosis. Do not invent recommendation numbers,
   evidence levels, recommendation strengths, or guideline statements.
6. Return exactly one GuidelineSkillResult for the selected skill. Use the exact original skill name.
   Include medical evidence only; do not include source block IDs, full-text line ranges, or locator
   text in guideline_evidence.
""".strip()


def guideline_skill_catalog() -> list[dict[str, str]]:
    skill_source = LocalDirLazySkillSource(source=LocalDir(src=SKILLS_DIR))
    return [
        {"name": skill.name, "description": skill.description}
        for skill in skill_source.list_skill_metadata(skills_path=".agents")
    ]


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
    *,
    native_structured_output: bool = True,
) -> Agent:
    return Agent(
        name="Guideline Orchestrator Agent",
        model=model,
        instructions=GUIDELINE_ORCHESTRATOR_INSTRUCTIONS,
        output_type=output_type if native_structured_output else None,
    )


def build_guideline_expansion_agent(
    output_type: Type[BaseModel],
    model: str | Model,
    *,
    native_structured_output: bool = True,
) -> Agent:
    return Agent(
        name="Guideline Differential Expansion Agent",
        model=model,
        instructions=GUIDELINE_EXPANSION_INSTRUCTIONS,
        output_type=output_type if native_structured_output else None,
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
