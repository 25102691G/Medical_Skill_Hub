from __future__ import annotations

from pathlib import Path
from typing import Any, Type

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

Select a skill when its primary disease is the same disease as a diagnostic hypothesis or is a broader
disease that clearly covers that hypothesis. A broad skill may match a more specific hypothesis. For
example, cholangiocarcinoma matches intrahepatic cholangiocarcinoma, pancreatic cancer matches cancer
of the pancreatic head, and peptic ulcer matches gastric ulcer with hemorrhage. Do not require
identical wording or identical disease granularity.

If a skill is narrower than the hypothesis, select it only when all required qualifiers, such as
subtype, stage, complication, hereditary status, metastatic site, procedure, or pregnancy, are
explicit in the hypothesis. Do not match based only on the same organ, a symptom, a treatment, or a
disease listed only as a differential diagnosis.

Return only direct matches. Return each directly matched skill at most once in direct_matches. Do not
perform differential expansion. Use exact skill names from the catalog. When at least one skill is
selected, set unused_reason to null. When no skill is selected, return an empty direct_matches list
plus a concise unused_reason.
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


GUIDELINE_RESULT_FILTER_INSTRUCTIONS = """
## GUIDELINE RESULT FILTER INSTRUCTIONS

You filter completed expanded guideline-skill results before they are supplied to a gastroenterology
final-diagnosis model. This is a recall-oriented filter. Directly matched guideline results are retained
outside this task and are not supplied for selection.

Use patient_information as the only source of facts about the current patient. Guideline conclusions
and evidence are external information. Do not treat an unreported patient finding as absent, and do not
invent patient findings.

Retain an expanded guideline result when it has any plausible diagnostic value for the current
hospitalization, including when its verified evidence:

* interprets an explicitly documented patient finding;
* helps distinguish a supplied diagnostic hypothesis;
* supports a plausible principal diagnosis suggested by the patient information;
* provides relevant criteria for documented imaging, endoscopy, pathology, laboratory findings, or
  disease course;
* supports exclusion using an explicitly documented negative patient finding.

Exclude an expanded guideline result only when it is clearly irrelevant or unusable, including when its
evidence is empty, contains only generic background or treatment information, relates only through a
general risk factor or nonspecific symptom, or its diagnostic conclusion depends on treating missing
information as a negative finding. If the diagnostic value is uncertain, retain the result.

Return only the exact skill names from expanded_guideline_results that should be retained. Do not add,
rename, summarize, or modify any guideline result.
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


QWEN_GUIDELINE_SKILL_EXECUTOR_INSTRUCTIONS = """
## QWEN TOOL EXECUTION RULES

The selected skill name is an exact identifier. Copy it verbatim without adding spaces or changing
punctuation.

Every exec_command call starts in a fresh working directory. The selected skill is always loaded at
the fixed path `.agents/selected_skill`. After load_skill, begin every command with
`cd '.agents/selected_skill' &&`. Read SKILL.md once, then follow its retrieval workflow. Run catalog
at most once, entries at most once, and sources at most once. For multiple IDs, repeat the option for
every value, for example `--heading-id H0001 --heading-id H0002` and
`--source-id L000001-L000003 --source-id L000010-L000012`. Do not place multiple values after one
option. After sources returns the verified text, immediately produce the final JSON result. Do not use
ls, find, head, grep, or direct cat commands to explore reference files.
""".strip()


class SelectedGuidelineSkillSource(LocalDirLazySkillSource):
    selected_skill_name: str

    async def load_skill(
        self,
        *,
        skill_name: str,
        session: Any,
        skills_path: str,
        user: Any = None,
    ) -> dict[str, str]:
        metadata = next(
            skill
            for skill in self.list_skill_metadata(
                skills_path=skills_path,
                source_grants=session.state.manifest.extra_path_grants,
            )
            if (
                skill.name == self.selected_skill_name
                or skill.path.name == self.selected_skill_name
            )
        )
        skill_source = self.source.model_copy(
            update={"src": SKILLS_DIR / metadata.path.name},
            deep=True,
        )
        fixed_path = Path(skills_path) / "selected_skill"
        await skill_source.apply(
            session,
            Path(session.state.manifest.root) / fixed_path,
            base_dir=Path.cwd(),
            user=user,
        )
        return {
            "status": "loaded",
            "skill_name": metadata.name,
            "path": fixed_path.as_posix(),
        }


def guideline_skill_catalog() -> list[dict[str, str]]:
    skill_source = LocalDirLazySkillSource(source=LocalDir(src=SKILLS_DIR))
    return [
        {"name": skill.name, "description": skill.description}
        for skill in skill_source.list_skill_metadata(skills_path=".agents")
    ]


def _build_guideline_skill_capability(
    selected_skill_name: str | None = None,
) -> Skills:
    skill_source = (
        SelectedGuidelineSkillSource(
            source=LocalDir(src=SKILLS_DIR),
            selected_skill_name=selected_skill_name,
        )
        if selected_skill_name is not None
        else LocalDirLazySkillSource(source=LocalDir(src=SKILLS_DIR))
    )
    return Skills(
        lazy_from=skill_source,
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


def build_guideline_result_filter_agent(
    output_type: Type[BaseModel],
    model: str | Model,
    *,
    native_structured_output: bool = True,
) -> Agent:
    return Agent(
        name="Guideline Result Filter Agent",
        model=model,
        instructions=GUIDELINE_RESULT_FILTER_INSTRUCTIONS,
        output_type=output_type if native_structured_output else None,
    )


def build_guideline_skill_executor_agent(
    output_type: Type[BaseModel],
    model: str | Model,
    *,
    native_structured_output: bool = True,
    qwen_mode: bool = False,
    selected_skill_name: str | None = None,
) -> SandboxAgent:
    bound_skill_name = selected_skill_name if qwen_mode else None
    capabilities = (
        [*Capabilities.default(), _build_guideline_skill_capability(bound_skill_name)]
        if native_structured_output
        else [Shell(), _build_guideline_skill_capability(bound_skill_name)]
    )
    return SandboxAgent(
        name="Guideline Skill Executor Agent",
        model=model,
        instructions=(
            f"{GUIDELINE_SKILL_EXECUTOR_INSTRUCTIONS}\n\n"
            f"{QWEN_GUIDELINE_SKILL_EXECUTOR_INSTRUCTIONS}"
            if qwen_mode
            else GUIDELINE_SKILL_EXECUTOR_INSTRUCTIONS
        ),
        output_type=output_type if native_structured_output else None,
        capabilities=capabilities,
        default_manifest=_build_guideline_skill_manifest(),
    )
