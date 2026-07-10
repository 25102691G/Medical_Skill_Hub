from __future__ import annotations

from pathlib import Path
from typing import Type

from agents import Agent
from agents.sandbox import Manifest, SandboxAgent, SandboxPathGrant
from agents.sandbox.capabilities import Capabilities, LocalDirLazySkillSource, Skills
from agents.sandbox.entries import LocalDir
from pydantic import BaseModel

from config import OPENAI_MODEL
from diagnosis.tools.disease_normalization_tool import normalize_disease_name


SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"

BASE_INSTRUCTIONS = """
You are a specialist in the field of Gastroenterology.

You will be provided and asked about a complicated clinical case;
Read it carefully and then provide a diverse and comprehensive differential diagnosis.
Also, you will be provided some knowledge about the patient's phenotype and online diagnosis suggestions as reference, please read it carefully.
All outputs must be written in English.
""".strip()


TRIAGE_INSTRUCTIONS = """
Enumerate the top 5 most likely diagnoses.
Each diagnosis should be a gastrointestinal disease. 
Use ## to tag the disease name. 
Make sure to reorder the diagnoses from most likely to least likely. 
Output all diagnoses and explanations in English.
The top 5 diagnoses are:
""".strip()


DIAGNOSIS_WITH_SKILLS_INSTRUCTIONS = """
This is the final diagnosis stage. Available skills directory provides the root directory for local disease guideline skills.

When the patient input, phenotype extraction result, or triage result contains symptoms, endoscopic findings, imaging findings, pathology findings, laboratory evidence, or an explicit suspected diagnosis pointing to a disease:
1. First inspect which skills are available under Available skills directory.
2. If a corresponding disease guideline skill exists, you must call load_skill to load that skill.
3. After loading it, read .agents/{skill_name}/SKILL.md and follow its workflow to read references or run scripts.
4. In the final answer, distinguish case-based reasoning from guideline-based evidence.

If the skill materials do not provide clear evidence, do not invent recommendation numbers, evidence levels, or recommendation strengths.

Before outputting topk_diagnoses, you must call normalize_disease_name for each diagnostic disease name,
and set the disease field to the normalized ICD11 diagnosis name.

All final diagnosis fields, evidence, missing information, next steps, summaries, and safety notes must be written in English.
""".strip()


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


def build_digestive_diagnosis_agent(
    output_type: Type[BaseModel],
    *,
    phase: str,
) -> Agent:
    instructions = [BASE_INSTRUCTIONS]
    tools = []

    if phase == "triage":
        instructions.append(TRIAGE_INSTRUCTIONS)
    elif phase == "final_diagnosis":
        instructions.append(
            DIAGNOSIS_WITH_SKILLS_INSTRUCTIONS
        )
        # TODO: This tool is added and the instructions require it, but the model may not use it. A regular function call is used as a temporary fallback.
        tools.append(normalize_disease_name)
        return SandboxAgent(
            name="Gastroenterology Diagnosis Agent",
            model=OPENAI_MODEL,
            instructions="\n\n".join(instructions),
            tools=tools,
            output_type=output_type,
            capabilities=[
                *Capabilities.default(),
                _build_guideline_skill_capability(),
            ],
            default_manifest=_build_guideline_skill_manifest(),
        )

    return Agent(
        name="Gastroenterology Diagnosis Agent",
        model=OPENAI_MODEL,
        instructions="\n\n".join(instructions),
        tools=tools,
        output_type=output_type,
    )
