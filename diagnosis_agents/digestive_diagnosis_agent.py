from __future__ import annotations

from pathlib import Path
from typing import Type

from agents import Agent
from agents.sandbox import Manifest, SandboxAgent, SandboxPathGrant
from agents.sandbox.capabilities import Capabilities, LocalDirLazySkillSource, Skills
from agents.sandbox.entries import LocalDir
from pydantic import BaseModel

from config import OPENAI_MODEL


SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"

DISEASE_SKILL_REGISTRY: dict[str, dict[str, object]] = {
    "克罗恩病": {
        "skill_name": "china-crohns-guideline-2023",
        "aliases": ["克罗恩病", "Crohn病", "Crohn disease", "Crohn's disease", "CD"],
    },
    "溃疡性结肠炎": {
        "skill_name": "china-ulcerative-colitis-guideline-2023",
        "aliases": ["溃疡性结肠炎", "ulcerative colitis", "UC"],
    },
    "肠白塞病": {
        "skill_name": "intestinal-behcet-consensus-2022",
        "aliases": ["肠白塞病", "肠型白塞病", "肠型贝赫切特综合征", "肠贝赫切特病", "intestinal Behcet"],
    },
    "肠结核": {
        "skill_name": "intestinal-tuberculosis-diagnosis-treatment",
        "aliases": ["肠结核", "intestinal tuberculosis", "ITB"],
    },
    "淋巴瘤": {
        "skill_name": "china-lymphoma-guideline-2022",
        "aliases": ["淋巴瘤", "lymphoma"],
    },
}

BASE_INSTRUCTIONS = """
You are a specialist in the field of Gastroenterology.

You will be provided and asked about a complicated clinical case;
Read it carefully and then provide a diverse and comprehensive differential diagnosis.
Also, you will be provided some knowledge about the patient's phenotype and online diagnosis suggestions as reference, please read it carefully.
""".strip()


TRIAGE_INSTRUCTIONS = """
Enumerate the top 5 most likely diagnoses.
Each diagnosis should be a gastrointestinal disease. 
Use ## to tag the disease name. 
Make sure to reorder the diagnoses from most likely to least likely. 
The top 5 diagnoses are:
""".strip()


DIAGNOSIS_WITH_SKILLS_INSTRUCTIONS = """
当前是最终诊断阶段。Available skills directory 提供本地疾病指南 skill 根目录。

当患者输入、表型抽取结果或 triage result 中出现疑似某个疾病的症状、内镜、影像、病理、实验室证据，或明确疑似诊断时：
1. 先查看 Available skills directory 下有哪些 skill。
2. 如存在对应疾病指南 skill，必须调用 load_skill 加载该 skill。
3. 加载后读取 .agents/{skill_name}/SKILL.md，并按其中工作流程读取 references 或运行 scripts。
4. 最终答案中要区分病例推理和指南依据。

如 skill 资料未检索到明确依据，不要编造推荐意见编号、证据等级或推荐强度。
""".strip()


def resolve_skill_names_for_diseases(disease_names: list[str]) -> list[str]:
    matched_skill_names: list[str] = []
    for disease_name in disease_names:
        normalized_disease_name = disease_name.lower()
        for item in DISEASE_SKILL_REGISTRY.values():
            skill_name = str(item["skill_name"])
            aliases = [str(alias).lower() for alias in item["aliases"]]
            if any(alias in normalized_disease_name for alias in aliases):
                if (SKILLS_DIR / skill_name).is_dir() and skill_name not in matched_skill_names:
                    matched_skill_names.append(skill_name)
    return matched_skill_names


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
                description="疾病指南 skill 源目录",
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
        return SandboxAgent(
            name="消化内科医疗诊断 Agent",
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
        name="消化内科医疗诊断 Agent",
        model=OPENAI_MODEL,
        instructions="\n\n".join(instructions),
        tools=tools,
        output_type=output_type,
    )
