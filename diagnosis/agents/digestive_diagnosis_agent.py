from __future__ import annotations

from typing import Type

from agents import Agent
from pydantic import BaseModel

from config import OPENAI_MODEL
from diagnosis.tools.disease_normalization_tool import normalize_disease_name


BASE_INSTRUCTIONS = """
You are a specialist in the field of Gastroenterology.

You will be provided and asked about a complicated clinical case;
Read it carefully and then provide a diverse and comprehensive differential diagnosis.
Also, you will be provided some knowledge about the patient's phenotype and online diagnosis suggestions as reference, please read it carefully.
All outputs must be written in English.
""".strip()


FINAL_DIAGNOSIS_INSTRUCTIONS = """
This is the final diagnosis stage. You will receive patient information, knowledge search output,
guideline search output, and similar-case retrieval output.

Use the guideline search output as pre-retrieved guideline evidence. Do not call load_skill or inspect
the local skills directory in this stage. In the final answer, distinguish case-based reasoning,
literature-search evidence, and guideline-based evidence.
Populate guideline_evidence only from the guideline search output. If no guideline evidence supports
a diagnosis, leave that diagnosis guideline_evidence as an empty list.

If the guideline search output does not provide clear evidence, do not invent recommendation numbers,
evidence levels, or recommendation strengths.

Before outputting topk_diagnoses, you must call normalize_disease_name for each diagnostic disease name,
and set the disease field to the normalized ICD11 diagnosis name.

All final diagnosis fields, evidence, missing information, next steps, summaries, and safety notes must be written in English.
""".strip()


def build_digestive_diagnosis_agent(
    output_type: Type[BaseModel],
    *,
    phase: str,
) -> Agent:
    if phase != "final_diagnosis":
        raise ValueError(f"Unsupported digestive diagnosis phase: {phase}")

    instructions = [BASE_INSTRUCTIONS]
    tools = []

    instructions.append(
        FINAL_DIAGNOSIS_INSTRUCTIONS
    )
    tools.append(normalize_disease_name)
    return Agent(
        name="Gastroenterology Diagnosis Agent",
        model=OPENAI_MODEL,
        instructions="\n\n".join(instructions),
        tools=tools,
        output_type=output_type,
    )
