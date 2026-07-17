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
""".strip()


FINAL_DIAGNOSIS_INSTRUCTIONS = """
This is the final diagnosis stage. You will receive patient information, knowledge search output,
guideline search output, and similar-case retrieval output.

Use the guideline search output as pre-retrieved guideline evidence. Do not call load_skill or inspect
the local skills directory in this stage. In the final answer, distinguish case-based reasoning,
literature-search evidence, similar-case evidence, and guideline-based evidence.
Populate guideline_evidence only from the guideline search output. If no guideline evidence supports
a diagnosis, leave that diagnosis guideline_evidence as an empty list.

Read each similar-case discharge_disease together with the discharge_text at the same ranked position,
and use them only as external reference evidence. A discharge_disease is the retrieved similar case's
discharge diagnosis, not the current patient's diagnosis. Do not treat symptoms, examination findings,
diagnoses, or outcomes from a retrieved similar case as facts observed in the current patient. If the
similar-case retrieval result is empty or not clinically relevant, do not infer support from it.

Populate supporting_evidence only with facts explicitly documented for the current patient. End every
supporting_evidence item with a source suffix in square brackets. Use the exact case section heading and,
when available, the specific examination name, for example "[入院时辅助资料-血常规]" or
"[住院经过-结肠镜]". If one item is supported by multiple case locations, append multiple suffixes. If
the case text has no explicit section heading for a fact, use "[病例原文]" instead of inventing a
section name. Do not use literature-search findings, similar-case findings, or guideline statements as
the source of supporting_evidence.

If the guideline search output does not provide clear evidence, do not invent recommendation numbers,
evidence levels, or recommendation strengths.

Before outputting topk_diagnoses, you must call normalize_disease_name for each diagnostic disease name.
Use the normalized ICD11 result to preserve the standardized diagnosis meaning.
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
