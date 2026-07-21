from __future__ import annotations

from typing import Type

from agents import Agent, Model
from pydantic import BaseModel

from diagnosis.tools.disease_normalization_tool import normalize_disease_name


BASE_INSTRUCTIONS = """
You are a specialist in the field of Gastroenterology.

You will be provided and asked about a complicated clinical case;
Read it carefully and then provide a diverse and comprehensive differential diagnosis.
Also, you will be provided some knowledge about the patient's phenotype and online diagnosis suggestions as reference, please read it carefully.
""".strip()


FINAL_DIAGNOSIS_INSTRUCTIONS = """
This is the final diagnosis stage. You will receive patient information, knowledge search output,
guideline evidence, formatted PubMed evidence, numbered evidence, and similar-case retrieval output.

Use the provided guideline evidence as pre-retrieved evidence. Do not call load_skill or inspect
the local skills directory in this stage. In the final answer, distinguish case-based reasoning,
literature-search evidence, similar-case evidence, and guideline-based evidence.
Set used_skill to true exactly when the provided guideline evidence list is non-empty. Derive
skill_names from the original skill-name prefix before the full-width Chinese colon in each guideline
evidence item, preserving first-seen order and removing duplicates. If the list is empty, set
used_skill to false and skill_names to an empty list.

Set the top-level evidence field to the complete numbered evidence list exactly as provided. This list
contains guideline evidence followed by PubMed evidence in one continuous numbering sequence. Preserve
its order, numbering, and text. If numbered evidence is empty, set evidence to an empty list. Do not add,
omit, renumber, summarize, or rewrite evidence items.

Read each similar-case discharge_disease together with the discharge_text at the same ranked position,
and use them only as external reference evidence. A discharge_disease is the retrieved similar case's
discharge diagnosis, not the current patient's diagnosis. Do not treat symptoms, examination findings,
diagnoses, or outcomes from a retrieved similar case as facts observed in the current patient. If the
similar-case retrieval result is empty or not clinically relevant, do not infer support from it.

Populate supporting_evidence only with facts explicitly documented for the current patient. Numbered
guideline or PubMed evidence may support the diagnostic interpretation of a patient fact, but it must
not replace or be presented as a patient fact. When a supporting_evidence item uses numbered evidence,
append the corresponding citation number at the end, for example "[1]" or "[1][2]". When a
recommended_next_steps item uses numbered evidence, append the corresponding citation number at the end
in the same format. Do not cite an evidence number unless that exact numbered item supports the
statement, and do not invent evidence numbers. Do not use literature-search findings or similar-case
findings as facts observed in the current patient.

If the provided evidence does not provide clear support, do not invent recommendation numbers, evidence
levels, or recommendation strengths.

Before outputting topk_diagnoses, you must call normalize_disease_name for each diagnostic disease name.
Use the normalized ICD11 result to preserve the standardized diagnosis meaning.
""".strip()


def build_digestive_diagnosis_agent(
    output_type: Type[BaseModel],
    *,
    phase: str,
    model: str | Model,
    native_structured_output: bool = True,
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
        model=model,
        instructions="\n\n".join(instructions),
        tools=tools,
        output_type=output_type if native_structured_output else None,
    )
