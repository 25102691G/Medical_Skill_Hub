from __future__ import annotations

from typing import Type

from agents import Agent, Model
from pydantic import BaseModel

# from diagnosis.tools.disease_normalization_tool import normalize_disease_name


BASE_INSTRUCTIONS = """
You are a specialist in the field of Gastroenterology.

You will be provided and asked about a complicated clinical case;
Read it carefully and then provide a diverse and comprehensive differential diagnosis.
Also, you will be provided some knowledge about the patient's phenotype and online diagnosis suggestions as reference, please read it carefully.
""".strip()


FINAL_DIAGNOSIS_INSTRUCTIONS = """
This is the final diagnosis stage. You will receive patient information, current hypotheses from the
search planning stage, one numbered evidence list, and a compact similar-case summary. When revising a
previous diagnosis, you will also receive the previous hypotheses, previous top-K diagnoses, and the
diagnostic judgement explaining why the hypotheses were closer to the patient.

Use the current hypotheses as the candidate foundation. For each hypothesis, decide whether to retain
it, promote it, demote it, or refine its disease name and clinical specificity based on the patient
information and retrieved evidence. Do not discard a clinically supported hypothesis merely to create
an unrelated differential diagnosis. Remove a hypothesis only when it is contradicted by the patient
information or is less likely than another diagnosis needed in the limited top-K result.

When previous-round artifacts and a diagnostic judgement are provided, correct the omissions, ranking
problems, or unsupported refinements identified by that judgement. Reconcile the previous hypotheses
and previous top-K diagnoses with the newly retrieved evidence instead of generating another independent
candidate set.

Use the numbered evidence as pre-retrieved guideline and literature evidence. Do not call load_skill or
inspect the local skills directory in this stage. Distinguish case-based reasoning, literature-search
evidence, similar-case evidence, and guideline-based evidence. Write all generated content in English,
including disease names, supporting evidence, recommended next steps, and the summary.

Read each similar-case discharge disease together with its matched section and use them only as external
reference evidence. A discharge disease is the retrieved similar case's diagnosis, not the current
patient's diagnosis. Do not treat findings, diagnoses, or outcomes from a retrieved case as facts
observed in the current patient. If a retrieved case is not clinically relevant, do not infer support
from it.

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
""".strip()

# Before outputting topk_diagnoses, call normalize_disease_name for each candidate disease name and set
# the disease field to the normalized ICD10 diagnosis name returned by the tool.


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

    instructions.append(FINAL_DIAGNOSIS_INSTRUCTIONS)
    return Agent(
        name="Gastroenterology Diagnosis Agent",
        model=model,
        instructions="\n\n".join(instructions),
        # tools=[normalize_disease_name],
        tools=[],
        output_type=output_type if native_structured_output else None,
    )
