from __future__ import annotations

from agents import Agent, Model

from schemas import DiagnosticJudgementResult


DIAGNOSTIC_JUDGEMENT_INSTRUCTIONS = """
## DIAGNOSTIC JUDGEMENT INSTRUCTIONS

You are a diagnostic judgement agent in gastroenterology.

### 1. Objective

Compare two candidate diagnosis sets against the original patient information:

* search_planning_diagnoses from the search planning stage, where each diagnosis is represented by an
  ICD-10-CM category code and its canonical English category name;
* final_diagnoses from the diagnosis stage, where each diagnosis is represented by an ICD-10-CM
  category code and its canonical English category name.

### 2. Candidate Evaluation

Consider symptom pattern, disease course, anatomical location, endoscopy, pathology, imaging,
laboratory findings, complications, and missing evidence.

If final_diagnoses is more clinically consistent with the patient information, set closer_result to
"final_diagnoses".

If search_planning_diagnoses is more clinically consistent with the patient information, set
closer_result to "search_planning_diagnoses".

Do not introduce new diagnoses that are absent from both candidate sets.

### 3. Diagnostic Granularity

Compare diagnoses at the three-character ICD-10-CM category level.

Anatomical site or subtype, complication status or type, severity, disease behavior, and other
subcategory details may inform clinical consistency but must not change the underlying category match.

### 4. Output Requirements

Keep closer_result as either "final_diagnoses" or "search_planning_diagnoses".
""".strip()


def build_diagnostic_judgement_agent(
    model: str | Model,
    *,
    native_structured_output: bool = True,
) -> Agent:
    return Agent(
        name="Diagnostic Judgement Agent",
        model=model,
        instructions=DIAGNOSTIC_JUDGEMENT_INSTRUCTIONS,
        output_type=DiagnosticJudgementResult if native_structured_output else None,
    )
