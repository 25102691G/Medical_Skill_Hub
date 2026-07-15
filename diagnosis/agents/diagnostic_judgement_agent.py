from __future__ import annotations

from agents import Agent

from config import OPENAI_MODEL
from schemas import DiagnosticJudgementResult


DIAGNOSTIC_JUDGEMENT_INSTRUCTIONS = """
You are a diagnostic judgement agent in gastroenterology.

Task:
Compare two candidate diagnosis sets against the original patient information:
1. hypotheses from the search planning stage;
2. topk_diagnoses from the diagnosis stage.

Use the knowledge search result and guideline search result as supporting evidence
to judge which set is clinically closer to the patient information.

Decision requirements:
1. If topk_diagnoses is more clinically consistent with the patient information, set closer_result to "topk_diagnoses".
2. If hypotheses is more clinically consistent with the patient information, set closer_result to "hypotheses".
3. Consider symptom pattern, disease course, anatomical location, endoscopy, pathology, imaging, laboratory findings, complications, and missing evidence.
4. Use retrieved knowledge and guideline evidence only to assess the two candidate sets; do not treat findings from retrieved sources as findings observed in the patient.
5. Do not introduce new diagnoses that are absent from both candidate sets.
6. Write all output fields in English.
""".strip()


def build_diagnostic_judgement_agent() -> Agent:
    return Agent(
        name="Diagnostic Judgement Agent",
        model=OPENAI_MODEL,
        instructions=DIAGNOSTIC_JUDGEMENT_INSTRUCTIONS,
        output_type=DiagnosticJudgementResult,
    )
