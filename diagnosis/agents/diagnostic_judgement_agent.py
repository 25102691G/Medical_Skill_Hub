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

Judge which set is clinically closer to the patient information.

Decision requirements:
1. If topk_diagnoses is more clinically consistent with the patient information, set closer_result to "topk_diagnoses" and should_stop to true.
2. If hypotheses is more clinically consistent with the patient information, set closer_result to "hypotheses" and should_stop to false.
3. Consider symptom pattern, disease course, anatomical location, endoscopy, pathology, imaging, laboratory findings, complications, and missing evidence.
4. Do not introduce new diagnoses that are absent from both candidate sets.
5. Write all output fields in English.
""".strip()


def build_diagnostic_judgement_agent() -> Agent:
    return Agent(
        name="Diagnostic Judgement Agent",
        model=OPENAI_MODEL,
        instructions=DIAGNOSTIC_JUDGEMENT_INSTRUCTIONS,
        output_type=DiagnosticJudgementResult,
    )
