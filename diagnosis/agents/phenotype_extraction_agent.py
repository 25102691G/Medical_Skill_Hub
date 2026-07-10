from __future__ import annotations

from agents import Agent

from config import OPENAI_MODEL
from schemas import PhenotypeExtractionResult


PHENOTYPE_EXTRACTION_INSTRUCTIONS = """
You are a medical expert specialized in gastrointestinal disease and phenotype extraction.

Task:
1. Extract patient phenotypes from the provided patient information only.
2. Focus on clinically meaningful symptoms, signs, laboratory abnormalities, imaging findings, endoscopic findings, pathology findings, complications, and relevant disease manifestations.
3. Output phenotype descriptions in English.
4. Do not invent findings that are not present in the patient text.
5. Do not provide diagnosis, treatment advice, or any extra narrative.
""".strip()


def build_phenotype_extraction_agent() -> Agent:
    return Agent(
        name="胃肠疾病表型提取 Agent",
        model=OPENAI_MODEL,
        instructions=PHENOTYPE_EXTRACTION_INSTRUCTIONS,
        output_type=PhenotypeExtractionResult,
    )
