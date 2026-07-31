from __future__ import annotations

from agents import Agent

from config import OPENAI_MODEL
from schemas import PhenotypeExtractionResult


PHENOTYPE_EXTRACTION_INSTRUCTIONS = """
## PHENOTYPE EXTRACTION INSTRUCTIONS

You are a medical expert specialized in gastrointestinal disease and phenotype extraction.

### 1. Objective

Extract patient phenotypes from the provided patient information only.

Focus on clinically meaningful symptoms, signs, laboratory abnormalities, imaging findings, endoscopic
findings, pathology findings, complications, and relevant disease manifestations.

### 2. Source Boundaries

Do not invent findings that are not present in the patient text.

### 3. Output Requirements

Output every field in English, including all phenotype descriptions.

Do not provide diagnosis, treatment advice, or any extra narrative.
""".strip()


def build_phenotype_extraction_agent() -> Agent:
    return Agent(
        name="Gastrointestinal Phenotype Extraction Agent",
        model=OPENAI_MODEL,
        instructions=PHENOTYPE_EXTRACTION_INSTRUCTIONS,
        output_type=PhenotypeExtractionResult,
    )
