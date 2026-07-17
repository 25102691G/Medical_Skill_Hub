from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PhenotypeItem(BaseModel):
    # hpo: str = Field(description="HPO identifier, or an empty string if it cannot be determined")
    phenotype: str = Field(description="English phenotype description")


class PhenotypeExtractionResult(BaseModel):
    phenotypes: list[PhenotypeItem] = Field(description="Patient phenotype list extracted from the case text")


class DiagnosisItem(BaseModel):
    rank: int = Field(description="Diagnosis ranking, starting from 1")
    disease: str = Field(description="Suspected diagnosis disease name")
    confidence: int = Field(ge=0, le=100, description="Integer confidence percentage from 0 to 100, for example 45 means 45%")
    supporting_evidence: list[str] = Field(
        description=(
            "Evidence from the current patient case supporting this diagnosis. Each item must end with "
            "a source suffix in the form [case section-specific examination], for example "
            "[入院时辅助资料-血常规]."
        )
    )
    missing_information: list[str] = Field(description="Additional information needed for further confirmation")
    recommended_next_steps: list[str] = Field(description="Recommended next examinations or clinical management directions")
    guideline_evidence: list[str] = Field(default_factory=list, description="Guideline evidence if a guideline skill is used")


class DiagnosisResult(BaseModel):
    used_skill: bool = Field(description="Whether a guideline skill was used before the final diagnosis stage")
    skill_names: list[str] = Field(description="List of skill names actually used")
    topk_diagnoses: list[DiagnosisItem] = Field(description="Top-K suspected diagnoses")
    summary: str = Field(description="Brief diagnostic analysis summary")
    safety_note: str = Field(description="Medical safety note")


class GuidelineSearchResult(BaseModel):
    used_skill: bool = Field(description="Whether any guideline skill was used")
    skill_names: list[str] = Field(description="List of guideline skill names actually used")
    guideline_evidence: list[str] = Field(description="Relevant guideline evidence extracted from loaded skills")
    summary: str = Field(description="Brief summary of guideline search findings")
    limitations: list[str] = Field(description="Limitations of the guideline skill search")


class SimilarCaseQueries(BaseModel):
    clinical_manifestations: list[str] = Field(
        description="Present illness history and positive symptoms explicitly documented in the case"
    )
    examination_results: list[str] = Field(
        description="Explicitly documented examination results, including laboratory, endoscopic, imaging, pathology, and microbiology findings"
    )


class SearchPlanningResult(BaseModel):
    hypotheses: list[str] = Field(max_length=5, description="Up to 5 major candidate diagnoses")
    search_queries: list[str] = Field(max_length=5, description="Up to 5 medical literature search queries")
    similar_case_queries: SimilarCaseQueries = Field(
        description="Structured case features for future similar-case retrieval"
    )


class SimilarCaseRetrievalResult(BaseModel):
    discharge_disease: list[str] = Field(
        max_length=10,
        description="Discharge diseases from the top 10 similar cases in retrieval rank order",
    )
    hadm_id: list[str] = Field(
        max_length=10,
        description="Hospital admission IDs from the top 10 similar cases in retrieval rank order",
    )
    discharge_texts: list[str] = Field(
        max_length=10,
        description="Discharge texts from the top 10 similar cases in retrieval rank order",
    )


class DiagnosticJudgementResult(BaseModel):
    closer_result: Literal["topk_diagnoses", "hypotheses"] = Field(
        description="Which candidate diagnosis set is closer to the patient information"
    )
    reason: str = Field(description="Reasoning for the diagnostic judgement")
