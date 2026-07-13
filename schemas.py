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
    supporting_evidence: list[str] = Field(description="Evidence in the case supporting this diagnosis")
    missing_information: list[str] = Field(description="Additional information needed for further confirmation")
    recommended_next_steps: list[str] = Field(description="Recommended next examinations or clinical management directions")
    guideline_evidence: list[str] = Field(default_factory=list, description="Guideline evidence if a guideline skill is used")


class DiagnosisResult(BaseModel):
    used_skill: bool = Field(description="Whether a skill was used in the final diagnosis stage")
    skill_names: list[str] = Field(description="List of skill names actually used")
    topk_diagnoses: list[DiagnosisItem] = Field(description="Top-K suspected diagnoses")
    summary: str = Field(description="Brief diagnostic analysis summary")
    safety_note: str = Field(description="Medical safety note")


class SearchQueryItem(BaseModel):
    intent: Literal[
        "acute_problem",
        "most_likely_disease",
        "diagnostic_criteria_or_case_features",
        "key_differential_diagnosis",
    ] = Field(description="Search intent: current acute problem, most likely disease, diagnostic criteria or case features, or key differential diagnosis")
    query: str = Field(description="Medical literature search query")


class SearchPlanningResult(BaseModel):
    problem_representation: str = Field(description="Brief representation of the most important current clinical problem")
    hypotheses: list[str] = Field(min_length=3, max_length=5, description="3-5 major candidate diagnoses")
    search_queries: list[SearchQueryItem] = Field(max_length=5, description="Up to 5 medical literature search queries")


class SimilarCaseDiagnosisItem(BaseModel):
    source_query: SearchQueryItem = Field(description="Original query used for similar-case retrieval")
    diagnosis: str = Field(description="Similar-case diagnosis corresponding to this query")
    matched_case_summary: str = Field(description="Brief summary of the similar case; if no real case database is connected, state that this is inferred only from the query")
    supporting_reason: str = Field(description="Why this query corresponds to this diagnosis")
    confidence: int = Field(ge=0, le=100, description="Integer diagnosis matching confidence percentage from 0 to 100")


class SimilarCaseRetrievalResult(BaseModel):
    items: list[SimilarCaseDiagnosisItem] = Field(description="Similar-case diagnosis result for each search query")
    summary: str = Field(description="Overall summary of the similar-case diagnosis results")
    limitations: list[str] = Field(description="Limitations of the current similar-case retrieval results")


class DiagnosticJudgementResult(BaseModel):
    closer_result: Literal["topk_diagnoses", "hypotheses"] = Field(
        description="Which candidate diagnosis set is closer to the patient information"
    )
    should_stop: bool = Field(
        description="Whether the current topk_diagnoses should be accepted as the final diagnosis result"
    )
    reason: str = Field(description="Reasoning for the diagnostic judgement")
