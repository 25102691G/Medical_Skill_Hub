from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PhenotypeItem(BaseModel):
    phenotype: str = Field(description="English phenotype description")


class PhenotypeExtractionResult(BaseModel):
    phenotypes: list[PhenotypeItem] = Field(description="Patient phenotype list extracted from the case text")


class PubMedEvidenceItem(BaseModel):
    pmid: str = Field(
        min_length=1,
        pattern=r"^\d+$",
        description="Exact numeric PubMed PMID copied from a retrieved result",
    )
    title: str = Field(
        min_length=1,
        description="Exact publication title copied from the retrieved PubMed result",
    )
    evidence: str = Field(
        min_length=1,
        description=(
            "Clinically relevant evidence faithfully extracted or summarized from the publication "
            "abstract, without adding conclusions that are absent from the abstract"
        ),
    )


class KnowledgeSearchResult(BaseModel):
    summary: str = Field(
        description=(
            "Brief synthesis of what the retrieved PubMed literature supports and any important "
            "limitations or insufficiency"
        )
    )
    pubmed_evidence: list[PubMedEvidenceItem] = Field(
        description=(
            "Relevant PubMed evidence items. Include an item only when its retrieved abstract contains "
            "clinically relevant evidence and both its PMID and title are available."
        )
    )


class DiagnosisItem(BaseModel):
    rank: int = Field(description="Diagnosis ranking, starting from 1")
    icd_code: str = Field(
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z][0-9][0-9A-Z]$",
        description="Three-character ICD-10-CM category code without a decimal point",
    )
    category_name: str = Field(
        description="Canonical English name corresponding to the ICD-10-CM category code"
    )
    confidence: int = Field(ge=0, le=100, description="Integer confidence percentage from 0 to 100, for example 45 means 45%")
    supporting_evidence: list[str] = Field(
        description=(
            "Evidence from the current patient case supporting this diagnosis. If numbered evidence "
            "supports the diagnostic interpretation, append the corresponding citation numbers, "
            "for example [1] or [1][2]."
        )
    )
    recommended_next_steps: list[str] = Field(
        description=(
            "Recommended next examinations or clinical management directions. If a step uses numbered "
            "evidence, append the corresponding citation numbers, for example [1] or [1][2]."
        )
    )


class FinalDiagnosisContent(BaseModel):
    topk_diagnoses: list[DiagnosisItem] = Field(description="Top-K suspected diagnoses")
    summary: str = Field(description="Brief diagnostic analysis summary")


class DiagnosisResult(BaseModel):
    used_skill: bool = Field(description="Whether a guideline skill was used before the final diagnosis stage")
    skill_names: list[str] = Field(description="List of skill names actually used")
    topk_diagnoses: list[DiagnosisItem] = Field(description="Top-K suspected diagnoses")
    summary: str = Field(description="Brief diagnostic analysis summary")
    evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Complete numbered evidence list derived from guideline evidence followed by PubMed "
            "evidence. Each item must use the format [number] source：evidence text."
        )
    )


class GuidelineSearchResult(BaseModel):
    used_skill: bool = Field(description="Whether any guideline skill was used")
    skill_names: list[str] = Field(description="List of guideline skill names actually used")
    guideline_evidence: list[str] = Field(
        description=(
            "Relevant guideline evidence extracted from loaded skills. Each item must use the format "
            "skill name：guideline evidence, preserving the original local skill name."
        )
    )


class HypothesisItem(BaseModel):
    icd_code: str = Field(
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z][0-9][0-9A-Z]$",
        description="Three-character ICD-10-CM category code without a decimal point",
    )
    category_name: str = Field(
        description="Canonical English name corresponding to the ICD-10-CM category code"
    )


class SearchPlanningResult(BaseModel):
    hypotheses: list[HypothesisItem] = Field(
        max_length=5,
        description="Up to 5 major candidate diagnoses at the ICD-10-CM category level",
    )
    search_queries: list[str] = Field(max_length=5, description="Up to 5 medical literature search queries")
    similar_case_queries: list[str] = Field(
        description=(
            "Explicitly documented positive clinical manifestations and examination results "
            "for similar-case retrieval"
        )
    )


class SimilarCaseSection(BaseModel):
    section: str = Field(description="Matched discharge summary section name")
    content: str = Field(description="Matched discharge summary section content")


class SimilarCaseRetrievalResult(BaseModel):
    discharge_disease: list[str] = Field(
        max_length=10,
        description="Discharge diseases from the top 10 similar cases in retrieval rank order",
    )
    icd_code: list[str] = Field(
        max_length=10,
        description="ICD codes corresponding to the retrieved similar cases in retrieval rank order",
    )
    Sections: list[list[SimilarCaseSection]] = Field(
        max_length=10,
        description=(
            "Matched discharge summary sections for each similar case in retrieval rank order"
        ),
    )


class DiagnosisRoundResult(BaseModel):
    round: int
    search_planning_result: SearchPlanningResult
    similar_case_retrieval_result: SimilarCaseRetrievalResult
    diagnosis_result: DiagnosisResult


class MultiRoundDiagnosisResult(BaseModel):
    is_multi_round: bool
    rounds: list[DiagnosisRoundResult]


class DiagnosisPipelineResult(BaseModel):
    multi_round_diagnosis: MultiRoundDiagnosisResult


class DiagnosticJudgementResult(BaseModel):
    closer_result: Literal["topk_diagnoses", "hypotheses"] = Field(
        description="Which candidate diagnosis set is closer to the patient information"
    )
    reason: str = Field(description="Reasoning for the diagnostic judgement")
