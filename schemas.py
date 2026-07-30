from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class PhenotypeItem(BaseModel):
    phenotype: str = Field(description="English phenotype description")


class PhenotypeExtractionResult(BaseModel):
    phenotypes: list[PhenotypeItem] = Field(description="Patient phenotype list extracted from the case text")


class PubMedSearchResult(BaseModel):
    pmid: str = Field(
        min_length=1,
        pattern=r"^\d+$",
        description="Numeric PubMed PMID from the retrieved result",
    )
    title: str = Field(
        min_length=1,
        description="Publication title from the retrieved PubMed result",
    )
    abstract: str = Field(
        min_length=1,
        description="Publication abstract from the retrieved PubMed result",
    )
    url: str = Field(description="PubMed publication URL")


class PubMedQueryResult(BaseModel):
    query: str = Field(description="Original PubMed search query")
    results: list[PubMedSearchResult] = Field(
        description="Retrieved PubMed results selected as relevant to the search queries"
    )


class KnowledgeSearchSelectionResult(BaseModel):
    selected_pmids: list[Annotated[str, Field(pattern=r"^\d+$")]] = Field(
        description="Numeric PMIDs selected from the provided PubMed search results"
    )


class KnowledgeSearchResult(BaseModel):
    relevant_pubmed_results: list[PubMedQueryResult] = Field(
        description=(
            "Original PubMed search results whose PMIDs were selected as relevant, grouped by query"
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


class GuidelineSkillResult(BaseModel):
    skill_name: str = Field(description="Original local guideline skill name")
    disease_name: str = Field(description="Disease evaluated by this guideline skill")
    guideline_evidence: list[str] = Field(
        description=(
            "Relevant evidence extracted from this skill and verified against its guideline full text"
        )
    )
    guideline_diagnosis: str = Field(
        description=(
            "Concise natural-language conclusion comparing the patient's positive features with the "
            "verified guideline information and stating whether the patient may have this disease"
        )
    )


class GuidelineSearchResult(BaseModel):
    used_skill: bool = Field(description="Whether any guideline skill was loaded and searched")
    skill_results: list[GuidelineSkillResult] = Field(
        description="Guideline evidence and diagnostic conclusion grouped by used skill"
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
    positive_features: list[str] = Field(
        description=(
            "Explicitly documented positive clinical manifestations and examination results "
            "for similar-case retrieval and guideline evidence search"
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
    guideline_search_result: GuidelineSearchResult
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
