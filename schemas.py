from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _normalize_icd_code(icd_code: str) -> str:
    return icd_code.strip().upper().replace(".", "")


class PhenotypeItem(BaseModel):
    phenotype: str = Field(description="English phenotype description")


class PhenotypeExtractionResult(BaseModel):
    phenotypes: list[PhenotypeItem] = Field(description="Patient phenotype list extracted from the case text")


class PubMedAbstractSection(BaseModel):
    section_index: int = Field(
        ge=1,
        description="One-based section index in the original PubMed abstract",
    )
    text: str = Field(
        min_length=1,
        description="Original text of one PubMed abstract section",
    )


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
    abstract_sections: list[PubMedAbstractSection] = Field(
        min_length=1,
        description="Publication abstract preserved as separate sections",
    )
    url: str = Field(description="PubMed publication URL")


class PubMedQueryResult(BaseModel):
    query: str = Field(description="Original PubMed search query")
    results: list[PubMedSearchResult] = Field(
        description="Retrieved PubMed results selected as relevant to the search queries"
    )


class SelectedPubMedSection(BaseModel):
    pmid: Annotated[str, Field(pattern=r"^\d+$")] = Field(
        description="Numeric PMID copied from the provided PubMed search results",
    )
    section_index: int = Field(
        ge=1,
        description="Section index copied from the selected abstract section",
    )


class KnowledgeSearchSelectionResult(BaseModel):
    selected_sections: list[SelectedPubMedSection] = Field(
        description="PubMed abstract sections selected as relevant to the search queries"
    )


class KnowledgeSearchResult(BaseModel):
    relevant_pubmed_results: list[PubMedQueryResult] = Field(
        description=(
            "Original PubMed search results containing only selected abstract sections, grouped by query"
        )
    )
    reason: str | None = Field(
        default=None,
        description="Failure reason when knowledge search returns an empty result",
    )


class DiagnosisItem(BaseModel):
    rank: int = Field(ge=1, le=5, description="Diagnosis ranking, starting from 1")
    icd_code: str = Field(
        min_length=3,
        max_length=7,
        pattern=r"^[A-Z][0-9][0-9A-Z]{1,5}$",
        description=(
            "Complete three-to-seven-character ICD-10-CM code without a decimal point"
        ),
    )
    category_name: str = Field(
        description="Canonical English description corresponding to the ICD-10-CM code"
    )
    confidence: int = Field(ge=0, le=100, description="Integer confidence percentage from 0 to 100, for example 45 means 45%")
    supporting_evidence: list[str] = Field(
        default_factory=list,
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

    normalize_icd_code = field_validator("icd_code", mode="before")(
        _normalize_icd_code
    )


class ExcludedPlanningCandidate(BaseModel):
    icd_code: str = Field(
        min_length=3,
        max_length=7,
        pattern=r"^[A-Z][0-9][0-9A-Z]{1,5}$",
        description="Complete ICD-10-CM code copied from a search planning candidate",
    )
    category_name: str = Field(
        description="Category name copied from the corresponding search planning candidate"
    )
    patient_contrary_evidence: list[str] = Field(
        min_length=1,
        description=(
            "Explicit current-patient findings that justify excluding this planning candidate "
            "from the final top five"
        ),
    )

    normalize_icd_code = field_validator("icd_code", mode="before")(
        _normalize_icd_code
    )


class FinalDiagnosisContent(BaseModel):
    topk_diagnoses: list[DiagnosisItem] = Field(
        min_length=5,
        max_length=5,
        description="Exactly five ranked principal-diagnosis ICD candidates",
    )
    excluded_planning_candidates: list[ExcludedPlanningCandidate] = Field(
        description=(
            "Search planning candidates omitted from the final top five, each supported by explicit "
            "contrary evidence from the current patient"
        ),
    )
    summary: str = Field(description="Brief diagnostic analysis summary")

    @model_validator(mode="after")
    def validate_rankings(self) -> "FinalDiagnosisContent":
        diagnosis_codes = [item.icd_code for item in self.topk_diagnoses]
        if len(diagnosis_codes) != len(set(diagnosis_codes)):
            raise ValueError("Final diagnosis ICD codes must be unique.")
        if [item.rank for item in self.topk_diagnoses] != [1, 2, 3, 4, 5]:
            raise ValueError("Final diagnosis ranks must be exactly 1 through 5 in list order.")
        excluded_codes = [
            item.icd_code for item in self.excluded_planning_candidates
        ]
        if len(excluded_codes) != len(set(excluded_codes)):
            raise ValueError("Excluded planning candidate ICD codes must be unique.")
        return self


class DiagnosisResult(BaseModel):
    used_skill: bool = Field(description="Whether a guideline skill was used before the final diagnosis stage")
    skill_names: list[str] = Field(description="List of skill names actually used")
    topk_diagnoses: list[DiagnosisItem] = Field(
        description="Five ranked principal-diagnosis ICD candidates"
    )
    excluded_planning_candidates: list[ExcludedPlanningCandidate] = Field(
        default_factory=list,
        description="Search planning candidates excluded from the final top five",
    )
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
    unused_reason: str | None = Field(
        description="Reason no guideline skill was used; null when used_skill is true",
    )
    skill_results: list[GuidelineSkillResult] = Field(
        description="Guideline evidence and diagnostic conclusion grouped by used skill"
    )
    reason: str | None = Field(
        default=None,
        description="Failure reason when guideline search returns an empty result",
    )


class HypothesisItem(BaseModel):
    icd_code: str = Field(
        min_length=3,
        max_length=7,
        pattern=r"^[A-Z][0-9][0-9A-Z]{1,5}$",
        description="Complete three-to-seven-character ICD-10-CM code without a decimal point",
    )
    category_name: str = Field(
        description="Canonical English description corresponding to the ICD-10-CM code"
    )

    normalize_icd_code = field_validator("icd_code", mode="before")(
        _normalize_icd_code
    )


class SearchPlanningResult(BaseModel):
    hypotheses: list[HypothesisItem] = Field(
        max_length=5,
        description="Up to 5 major candidate diagnoses using the most specific available ICD-10-CM code",
    )
    search_queries: list[str] = Field(max_length=5, description="Up to 5 medical literature search queries")
    positive_features: list[str] = Field(
        description=(
            "Explicitly documented positive clinical manifestations and examination results "
            "for similar-case retrieval and guideline evidence search"
        )
    )
    reason: str | None = Field(
        default=None,
        description="Failure reason when search planning returns an empty result",
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
    reason: str | None = Field(
        default=None,
        description="Failure reason when similar-case retrieval returns an empty result",
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
    closer_result: Literal["final_diagnoses", "search_planning_diagnoses"] = Field(
        description="Which candidate diagnosis set is closer to the patient information"
    )
    reason: str = Field(description="Reasoning for the diagnostic judgement")
