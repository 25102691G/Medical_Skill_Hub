from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)


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
            "Reasons grounded in explicit current-patient findings for excluding this planning "
            "candidate or replacing it with a corrected ICD-10-CM code; supporting numbered "
            "guideline or literature evidence may be cited"
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
            "Search planning candidates not selected unchanged in the final top five, each with "
            "patient-grounded reasons for exclusion or ICD-10-CM correction"
        ),
    )
    summary: str = Field(description="Brief diagnostic analysis summary")

    @model_validator(mode="after")
    def validate_rankings(self) -> "FinalDiagnosisContent":
        diagnosis_codes = [item.icd_code for item in self.topk_diagnoses]
        if len(diagnosis_codes) != len(set(diagnosis_codes)):
            duplicate_codes = sorted(
                {
                    icd_code
                    for icd_code in diagnosis_codes
                    if diagnosis_codes.count(icd_code) > 1
                }
            )
            raise ValueError(
                f"Final diagnosis ICD codes must be unique. Duplicates: {duplicate_codes}."
            )
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
            "Numbered guideline and PubMed evidence cited by the final diagnoses. Each item must "
            "use the format [number] source：evidence text."
        )
    )


class GuidelineSkillResult(BaseModel):
    skill_name: str = Field(description="Original local guideline skill name")
    disease_name: str = Field(description="Disease evaluated by this guideline skill")
    guideline_evidence: list[str] = Field(
        description=(
            "Relevant evidence retrieved and verified according to this skill's SKILL.md workflow"
        )
    )
    guideline_diagnosis: str = Field(
        description=(
            "Concise natural-language conclusion comparing the patient's structured findings with the "
            "verified guideline information and stating whether the patient may have this disease"
        )
    )


class GuidelineDirectSkillMatch(BaseModel):
    skill_name: str = Field(description="Exact directly matched local guideline skill name")


class GuidelineExpandedSkillMatch(BaseModel):
    skill_name: str = Field(description="Exact skill name selected by forward expansion")
    source_skill_name: str = Field(
        description="Directly matched skill whose differential disease caused this expansion"
    )
    differential_disease: str = Field(
        description="Explicit differential disease linking the source skill to this skill"
    )


class GuidelineDirectSkillSelection(BaseModel):
    direct_matches: list[GuidelineDirectSkillMatch] = Field(
        default_factory=list,
        description="Skills whose primary disease directly matched a diagnostic hypothesis",
    )
    unused_reason: str | None = Field(
        description="Specific reason no skill matched; null when at least one skill was selected",
    )


class GuidelineDifferentialSkillTargets(BaseModel):
    differential_disease: str = Field(
        description="Exact differential disease copied from the source skill description"
    )
    skill_names: list[str] = Field(
        default_factory=list,
        description="Exact available skill names whose primary disease directly matches it",
    )


class GuidelineSourceSkillExpansion(BaseModel):
    source_skill_name: str = Field(
        description="Exact directly matched source skill name"
    )
    differential_matches: list[GuidelineDifferentialSkillTargets] = Field(
        default_factory=list,
        description="Target skills grouped by an explicit source-skill differential disease",
    )


class GuidelineSkillExpansionSelection(BaseModel):
    source_matches: list[GuidelineSourceSkillExpansion] = Field(
        default_factory=list,
        description="One-hop differential expansion results grouped by direct source skill",
    )


class GuidelineSearchResult(BaseModel):
    used_skill: bool = Field(description="Whether any guideline skill was loaded and searched")
    unused_reason: str | None = Field(
        description="Reason no guideline skill was used; null when used_skill is true",
    )
    direct_matches: list[GuidelineDirectSkillMatch] = Field(
        default_factory=list,
        description="Skills whose primary disease directly matched a diagnostic hypothesis",
    )
    expanded_matches: list[GuidelineExpandedSkillMatch] = Field(
        default_factory=list,
        description="One-hop forward differential expansions from directly matched skills",
    )
    skill_results: list[GuidelineSkillResult] = Field(
        description="Guideline evidence and diagnostic conclusion grouped by used skill"
    )
    reason: str | None = Field(
        default=None,
        description="Failure reason when guideline search returns an empty result",
    )

    @computed_field
    @property
    def skill_names(self) -> list[str]:
        return list(
            dict.fromkeys(
                skill_result.skill_name
                for skill_result in self.skill_results
            )
        )

    @model_serializer(mode="wrap")
    def serialize_with_skill_names_before_results(self, handler):
        serialized = handler(self)
        if "skill_names" not in serialized or "skill_results" not in serialized:
            return serialized

        skill_names = serialized.pop("skill_names")
        ordered = {}
        for field_name, value in serialized.items():
            if field_name == "skill_results":
                ordered["skill_names"] = skill_names
            ordered[field_name] = value
        return ordered


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


class LlmHypothesesResult(BaseModel):
    llm_hypotheses: list[HypothesisItem] = Field(
        max_length=5,
        description=(
            "Up to 5 principal-diagnosis hypotheses generated directly from the original case text"
        ),
    )


class PositiveFeaturesResult(BaseModel):
    present_illness_history: list[str] = Field(
        description=(
            "Chief complaint and positive current history-of-present-illness findings extracted "
            "directly from the original case text, excluding denied or absent symptoms"
        )
    )
    past_medical_history: list[str] = Field(
        description=(
            "Preadmission medications, procedures completed before the current admission, and "
            "past medical conditions extracted directly from the original case text"
        )
    )
    physical_exam: list[str] = Field(
        description=(
            "Positive physical-examination findings and abnormal vital signs from the original "
            "case text, excluding negative or normal findings"
        )
    )
    family_history: list[str] = Field(
        description=(
            "Explicitly documented relevant diseases in affected relatives, excluding negative "
            "family history"
        )
    )
    pertinent_results: list[str] = Field(
        description=(
            "Positive or abnormal laboratory, imaging, endoscopic, pathology, and microbiology "
            "results from the original case text, excluding negative or normal results"
        )
    )


class PreprocessingResult(BaseModel):
    llm_hypotheses: list[HypothesisItem] = Field(
        max_length=5,
        description=(
            "Up to 5 principal-diagnosis hypotheses generated directly from the original case text"
        ),
    )
    positive_features: PositiveFeaturesResult = Field(
        description="Patient findings structured into the five similar-case matching fields"
    )


class SearchPlanningResult(BaseModel):
    hypotheses: list[HypothesisItem] = Field(
        max_length=10,
        description=(
            "Up to 10 unique principal-diagnosis candidates merged from direct LLM hypotheses "
            "and similar cases"
        ),
    )
    search_queries: list[str] = Field(
        max_length=10,
        description=(
            "Five to ten concise PubMed search queries that collectively cover every hypothesis; "
            "each query combines one or more clinically relevant candidate diseases with the most "
            "discriminative positive patient feature, and may be empty only when search planning fails"
        ),
    )
    reason: str | None = Field(
        default=None,
        description="Failure reason when search planning returns an empty result",
    )


class SimilarCaseSection(BaseModel):
    section: str = Field(description="Matched discharge summary section name")
    content: str = Field(description="Matched discharge summary section content")


class ScoredSimilarCaseSection(SimilarCaseSection):
    score: float = Field(description="Section retrieval score")


class SimilarCaseCandidate(BaseModel):
    discharge_disease: str = Field(description="Discharge disease of the similar case")
    icd_code: str = Field(description="ICD code corresponding to the discharge disease")


class RetrievedSimilarCase(SimilarCaseCandidate):
    hadm_id: str = Field(description="Hospital admission ID of the similar case")
    score: float = Field(description="Aggregated case retrieval score")
    sections: list[ScoredSimilarCaseSection] = Field(
        description="Top matched sections in this retrieval branch"
    )


class FusedSimilarCase(SimilarCaseCandidate):
    hadm_id: str = Field(description="Hospital admission ID of the similar case")
    rrf_score: float = Field(description="Reciprocal rank fusion score")
    bm25_rank: int | None = Field(description="Case rank in the BM25 branch")
    embedding_rank: int | None = Field(
        description="Case rank in the dense embedding branch"
    )
    bm25_sections: list[ScoredSimilarCaseSection] = Field(
        description="Top sections matched by BM25"
    )
    embedding_sections: list[ScoredSimilarCaseSection] = Field(
        description="Top sections matched by dense embedding retrieval"
    )
    sections: list[ScoredSimilarCaseSection] = Field(
        description="Top matched sections after weighted rank fusion"
    )


class SimilarCaseRetrievalResult(BaseModel):
    bm25: list[RetrievedSimilarCase] = Field(
        default_factory=list,
        max_length=5,
        description="Top five diseases and matched sections after BM25 retrieval",
    )
    embedding: list[RetrievedSimilarCase] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Top five diseases and matched sections after dense embedding retrieval"
        ),
    )
    rrf: list[FusedSimilarCase] = Field(
        default_factory=list,
        max_length=5,
        description="Top five diseases and retrieval details after rank fusion",
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
    llm_hypotheses_result: LlmHypothesesResult
    positive_features_result: PositiveFeaturesResult
    multi_round_diagnosis: MultiRoundDiagnosisResult


class DiagnosticJudgementResult(BaseModel):
    closer_result: Literal["final_diagnoses", "search_planning_diagnoses"] = Field(
        description="Which candidate diagnosis set is closer to the patient information"
    )
    reason: str = Field(description="Reasoning for the diagnostic judgement")
