from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PhenotypeItem(BaseModel):
    # hpo: str = Field(description="HPO 编号，如无法确定则为空字符串")
    phenotype: str = Field(description="英文表型描述")


class PhenotypeExtractionResult(BaseModel):
    phenotypes: list[PhenotypeItem] = Field(description="从病例文本中提取的患者表型列表")


class TriageResult(BaseModel):
    has_preliminary_disease: bool = Field(description="是否已经存在较明确的初步疾病方向")
    suspected_diseases: list[str] = Field(description="病例文本中明确提到的疑似、考虑、倾向、初步诊断或待排疾病")
    should_use_skill: bool = Field(description="是否需要启用现有疾病指南 skill")
    matched_skills: list[str] = Field(description="命中的 skill 名称列表")
    reason: str = Field(description="触发或不触发 skill 的依据")
    topk_without_skill: list[str] = Field(description="不使用 skill 时的初步疑似诊断列表")


class DiagnosisItem(BaseModel):
    rank: int = Field(description="诊断排序，从 1 开始")
    disease: str = Field(description="疑似诊断疾病名称")
    confidence: int = Field(ge=0, le=100, description="置信度百分比整数，范围 0-100，例如 45 表示 45%")
    supporting_evidence: list[str] = Field(description="病例中支持该诊断的证据")
    missing_information: list[str] = Field(description="进一步确诊需要补充的信息")
    recommended_next_steps: list[str] = Field(description="建议下一步完善的检查或临床处理方向")
    guideline_evidence: list[str] = Field(default_factory=list, description="如使用指南 skill，列出指南依据")


class DiagnosisResult(BaseModel):
    used_skill: bool = Field(description="最终诊断阶段是否使用了 skill")
    skill_names: list[str] = Field(description="实际使用的 skill 名称列表")
    topk_diagnoses: list[DiagnosisItem] = Field(description="TopK 疑似诊断")
    summary: str = Field(description="简要诊断分析总结")
    safety_note: str = Field(description="医疗安全提示")


class SearchQueryItem(BaseModel):
    intent: Literal[
        "acute_problem",
        "most_likely_disease",
        "diagnostic_criteria_or_case_features",
        "key_differential_diagnosis",
    ] = Field(description="检索意图：当前急性问题、最可能的疾病、诊断标准或病例特征、关键鉴别诊断")
    query: str = Field(description="医学文献检索查询词")


class SearchPlanningResult(BaseModel):
    problem_representation: str = Field(description="当前最重要临床问题的简要表述")
    hypotheses: list[str] = Field(min_length=3, max_length=5, description="3-5 个主要候选诊断")
    search_queries: list[SearchQueryItem] = Field(max_length=5, description="最多 5 条医学文献检索查询")


class SimilarCaseDiagnosisItem(BaseModel):
    source_query: SearchQueryItem = Field(description="用于相似病例检索的原始查询")
    diagnosis: str = Field(description="该查询对应的相似病例诊断")
    matched_case_summary: str = Field(description="相似病例的简要摘要；如未接入真实病例库，应说明仅基于查询归纳")
    supporting_reason: str = Field(description="为什么该查询对应这个诊断")
    confidence: int = Field(ge=0, le=100, description="诊断匹配置信度百分比整数，范围 0-100")


class SimilarCaseRetrievalResult(BaseModel):
    items: list[SimilarCaseDiagnosisItem] = Field(description="每条检索查询对应的相似病例诊断结果")
    summary: str = Field(description="相似病例诊断结果的整体总结")
    limitations: list[str] = Field(description="当前相似病例检索结果的限制")
