from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections.abc import Callable
from typing import TypeVar

from agents import (
    Agent,
    MaxTurnsExceeded,
    Model,
    ModelSettings,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    RunConfig,
    Runner,
)
from agents.sandbox import SandboxRunConfig
from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient
from openai import AsyncOpenAI
from pydantic import BaseModel

from config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_THINKING,
    OPENAI_MODEL,
)
from diagnosis.agents.digestive_diagnosis_agent import build_digestive_diagnosis_agent
from diagnosis.agents.diagnostic_judgement_agent import build_diagnostic_judgement_agent
from diagnosis.agents.guideline_searcher_agent import (
    SKILLS_DIR,
    GuidelineToolState,
    build_guideline_searcher_agent,
)
from diagnosis.agents.knowledge_searcher_agent import (
    build_knowledge_searcher_agent,
    search_pubmed_queries,
)
from diagnosis.agents.preprocessing_agent import (
    build_preprocessing_agent,
)
from diagnosis.agents.search_planning_agent import build_search_planning_agent
from diagnosis.agents.similar_case_retrieval_agent import retrieve_similar_cases
from schemas import (
    DiagnosisPipelineResult,
    DiagnosisRoundResult,
    DiagnosisResult,
    DiagnosticJudgementResult,
    FinalDiagnosisContent,
    GuidelineSearchResult,
    HypothesisItem,
    KnowledgeSearchResult,
    KnowledgeSearchSelectionResult,
    LlmHypothesesResult,
    MultiRoundDiagnosisResult,
    PositiveFeaturesResult,
    PreprocessingResult,
    PubMedQueryResult,
    SearchPlanningResult,
    SimilarCaseRetrievalResult,
)


DiagnosisProgressCallback = Callable[[str, str, str | None], None]
StructuredResultT = TypeVar("StructuredResultT", bound=BaseModel)


def _to_jsonable(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def _as_json(model_object: object) -> str:
    return json.dumps(_to_jsonable(model_object), ensure_ascii=False, indent=2)


def _uses_native_structured_output(model: str | Model) -> bool:
    return not isinstance(model, OpenAIChatCompletionsModel)


def _diagnosis_model_settings(model: str | Model) -> ModelSettings:
    if not isinstance(model, OpenAIChatCompletionsModel):
        return ModelSettings(temperature=0)
    thinking_type = "enabled" if DEEPSEEK_THINKING else "disabled"
    return ModelSettings(
        temperature=0,
        max_tokens=16384,
        extra_body={"thinking": {"type": thinking_type}},
        extra_args={"response_format": {"type": "json_object"}},
    )


def _prepare_structured_prompt(
    prompt: str,
    output_type: type[BaseModel],
    *,
    native_structured_output: bool,
) -> str:
    if native_structured_output:
        return prompt
    return (
        f"{prompt}\n\n"
        "## Output Requirements\n\n"
        "Return only one valid JSON object matching this JSON Schema. "
        "Do not wrap the JSON in Markdown fences or add explanatory text.\n\n"
        "<OUTPUT_SCHEMA>\n"
        f"{json.dumps(output_type.model_json_schema(), ensure_ascii=False)}\n"
        "</OUTPUT_SCHEMA>"
    )


def _parse_structured_result(
    result: object,
    output_type: type[StructuredResultT],
) -> StructuredResultT:
    if isinstance(result, output_type):
        return result
    stripped = str(result).strip()
    if not stripped:
        raise ValueError("Model returned empty JSON output.")
    return output_type.model_validate_json(stripped)


def _stage_failure_reason(stage_name: str, exc: Exception) -> str:
    return f"{stage_name} failed: {type(exc).__name__}: {exc}"


def _print_debug_section(title: str, model_object: object) -> None:
    print(f"\n===== {title} =====", file=sys.stderr)
    print(_as_json(model_object), file=sys.stderr)


def _notify_agent_started(
    progress_callback: DiagnosisProgressCallback | None,
    agent_name: str,
    round_index: int | None,
) -> None:
    if progress_callback is not None:
        progress_callback(
            "agent_started",
            agent_name,
            str(round_index) if round_index is not None else None,
        )


def _publish_stage_result(
    title: str,
    model_object: object,
    *,
    debug: bool,
    progress_callback: DiagnosisProgressCallback | None,
) -> None:
    if debug:
        _print_debug_section(title, model_object)
    if progress_callback is not None:
        progress_callback("stage_completed", title, _as_json(model_object))


async def _run_preprocessing_async(
    case_text: str,
    *,
    model: str | Model,
    debug: bool = False,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> PreprocessingResult:
    native_structured_output = _uses_native_structured_output(model)
    agent = build_preprocessing_agent(
        model,
        native_structured_output=native_structured_output,
    )
    prompt = _prepare_structured_prompt(
        (
            "<PATIENT_INFORMATION>\n"
            f"{case_text}\n"
            "</PATIENT_INFORMATION>"
        ),
        PreprocessingResult,
        native_structured_output=native_structured_output,
    )
    _notify_agent_started(
        progress_callback,
        "Preprocessing Agent",
        None,
    )
    raw_result = (
        await Runner.run(
            agent,
            prompt,
            run_config=RunConfig(model_settings=_diagnosis_model_settings(model)),
        )
    ).final_output
    result = _parse_structured_result(raw_result, PreprocessingResult)
    _publish_stage_result(
        "Preprocessing Result",
        result,
        debug=debug,
        progress_callback=progress_callback,
    )
    return result


def _merge_planning_hypotheses(
    llm_hypotheses_result: LlmHypothesesResult,
    similar_case_retrieval_result: SimilarCaseRetrievalResult,
) -> list[HypothesisItem]:
    merged_hypotheses: list[HypothesisItem] = []
    seen_codes: set[str] = set()
    for hypothesis in llm_hypotheses_result.llm_hypotheses:
        if hypothesis.icd_code in seen_codes:
            continue
        seen_codes.add(hypothesis.icd_code)
        merged_hypotheses.append(hypothesis)

    for similar_case in similar_case_retrieval_result.rerank:
        hypothesis = HypothesisItem(
            icd_code=similar_case.icd_code,
            category_name=similar_case.discharge_disease,
        )
        if hypothesis.icd_code in seen_codes:
            continue
        seen_codes.add(hypothesis.icd_code)
        merged_hypotheses.append(hypothesis)
        if len(merged_hypotheses) == 10:
            break
    return [
        hypothesis
        for hypothesis in merged_hypotheses
        if not any(
            other.icd_code.startswith(hypothesis.icd_code)
            and len(other.icd_code) > len(hypothesis.icd_code)
            for other in merged_hypotheses
        )
    ]


async def _run_search_planning_async(
    case_text: str,
    llm_hypotheses_result: LlmHypothesesResult,
    positive_features_result: PositiveFeaturesResult,
    similar_case_retrieval_result: SimilarCaseRetrievalResult,
    *,
    model: str | Model,
    previous_search_planning_result: SearchPlanningResult | None = None,
    previous_diagnosis_result: DiagnosisResult | None = None,
    diagnostic_judgement_result: DiagnosticJudgementResult | None = None,
    previous_guideline_evidence: list[str] | None = None,
    debug: bool = False,
    round_index: int | None = None,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> SearchPlanningResult:
    native_structured_output = _uses_native_structured_output(model)
    search_planning_agent = build_search_planning_agent(
        model,
        native_structured_output=native_structured_output,
    )
    merged_hypotheses = _merge_planning_hypotheses(
        llm_hypotheses_result,
        similar_case_retrieval_result,
    )
    similar_case_diagnoses = [
        {
            "discharge_disease": similar_case.discharge_disease,
            "icd_code": similar_case.icd_code,
        }
        for similar_case in similar_case_retrieval_result.rerank
    ]
    search_planning_prompt = (
        "<PATIENT_INFORMATION>\n"
        f"{case_text}\n"
        "</PATIENT_INFORMATION>\n\n"
        "<LLM_HYPOTHESES_RESULT>\n"
        f"{_as_json(llm_hypotheses_result)}\n"
        "</LLM_HYPOTHESES_RESULT>\n\n"
        "<POSITIVE_FEATURES_RESULT>\n"
        f"{_as_json(positive_features_result)}\n"
        "</POSITIVE_FEATURES_RESULT>\n\n"
        "<SIMILAR_CASE_DIAGNOSES>\n"
        f"{_as_json(similar_case_diagnoses)}\n"
        "</SIMILAR_CASE_DIAGNOSES>\n\n"
        "<MERGED_HYPOTHESES>\n"
        f"{_as_json(merged_hypotheses)}\n"
        "</MERGED_HYPOTHESES>\n\n"
        "## Task\n\n"
        "Copy MERGED_HYPOTHESES exactly into hypotheses and generate focused search_queries. "
        "Set reason to null when planning completes normally."
    )
    if previous_search_planning_result and previous_diagnosis_result and diagnostic_judgement_result:
        search_planning_prompt += (
            "\n\n"
            "<PREVIOUS_SEARCH_PLANNING_RESULT>\n"
            f"{_as_json(previous_search_planning_result)}\n"
            "</PREVIOUS_SEARCH_PLANNING_RESULT>\n\n"
            "<PREVIOUS_DIAGNOSIS_RESULT>\n"
            f"{_as_json(previous_diagnosis_result)}\n"
            "</PREVIOUS_DIAGNOSIS_RESULT>\n\n"
            "<DIAGNOSTIC_JUDGEMENT>\n"
            f"{_as_json(diagnostic_judgement_result)}\n"
            "</DIAGNOSTIC_JUDGEMENT>\n\n"
            "<PREVIOUS_GUIDELINE_EVIDENCE>\n"
            f"{_as_json(previous_guideline_evidence or [])}\n"
            "</PREVIOUS_GUIDELINE_EVIDENCE>\n\n"
            "## Task\n\n"
            "The diagnostic judgement found that search_planning_diagnoses were closer to the "
            "patient information than the previous final_diagnoses. Regenerate improved "
            "search_queries for the next diagnosis round. Copy MERGED_HYPOTHESES exactly into "
            "hypotheses. Use the previous artifacts, including previous guideline evidence, only to "
            "improve the retrieval strategy, and do not treat their contents as new patient facts."
        )

    search_planning_prompt = _prepare_structured_prompt(
        search_planning_prompt,
        SearchPlanningResult,
        native_structured_output=native_structured_output,
    )

    _notify_agent_started(progress_callback, "Search Planning Agent", round_index)
    raw_result = (
        await Runner.run(
            search_planning_agent,
            search_planning_prompt,
            run_config=RunConfig(model_settings=_diagnosis_model_settings(model)),
        )
    ).final_output
    raw_planning_result = _parse_structured_result(raw_result, SearchPlanningResult)
    result = SearchPlanningResult(
        hypotheses=merged_hypotheses,
        search_queries=raw_planning_result.search_queries,
        reason=raw_planning_result.reason,
    )
    _publish_stage_result(
        f"Search Planning Result - Round {round_index}",
        result,
        debug=debug,
        progress_callback=progress_callback,
    )
    return result


async def _run_search_planning_with_fallback(
    case_text: str,
    llm_hypotheses_result: LlmHypothesesResult,
    positive_features_result: PositiveFeaturesResult,
    similar_case_retrieval_result: SimilarCaseRetrievalResult,
    *,
    model: str | Model,
    previous_search_planning_result: SearchPlanningResult | None = None,
    previous_diagnosis_result: DiagnosisResult | None = None,
    diagnostic_judgement_result: DiagnosticJudgementResult | None = None,
    previous_guideline_evidence: list[str] | None = None,
    debug: bool = False,
    round_index: int | None = None,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> SearchPlanningResult:
    try:
        return await _run_search_planning_async(
            case_text,
            llm_hypotheses_result,
            positive_features_result,
            similar_case_retrieval_result,
            model=model,
            previous_search_planning_result=previous_search_planning_result,
            previous_diagnosis_result=previous_diagnosis_result,
            diagnostic_judgement_result=diagnostic_judgement_result,
            previous_guideline_evidence=previous_guideline_evidence,
            debug=debug,
            round_index=round_index,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        merged_hypotheses = _merge_planning_hypotheses(
            llm_hypotheses_result,
            similar_case_retrieval_result,
        )
        result = SearchPlanningResult(
            hypotheses=merged_hypotheses,
            search_queries=[],
            reason=_stage_failure_reason("Search planning", exc),
        )
        _publish_stage_result(
            f"Search Planning Result - Round {round_index}",
            result,
            debug=debug,
            progress_callback=progress_callback,
        )
        return result


async def _run_knowledge_search_async(
    search_queries: list[str],
    *,
    model: str | Model,
    debug: bool = False,
    round_index: int | None = None,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> KnowledgeSearchResult:
    selected_queries = search_queries
    pubmed_results = await asyncio.to_thread(
        search_pubmed_queries,
        selected_queries,
    )
    native_structured_output = _uses_native_structured_output(model)
    knowledge_agent = build_knowledge_searcher_agent(
        model,
        native_structured_output=native_structured_output,
    )
    knowledge_prompt = (
        "<SEARCH_QUERIES>\n"
        f"{_as_json(selected_queries)}\n"
        "</SEARCH_QUERIES>\n\n"
        "<PUBMED_SEARCH_RESULTS>\n"
        f"{_as_json(pubmed_results)}\n"
        "</PUBMED_SEARCH_RESULTS>"
    )
    knowledge_prompt = _prepare_structured_prompt(
        knowledge_prompt,
        KnowledgeSearchSelectionResult,
        native_structured_output=native_structured_output,
    )
    _notify_agent_started(progress_callback, "Knowledge Searcher Agent", round_index)
    raw_result = (
        await Runner.run(
            knowledge_agent,
            knowledge_prompt,
            run_config=RunConfig(model_settings=_diagnosis_model_settings(model)),
        )
    ).final_output
    selection_result = _parse_structured_result(
        raw_result,
        KnowledgeSearchSelectionResult,
    )
    selected_sections = {
        (section.pmid, section.section_index)
        for section in selection_result.selected_sections
    }
    relevant_pubmed_results = []
    for query_result in pubmed_results:
        relevant_results = []
        for pubmed_result in query_result["results"]:
            abstract_sections = [
                section
                for section in pubmed_result["abstract_sections"]
                if (
                    pubmed_result["pmid"],
                    section["section_index"],
                )
                in selected_sections
            ]
            if abstract_sections:
                relevant_results.append(
                    {
                        **pubmed_result,
                        "abstract_sections": abstract_sections,
                    }
                )
        if relevant_results:
            relevant_pubmed_results.append(
                PubMedQueryResult(
                    query=query_result["query"],
                    results=relevant_results,
                )
            )
    result = KnowledgeSearchResult(
        relevant_pubmed_results=relevant_pubmed_results,
        reason="; ".join(
            query_result["reason"]
            for query_result in pubmed_results
            if query_result.get("reason")
        ) or None,
    )
    _publish_stage_result(
        f"Knowledge Search Result - Round {round_index}",
        result,
        debug=debug,
        progress_callback=progress_callback,
    )
    return result


def _format_pubmed_results(
    knowledge_search_result: KnowledgeSearchResult,
) -> list[str]:
    return [
        f"PubMed PMID {item.pmid}（{item.title}）：{section.text}"
        for query_result in knowledge_search_result.relevant_pubmed_results
        for item in query_result.results
        for section in item.abstract_sections
    ]


def _run_similar_case_retrieval(
    positive_features: list[str],
    *,
    debug: bool = False,
    round_index: int | None = None,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> SimilarCaseRetrievalResult:
    _notify_agent_started(progress_callback, "Similar Case Retrieval Agent", round_index)
    ranking_details: list[dict[str, object]] = []
    result = retrieve_similar_cases(
        positive_features,
        debug=debug,
        ranking_callback=(
            ranking_details.append
            if progress_callback is not None
            else None
        ),
    )
    if progress_callback is not None:
        progress_callback(
            "stage_completed",
            f"Similar Case Retrieval Rankings - Round {round_index}",
            _as_json({"rankings": ranking_details}),
        )
    _publish_stage_result(
        f"Similar Case Retrieval Result - Round {round_index}",
        result,
        debug=debug,
        progress_callback=progress_callback,
    )
    return result


async def _run_guideline_search_async(
    hypotheses: list[HypothesisItem],
    positive_features: list[str],
    *,
    model: str | Model,
    debug: bool = False,
    round_index: int | None = None,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> GuidelineSearchResult:
    native_structured_output = _uses_native_structured_output(model)
    guideline_agent = build_guideline_searcher_agent(
        GuidelineSearchResult,
        model,
        native_structured_output=native_structured_output,
    )
    guideline_prompt = (
        "<DIAGNOSTIC_HYPOTHESES>\n"
        f"{_as_json(hypotheses)}\n"
        "</DIAGNOSTIC_HYPOTHESES>\n\n"
        "## Skill Selection\n\n"
        "Select skills only when their disease scope directly corresponds to at least one diagnostic "
        "hypothesis. A shared broad category without a specific disease match is insufficient. Do not "
        "select a skill from an unrelated disease category. Every subtype, stage, metastatic site, "
        "complication, procedure, or other condition required by a skill must be explicit in the "
        "diagnostic hypothesis set; a general disease hypothesis alone does not match a narrower "
        "skill. Do not infer these conditions from positive_features.\n\n"
        "<POSITIVE_FEATURES>\n"
        f"{_as_json(positive_features)}\n"
        "</POSITIVE_FEATURES>\n\n"
        "<AVAILABLE_SKILLS_DIRECTORY>\n"
        f"{SKILLS_DIR}\n"
        "</AVAILABLE_SKILLS_DIRECTORY>\n\n"
        "## Guideline Retrieval\n\n"
        "After selecting the skills, use only positive_features to search within each skill and compare "
        "the patient features with guideline information verified against the guideline full text. "
        "Do not use hypotheses as within-skill search terms or patient evidence. Keep each skill's "
        "evidence and guideline diagnosis together in one skill_results item.\n\n"
        "## Result Requirements\n\n"
        "When no guideline skill is used, set used_skill to false, explain the specific reason in "
        "unused_reason, and return an empty skill_results array. Set unused_reason to null when at "
        "least one guideline skill is used."
    )
    if not native_structured_output:
        guideline_prompt = _prepare_structured_prompt(
            guideline_prompt,
            GuidelineSearchResult,
            native_structured_output=False,
        )
        guideline_prompt += (
            "\n\n## JSON Output Example\n\n"
            "The following example shows the required JSON structure. Replace every example value "
            "with the actual result. Use the JSON null value, not the string \"null\".\n\n"
            "<JSON_OUTPUT_EXAMPLE>\n"
            '{"used_skill":true,"unused_reason":null,"skill_results":['
            '{"skill_name":"actual selected skill name","disease_name":"evaluated disease",'
            '"guideline_evidence":[],"guideline_diagnosis":"actual diagnostic conclusion"}],'
            '"reason":null}\n'
            "</JSON_OUTPUT_EXAMPLE>"
        )
    _notify_agent_started(progress_callback, "Guideline Searcher Agent", round_index)
    if native_structured_output:
        raw_result = (
            await Runner.run(
                guideline_agent,
                guideline_prompt,
                run_config=RunConfig(
                    model_settings=_diagnosis_model_settings(model),
                    sandbox=SandboxRunConfig(
                        client=UnixLocalSandboxClient(),
                    ),
                ),
            )
        ).final_output
    else:
        tool_state = GuidelineToolState()
        try:
            raw_result = (
                await Runner.run(
                    guideline_agent,
                    guideline_prompt,
                    context=tool_state,
                    max_turns=100,
                    run_config=RunConfig(model_settings=_diagnosis_model_settings(model)),
                )
            ).final_output
        except MaxTurnsExceeded:
            accessed_skills = sorted(tool_state.accessed_skills)
            attempted_detail = (
                f" Attempted skills: {', '.join(accessed_skills)}."
                if accessed_skills
                else ""
            )
            result = GuidelineSearchResult(
                used_skill=False,
                unused_reason="Guideline search did not complete.",
                skill_results=[],
                reason=(
                    "Guideline search exceeded the maximum number of agent turns before producing "
                    f"a final result.{attempted_detail}"
                ),
            )
            _publish_stage_result(
                f"Guideline Search Result - Round {round_index}",
                result,
                debug=debug,
                progress_callback=progress_callback,
            )
            return result
    try:
        result = _parse_structured_result(raw_result, GuidelineSearchResult)
    except ValueError as first_exc:
        if native_structured_output:
            result = GuidelineSearchResult(
                used_skill=False,
                unused_reason="Guideline search failed.",
                skill_results=[],
                reason=(
                    "Guideline search result could not be parsed as valid structured output: "
                    f"{type(first_exc).__name__}: {first_exc}"
                ),
            )
        else:
            retry_accessed_skills: list[str] = []
            retry_tool_state: GuidelineToolState | None = None
            try:
                if str(raw_result).strip():
                    repair_agent = Agent(
                        name="Guideline Result Repair Agent",
                        model=model,
                        instructions=(
                            "Repair one invalid GuidelineSearchResult JSON object without calling "
                            "tools. Preserve every supported skill name, disease name, and guideline "
                            "evidence item. Add or correct only fields required by the schema. For each "
                            "skill result, write a concise guideline_diagnosis using only the supplied "
                            "positive features and existing guideline evidence. Return exactly one JSON "
                            "object and no other text."
                        ),
                    )
                    repair_prompt = _prepare_structured_prompt(
                        (
                            "<POSITIVE_FEATURES>\n"
                            f"{_as_json(positive_features)}\n"
                            "</POSITIVE_FEATURES>\n\n"
                            "<INVALID_OUTPUT>\n"
                            f"{raw_result}\n"
                            "</INVALID_OUTPUT>\n\n"
                            "<VALIDATION_ERROR>\n"
                            f"{type(first_exc).__name__}: {first_exc}\n"
                            "</VALIDATION_ERROR>"
                        ),
                        GuidelineSearchResult,
                        native_structured_output=False,
                    )
                    retry_raw_result = (
                        await Runner.run(
                            repair_agent,
                            repair_prompt,
                            max_turns=1,
                            run_config=RunConfig(
                                model_settings=_diagnosis_model_settings(model)
                            ),
                        )
                    ).final_output
                else:
                    retry_tool_state = GuidelineToolState()
                    retry_raw_result = (
                        await Runner.run(
                            guideline_agent,
                            (
                                f"{guideline_prompt}\n\n"
                                "## Retry Requirement\n\n"
                                "The previous attempt returned an empty response. Repeat the bounded "
                                "guideline search once. After the permitted tool calls, stop calling "
                                "tools and return the complete final JSON object immediately."
                            ),
                            context=retry_tool_state,
                            max_turns=100,
                            run_config=RunConfig(
                                model_settings=_diagnosis_model_settings(model)
                            ),
                        )
                    ).final_output
                    retry_accessed_skills = sorted(retry_tool_state.accessed_skills)
                result = _parse_structured_result(
                    retry_raw_result,
                    GuidelineSearchResult,
                )
            except (MaxTurnsExceeded, ValueError) as retry_exc:
                if retry_tool_state is not None:
                    retry_accessed_skills = sorted(
                        retry_tool_state.accessed_skills
                    )
                attempted_detail = (
                    f" Attempted skills during retry: {', '.join(retry_accessed_skills)}."
                    if retry_accessed_skills
                    else ""
                )
                result = GuidelineSearchResult(
                    used_skill=False,
                    unused_reason="Guideline search failed after one retry.",
                    skill_results=[],
                    reason=(
                        "Initial guideline result error: "
                        f"{type(first_exc).__name__}: {first_exc}. "
                        "Retry error: "
                        f"{type(retry_exc).__name__}: {retry_exc}.{attempted_detail}"
                    ),
                )
    _publish_stage_result(
        f"Guideline Search Result - Round {round_index}",
        result,
        debug=debug,
        progress_callback=progress_callback,
    )
    return result


async def _run_final_diagnosis_async(
    case_text: str,
    search_planning_result: SearchPlanningResult,
    knowledge_search_result: KnowledgeSearchResult,
    guideline_search_result: GuidelineSearchResult,
    similar_case_retrieval_result: SimilarCaseRetrievalResult,
    *,
    model: str | Model,
    previous_diagnosis_result: DiagnosisResult | None = None,
    diagnostic_judgement_result: DiagnosticJudgementResult | None = None,
    corrective: bool = False,
    debug: bool = False,
    round_index: int | None = None,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> DiagnosisResult:
    native_structured_output = _uses_native_structured_output(model)
    diagnosis_agent = build_digestive_diagnosis_agent(
        FinalDiagnosisContent,
        model=model,
        native_structured_output=native_structured_output,
    )
    similar_case_summary = [
        {
            "rank": rank,
            "icd_code": similar_case.icd_code.strip().upper().replace(".", ""),
            "discharge_disease": similar_case.discharge_disease,
            "matched_sections": similar_case.sections,
        }
        for rank, similar_case in enumerate(
            similar_case_retrieval_result.rerank,
            start=1,
        )
    ]
    candidate_diagnoses = []
    candidate_names: dict[str, str] = {}
    for hypothesis in search_planning_result.hypotheses:
        if hypothesis.icd_code in candidate_names:
            continue
        candidate_names[hypothesis.icd_code] = hypothesis.category_name
        candidate_diagnoses.append(
            {
                "source": "search_planning",
                "icd_code": hypothesis.icd_code,
                "category_name": hypothesis.category_name,
            }
        )
    pubmed_results = _format_pubmed_results(knowledge_search_result)
    guideline_evidence = [
        f"{skill_result.skill_name}：{evidence}"
        for skill_result in guideline_search_result.skill_results
        for evidence in skill_result.guideline_evidence
    ]
    combined_evidence = [
        *guideline_evidence,
        *pubmed_results,
    ]
    numbered_evidence = [
        f"[{index}] {evidence}"
        for index, evidence in enumerate(combined_evidence, start=1)
    ]
    guideline_diagnoses = [
        skill_result.guideline_diagnosis
        for skill_result in guideline_search_result.skill_results
    ]
    revision_context = ""
    if (
        previous_diagnosis_result is not None
        and diagnostic_judgement_result is not None
    ):
        revision_context = (
            "<PREVIOUS_TOPK_DIAGNOSES>\n"
            f"{_as_json(previous_diagnosis_result.topk_diagnoses)}\n"
            "</PREVIOUS_TOPK_DIAGNOSES>\n\n"
            "<DIAGNOSTIC_JUDGEMENT>\n"
            f"{_as_json(diagnostic_judgement_result)}\n"
            "</DIAGNOSTIC_JUDGEMENT>\n\n"
            "## Revision Instructions\n\n"
            "Revise the diagnosis specifically to correct the candidate omissions and ranking "
            "problems identified by the diagnostic judgement.\n\n"
        )
    diagnosis_prompt = (
        "<PATIENT_INFORMATION>\n"
        f"{case_text}\n"
        "</PATIENT_INFORMATION>\n\n"
        "<CANDIDATE_DIAGNOSES>\n"
        f"{_as_json(candidate_diagnoses)}\n"
        "</CANDIDATE_DIAGNOSES>\n\n"
        "<NUMBERED_EVIDENCE>\n"
        f"{_as_json(numbered_evidence)}\n"
        "</NUMBERED_EVIDENCE>\n\n"
        "<GUIDELINE_RESULTS>\n"
        f"{_as_json(guideline_diagnoses)}\n"
        "</GUIDELINE_RESULTS>\n\n"
        "<SIMILAR_CASES>\n"
        f"{_as_json(similar_case_summary)}\n"
        "</SIMILAR_CASES>\n\n"
        f"{revision_context}"
        "## Task\n\n"
        "Please output exactly five ranked principal-diagnosis candidates."
    )
    diagnosis_prompt = _prepare_structured_prompt(
        diagnosis_prompt,
        FinalDiagnosisContent,
        native_structured_output=native_structured_output,
    )
    agent_name = (
        "Corrective Digestive Diagnosis Agent"
        if corrective
        else "Digestive Diagnosis Agent"
    )
    _notify_agent_started(progress_callback, agent_name, round_index)
    raw_result = (
        await Runner.run(
            diagnosis_agent,
            diagnosis_prompt,
            run_config=RunConfig(model_settings=_diagnosis_model_settings(model)),
        )
    ).final_output
    diagnosis_content = _parse_structured_result(raw_result, FinalDiagnosisContent)
    citation_pattern = re.compile(r"\[(\d+)\]")
    for diagnosis in diagnosis_content.topk_diagnoses:
        expected_category_name = candidate_names.get(diagnosis.icd_code)
        if (
            expected_category_name is not None
            and diagnosis.category_name != expected_category_name
        ):
            raise ValueError(
                f"Final diagnosis changed the candidate name for {diagnosis.icd_code}: "
                f"expected {expected_category_name!r}, got {diagnosis.category_name!r}"
            )
        if expected_category_name is None and not any(
            1 <= int(reference) <= len(numbered_evidence)
            for evidence in diagnosis.supporting_evidence
            for reference in citation_pattern.findall(evidence)
        ):
            raise ValueError(
                f"Final diagnosis outside the planning candidate set must cite supporting "
                f"guideline or PubMed evidence: {diagnosis.icd_code}."
            )

    selected_codes = {
        diagnosis.icd_code for diagnosis in diagnosis_content.topk_diagnoses
    }
    planning_candidates = {
        hypothesis.icd_code: hypothesis.category_name
        for hypothesis in search_planning_result.hypotheses
    }
    refined_planning_codes = {
        planning_code
        for diagnosis in diagnosis_content.topk_diagnoses
        if diagnosis.icd_code not in planning_candidates
        for planning_code in planning_candidates
        if diagnosis.icd_code[:3] == planning_code[:3]
    }
    retained_refined_sources = refined_planning_codes & selected_codes
    if retained_refined_sources:
        raise ValueError(
            "A planning candidate replaced by an ICD code with the same first three characters "
            "must be excluded rather than retained unchanged. "
            f"Conflicting candidates: {sorted(retained_refined_sources)}."
        )
    excluded_candidates = {
        candidate.icd_code: candidate
        for candidate in diagnosis_content.excluded_planning_candidates
    }
    expected_excluded_codes = set(planning_candidates) - selected_codes
    missing_excluded_codes = expected_excluded_codes - set(excluded_candidates)
    if missing_excluded_codes:
        raise ValueError(
            "Every search planning candidate omitted from the final top five must include "
            "patient-grounded exclusion or ICD correction reasons. "
            f"Missing {sorted(missing_excluded_codes)}; "
            f"final top five contains {sorted(selected_codes)}."
        )
    validated_excluded_candidates = []
    for icd_code in planning_candidates:
        if icd_code not in expected_excluded_codes:
            continue
        candidate = excluded_candidates[icd_code]
        if candidate.category_name != planning_candidates[icd_code]:
            raise ValueError(
                f"Excluded planning candidate changed the candidate name for {icd_code}."
            )
        if not all(
            evidence.strip() for evidence in candidate.patient_contrary_evidence
        ):
            raise ValueError(
                f"Excluded planning candidate {icd_code} has an empty exclusion or correction reason."
            )
        validated_excluded_candidates.append(candidate)

    skill_names = [
        skill_result.skill_name
        for skill_result in guideline_search_result.skill_results
    ]
    referenced_numbers = {
        int(reference)
        for text in [
            *[
                item
                for diagnosis in diagnosis_content.topk_diagnoses
                for item in [
                    *diagnosis.supporting_evidence,
                    *diagnosis.recommended_next_steps,
                ]
            ],
            *[
                item
                for candidate in validated_excluded_candidates
                for item in candidate.patient_contrary_evidence
            ],
        ]
        for reference in citation_pattern.findall(text)
        if 1 <= int(reference) <= len(numbered_evidence)
    }
    ordered_numbers = [
        number
        for number in range(1, len(numbered_evidence) + 1)
        if number in referenced_numbers
    ]
    citation_mapping = {
        old_number: new_number
        for new_number, old_number in enumerate(ordered_numbers, start=1)
    }
    filtered_evidence = [
        citation_pattern.sub(
            f"[{citation_mapping[old_number]}]",
            numbered_evidence[old_number - 1],
            count=1,
        )
        for old_number in ordered_numbers
    ]
    for diagnosis in diagnosis_content.topk_diagnoses:
        diagnosis.supporting_evidence = [
            citation_pattern.sub(
                lambda match: (
                    f"[{citation_mapping[int(match.group(1))]}]"
                    if int(match.group(1)) in citation_mapping
                    else ""
                ),
                text,
            ).strip()
            for text in diagnosis.supporting_evidence
        ]
        diagnosis.recommended_next_steps = [
            citation_pattern.sub(
                lambda match: (
                    f"[{citation_mapping[int(match.group(1))]}]"
                    if int(match.group(1)) in citation_mapping
                    else ""
                ),
                text,
            ).strip()
            for text in diagnosis.recommended_next_steps
        ]
    for candidate in validated_excluded_candidates:
        candidate.patient_contrary_evidence = [
            citation_pattern.sub(
                lambda match: (
                    f"[{citation_mapping[int(match.group(1))]}]"
                    if int(match.group(1)) in citation_mapping
                    else ""
                ),
                text,
            ).strip()
            for text in candidate.patient_contrary_evidence
        ]
    result = DiagnosisResult(
        used_skill=guideline_search_result.used_skill,
        skill_names=skill_names,
        topk_diagnoses=diagnosis_content.topk_diagnoses,
        excluded_planning_candidates=validated_excluded_candidates,
        summary=diagnosis_content.summary,
        evidence=filtered_evidence,
    )
    result_title = (
        f"Corrective Final Diagnosis Result - Round {round_index}"
        if corrective
        else f"Final Diagnosis Result - Round {round_index}"
    )
    _publish_stage_result(
        result_title,
        result,
        debug=debug,
        progress_callback=progress_callback,
    )
    return result


async def _run_diagnostic_judgement_async(
    case_text: str,
    search_planning_diagnoses: list[HypothesisItem],
    diagnosis_result: DiagnosisResult,
    *,
    model: str | Model,
    debug: bool = False,
    round_index: int | None = None,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> DiagnosticJudgementResult:
    native_structured_output = _uses_native_structured_output(model)
    diagnostic_judgement_agent = build_diagnostic_judgement_agent(
        model,
        native_structured_output=native_structured_output,
    )
    final_diagnoses = [
        {
            "icd_code": diagnosis.icd_code,
            "category_name": diagnosis.category_name,
        }
        for diagnosis in diagnosis_result.topk_diagnoses
    ]
    diagnostic_judgement_prompt = (
        "<PATIENT_INFORMATION>\n"
        f"{case_text}\n"
        "</PATIENT_INFORMATION>\n\n"
        "<SEARCH_PLANNING_DIAGNOSES>\n"
        f"{_as_json(search_planning_diagnoses)}\n"
        "</SEARCH_PLANNING_DIAGNOSES>\n\n"
        "<FINAL_DIAGNOSES>\n"
        f"{_as_json(final_diagnoses)}\n"
        "</FINAL_DIAGNOSES>\n\n"
        "## Task\n\n"
        "Judge whether final_diagnoses or search_planning_diagnoses is closer to the patient "
        "information. "
        "Keep closer_result as the required enum value."
    )
    diagnostic_judgement_prompt = _prepare_structured_prompt(
        diagnostic_judgement_prompt,
        DiagnosticJudgementResult,
        native_structured_output=native_structured_output,
    )
    _notify_agent_started(progress_callback, "Diagnostic Judgement Agent", round_index)
    raw_result = (
        await Runner.run(
            diagnostic_judgement_agent,
            diagnostic_judgement_prompt,
            run_config=RunConfig(model_settings=_diagnosis_model_settings(model)),
        )
    ).final_output
    result = _parse_structured_result(raw_result, DiagnosticJudgementResult)
    _publish_stage_result(
        f"Diagnostic Judgement Result - Round {round_index}",
        result,
        debug=debug,
        progress_callback=progress_callback,
    )
    return result


async def make_diagnosis_pipeline_async(
    case_text: str,
    *,
    model: str | Model | None = None,
    debug: bool = False,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> DiagnosisPipelineResult:
    diagnosis_model = model or OPENAI_MODEL
    max_diagnosis_rounds = 2

    preprocessing_result = await _run_preprocessing_async(
        case_text,
        model=diagnosis_model,
        debug=debug,
        progress_callback=progress_callback,
    )
    llm_hypotheses_result = LlmHypothesesResult(
        llm_hypotheses=preprocessing_result.llm_hypotheses
    )
    positive_features_result = PositiveFeaturesResult(
        positive_features=preprocessing_result.positive_features
    )
    try:
        similar_case_retrieval_result = await asyncio.to_thread(
            _run_similar_case_retrieval,
            positive_features_result.positive_features,
            debug=debug,
            round_index=1,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        similar_case_retrieval_result = SimilarCaseRetrievalResult(
            reason=_stage_failure_reason("Similar-case retrieval", exc),
        )
        _publish_stage_result(
            "Similar Case Retrieval Result - Round 1",
            similar_case_retrieval_result,
            debug=debug,
            progress_callback=progress_callback,
        )

    search_planning_result = await _run_search_planning_with_fallback(
        case_text,
        llm_hypotheses_result,
        positive_features_result,
        similar_case_retrieval_result,
        model=diagnosis_model,
        debug=debug,
        round_index=1,
        progress_callback=progress_callback,
    )
    previous_diagnosis_result: DiagnosisResult | None = None
    previous_diagnostic_judgement_result: DiagnosticJudgementResult | None = None
    diagnosis_rounds: list[DiagnosisRoundResult] = []

    for round_index in range(1, max_diagnosis_rounds + 1):
        (
            knowledge_search_outcome,
            guideline_search_outcome,
        ) = await asyncio.gather(
            _run_knowledge_search_async(
                search_planning_result.search_queries,
                model=diagnosis_model,
                debug=debug,
                round_index=round_index,
                progress_callback=progress_callback,
            ),
            _run_guideline_search_async(
                search_planning_result.hypotheses,
                positive_features_result.positive_features,
                model=diagnosis_model,
                debug=debug,
                round_index=round_index,
                progress_callback=progress_callback,
            ),
            return_exceptions=True,
        )

        if isinstance(knowledge_search_outcome, Exception):
            knowledge_search_result = KnowledgeSearchResult(
                relevant_pubmed_results=[],
                reason=_stage_failure_reason(
                    "Knowledge search",
                    knowledge_search_outcome,
                ),
            )
            _publish_stage_result(
                f"Knowledge Search Result - Round {round_index}",
                knowledge_search_result,
                debug=debug,
                progress_callback=progress_callback,
            )
        else:
            knowledge_search_result = knowledge_search_outcome

        if isinstance(guideline_search_outcome, Exception):
            guideline_search_result = GuidelineSearchResult(
                used_skill=False,
                unused_reason="Guideline search failed.",
                skill_results=[],
                reason=_stage_failure_reason(
                    "Guideline search",
                    guideline_search_outcome,
                ),
            )
            _publish_stage_result(
                f"Guideline Search Result - Round {round_index}",
                guideline_search_result,
                debug=debug,
                progress_callback=progress_callback,
            )
        else:
            guideline_search_result = guideline_search_outcome

        diagnosis_result = await _run_final_diagnosis_async(
            case_text,
            search_planning_result,
            knowledge_search_result,
            guideline_search_result,
            similar_case_retrieval_result,
            model=diagnosis_model,
            previous_diagnosis_result=previous_diagnosis_result,
            diagnostic_judgement_result=previous_diagnostic_judgement_result,
            debug=debug,
            round_index=round_index,
            progress_callback=progress_callback,
        )

        diagnostic_judgement_result = await _run_diagnostic_judgement_async(
            case_text,
            search_planning_result.hypotheses,
            diagnosis_result,
            model=diagnosis_model,
            debug=debug,
            round_index=round_index,
            progress_callback=progress_callback,
        )

        if (
            diagnostic_judgement_result.closer_result != "final_diagnoses"
            and round_index == max_diagnosis_rounds
        ):
            diagnosis_result = await _run_final_diagnosis_async(
                case_text,
                search_planning_result,
                knowledge_search_result,
                guideline_search_result,
                similar_case_retrieval_result,
                model=diagnosis_model,
                previous_diagnosis_result=diagnosis_result,
                diagnostic_judgement_result=diagnostic_judgement_result,
                corrective=True,
                debug=debug,
                round_index=round_index,
                progress_callback=progress_callback,
            )

        diagnosis_rounds.append(
            DiagnosisRoundResult(
                round=round_index,
                search_planning_result=search_planning_result,
                similar_case_retrieval_result=similar_case_retrieval_result,
                guideline_search_result=guideline_search_result,
                diagnosis_result=diagnosis_result,
            )
        )

        if (
            diagnostic_judgement_result.closer_result == "final_diagnoses"
            or round_index == max_diagnosis_rounds
        ):
            return DiagnosisPipelineResult(
                llm_hypotheses_result=llm_hypotheses_result,
                positive_features_result=positive_features_result,
                multi_round_diagnosis=MultiRoundDiagnosisResult(
                    is_multi_round=len(diagnosis_rounds) > 1,
                    rounds=diagnosis_rounds,
                )
            )

        previous_diagnosis_result = diagnosis_result
        previous_diagnostic_judgement_result = diagnostic_judgement_result
        search_planning_result = await _run_search_planning_with_fallback(
            case_text,
            llm_hypotheses_result,
            positive_features_result,
            similar_case_retrieval_result,
            model=diagnosis_model,
            previous_search_planning_result=search_planning_result,
            previous_diagnosis_result=diagnosis_result,
            diagnostic_judgement_result=diagnostic_judgement_result,
            previous_guideline_evidence=[
                f"{skill_result.skill_name}：{evidence}"
                for skill_result in guideline_search_result.skill_results
                for evidence in skill_result.guideline_evidence
            ],
            debug=debug,
            round_index=round_index + 1,
            progress_callback=progress_callback,
        )

async def make_diagnosis_async(
    case_text: str,
    *,
    model: str | Model | None = None,
    debug: bool = False,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> DiagnosisResult:
    pipeline_result = await make_diagnosis_pipeline_async(
        case_text,
        model=model,
        debug=debug,
        progress_callback=progress_callback,
    )
    return pipeline_result.multi_round_diagnosis.rounds[-1].diagnosis_result


def build_diagnosis_model(
    provider: str,
    *,
    openai_api_key: str = "",
    openai_model: str = "",
    deepseek_api_key: str = "",
    deepseek_model: str = "",
) -> Model:
    normalized_provider = provider.strip().lower()
    if normalized_provider == "openai":
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI API key is required.")
        return OpenAIResponsesModel(
            model=openai_model or OPENAI_MODEL,
            openai_client=AsyncOpenAI(api_key=api_key),
        )
    if normalized_provider == "deepseek":
        api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("DEEPSEEK API key is required.")
        return OpenAIChatCompletionsModel(
            model=deepseek_model or DEEPSEEK_MODEL,
            openai_client=AsyncOpenAI(
                api_key=api_key,
                base_url=DEEPSEEK_BASE_URL,
            ),
        )
    raise ValueError("Model provider must be openai or deepseek.")
