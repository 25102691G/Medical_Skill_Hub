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
    build_guideline_expansion_agent,
    build_guideline_orchestrator_agent,
    build_guideline_result_filter_agent,
    build_guideline_skill_executor_agent,
    guideline_skill_catalog,
)
from diagnosis.agents.knowledge_searcher_agent import (
    build_knowledge_searcher_agent,
    search_pubmed_queries,
)
from diagnosis.agents.preprocessing_agent import (
    build_hypothesis_preprocessing_agent,
    build_positive_feature_preprocessing_agent,
)
from diagnosis.agents.search_planning_agent import build_search_planning_agent
from diagnosis.agents.similar_case_retrieval_agent import retrieve_similar_cases
from schemas import (
    DiagnosisPipelineResult,
    DiagnosisRoundResult,
    DiagnosisResult,
    DiagnosticJudgementResult,
    ExcludedPlanningCandidate,
    FinalDiagnosisContent,
    GuidelineDirectSkillSelection,
    GuidelineDirectSkillMatch,
    GuidelineExpandedResultSelection,
    GuidelineExpandedSkillMatch,
    GuidelineResultFilterResult,
    GuidelineSearchResult,
    GuidelineSkillExpansionSelection,
    GuidelineSkillResult,
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


class DiagnosisStageError(ValueError):
    def __init__(self, stage: str, error: Exception):
        self.stage = stage
        super().__init__(
            f"{stage} failed after one correction: {type(error).__name__}: {error}"
        )


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


def _max_turns_failure_reason(stage_name: str, exc: MaxTurnsExceeded) -> str:
    run_data = exc.run_data
    if run_data is None:
        return f"{stage_name} exceeded its maximum number of model turns."

    tool_calls = [
        item
        for item in run_data.new_items
        if getattr(item, "type", None) == "tool_call_item"
    ]
    tool_names = [item.tool_name or "unknown_tool" for item in tool_calls]
    tool_counts = {
        tool_name: tool_names.count(tool_name)
        for tool_name in dict.fromkeys(tool_names)
    }
    tool_call_signatures: list[tuple[str, str]] = []
    tool_call_summaries: list[str] = []
    for item in tool_calls:
        tool_name = item.tool_name or "unknown_tool"
        raw_item = item.raw_item
        arguments = (
            raw_item.get("arguments", "")
            if isinstance(raw_item, dict)
            else getattr(raw_item, "arguments", "")
        )
        argument_text = str(arguments)
        tool_call_signatures.append((tool_name, argument_text))
        try:
            parsed_arguments = json.loads(argument_text)
        except (TypeError, json.JSONDecodeError):
            parsed_arguments = argument_text
        if isinstance(parsed_arguments, dict):
            detail = parsed_arguments.get("cmd") or parsed_arguments.get(
                "skill_name"
            ) or json.dumps(parsed_arguments, ensure_ascii=False, sort_keys=True)
        else:
            detail = parsed_arguments
        normalized_detail = " ".join(str(detail).split())
        if len(normalized_detail) > 160:
            normalized_detail = f"{normalized_detail[:157]}..."
        tool_call_summaries.append(f"{tool_name}({normalized_detail})")

    signature_counts = {
        signature: tool_call_signatures.count(signature)
        for signature in dict.fromkeys(tool_call_signatures)
    }
    exact_duplicate_calls = sum(
        max(count - 1, 0) for count in signature_counts.values()
    )
    last_tool_summaries: list[str] = []
    if run_data.raw_responses:
        for item in run_data.raw_responses[-1].output:
            item_type = getattr(item, "type", "")
            if "call" not in item_type:
                continue
            tool_name = getattr(item, "name", None) or getattr(
                item, "tool_name", None
            )
            if tool_name:
                arguments = getattr(item, "arguments", "")
                normalized_arguments = " ".join(str(arguments).split())
                if len(normalized_arguments) > 160:
                    normalized_arguments = f"{normalized_arguments[:157]}..."
                last_tool_summaries.append(
                    f"{tool_name}({normalized_arguments})"
                )

    return (
        f"{stage_name} exceeded its maximum number of model turns. "
        f"Model calls: {len(run_data.raw_responses)}; tool calls: {len(tool_names)}; "
        f"tool call counts: {tool_counts or {}}; exact duplicate calls: "
        f"{exact_duplicate_calls}; tool call sequence: {tool_call_summaries}; "
        f"last response tool calls: {last_tool_summaries or []}."
    )


def _validate_final_diagnosis_content(
    diagnosis_content: FinalDiagnosisContent,
    candidate_names: dict[str, str],
    planning_candidates: dict[str, str],
) -> list[ExcludedPlanningCandidate]:
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
            evidence.strip() for evidence in diagnosis.supporting_evidence
        ):
            raise ValueError(
                f"Final diagnosis outside the planning candidate set must include supporting "
                f"evidence from the current patient: {diagnosis.icd_code}."
            )

    selected_codes = {
        diagnosis.icd_code for diagnosis in diagnosis_content.topk_diagnoses
    }
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
    return validated_excluded_candidates


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
    hypothesis_agent = build_hypothesis_preprocessing_agent(
        model,
        native_structured_output=native_structured_output,
    )
    positive_feature_agent = build_positive_feature_preprocessing_agent(
        model,
        native_structured_output=native_structured_output,
    )
    patient_information = (
        "<PATIENT_INFORMATION>\n"
        f"{case_text}\n"
        "</PATIENT_INFORMATION>"
    )
    hypothesis_prompt = _prepare_structured_prompt(
        patient_information,
        LlmHypothesesResult,
        native_structured_output=native_structured_output,
    )
    positive_feature_prompt = _prepare_structured_prompt(
        patient_information,
        PositiveFeaturesResult,
        native_structured_output=native_structured_output,
    )
    _notify_agent_started(
        progress_callback,
        "Preprocessing Agent",
        None,
    )
    hypothesis_run, positive_feature_run = await asyncio.gather(
        Runner.run(
            hypothesis_agent,
            hypothesis_prompt,
            run_config=RunConfig(model_settings=_diagnosis_model_settings(model)),
        ),
        Runner.run(
            positive_feature_agent,
            positive_feature_prompt,
            run_config=RunConfig(model_settings=_diagnosis_model_settings(model)),
        ),
    )
    llm_hypotheses_result = _parse_structured_result(
        hypothesis_run.final_output,
        LlmHypothesesResult,
    )
    try:
        positive_features_result = _parse_structured_result(
            positive_feature_run.final_output,
            PositiveFeaturesResult,
        )
    except ValueError as first_exc:
        correction_prompt = (
            f"{positive_feature_prompt}\n\n"
            "## Correction Required\n\n"
            f"The previous response failed validation: {type(first_exc).__name__}: {first_exc}\n"
            "Generate the complete five-field JSON object again. Return concise, non-duplicated "
            "patient findings without copying long passages from the case, and ensure the JSON is "
            "complete and valid."
        )
        retry_output = (
            await Runner.run(
                positive_feature_agent,
                correction_prompt,
                run_config=RunConfig(model_settings=_diagnosis_model_settings(model)),
            )
        ).final_output
        try:
            positive_features_result = _parse_structured_result(
                retry_output,
                PositiveFeaturesResult,
            )
        except ValueError as retry_exc:
            raise DiagnosisStageError(
                "positive_feature_preprocessing",
                retry_exc,
            ) from retry_exc
    result = PreprocessingResult(
        llm_hypotheses=llm_hypotheses_result.llm_hypotheses,
        positive_features=positive_features_result,
    )
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

    for similar_case in similar_case_retrieval_result.rrf:
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
        for similar_case in similar_case_retrieval_result.rrf
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
    positive_features: PositiveFeaturesResult,
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
    case_text: str,
    hypotheses: list[HypothesisItem],
    positive_features: PositiveFeaturesResult,
    *,
    model: str | Model,
    debug: bool = False,
    round_index: int | None = None,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> GuidelineSearchResult:
    native_structured_output = _uses_native_structured_output(model)
    catalog = guideline_skill_catalog()
    catalog_by_name = {item["name"]: item for item in catalog}
    available_skill_names = {item["name"] for item in catalog}

    async def run_one_skill(skill_name: str) -> GuidelineSkillResult:
        skill_agent = build_guideline_skill_executor_agent(
            GuidelineSkillResult,
            model,
            native_structured_output=native_structured_output,
        )
        skill_prompt = _prepare_structured_prompt(
            (
                "<SELECTED_SKILL_NAME>\n"
                f"{skill_name}\n"
                "</SELECTED_SKILL_NAME>\n\n"
                "<PATIENT_FEATURES>\n"
                f"{_as_json(positive_features)}\n"
                "</PATIENT_FEATURES>\n\n"
                "Load exactly the selected native skill, completely read its SKILL.md, and "
                "follow the workflow defined there. Return one result for this skill only."
            ),
            GuidelineSkillResult,
            native_structured_output=native_structured_output,
        )
        skill_run = await Runner.run(
            skill_agent,
            skill_prompt,
            max_turns=15,
            run_config=RunConfig(
                model_settings=_diagnosis_model_settings(model),
                sandbox=SandboxRunConfig(
                    client=UnixLocalSandboxClient(),
                ),
            ),
        )
        loaded_skill = any(
            getattr(item, "type", None) == "tool_call_item"
            and getattr(item, "tool_name", None) == "load_skill"
            for item in skill_run.new_items
        )
        if not loaded_skill:
            raise ValueError(
                f"Skill executor did not call load_skill for {skill_name!r}."
            )
        raw_result = skill_run.final_output
        try:
            skill_result = _parse_structured_result(raw_result, GuidelineSkillResult)
        except ValueError as first_exc:
            if native_structured_output or not str(raw_result).strip():
                raise
            repair_agent = Agent(
                name="Guideline Skill Result Repair Agent",
                model=model,
                instructions=(
                    "Repair one invalid GuidelineSkillResult JSON object without calling tools. "
                    "Preserve the selected skill name and every supported evidence statement. "
                    "Only correct JSON syntax and fields required by the schema. Return exactly "
                    "one JSON object and no other text."
                ),
            )
            repair_prompt = _prepare_structured_prompt(
                (
                    "<SELECTED_SKILL_NAME>\n"
                    f"{skill_name}\n"
                    "</SELECTED_SKILL_NAME>\n\n"
                    "<INVALID_OUTPUT>\n"
                    f"{raw_result}\n"
                    "</INVALID_OUTPUT>\n\n"
                    "<VALIDATION_ERROR>\n"
                    f"{type(first_exc).__name__}: {first_exc}\n"
                    "</VALIDATION_ERROR>"
                ),
                GuidelineSkillResult,
                native_structured_output=False,
            )
            repaired_result = (
                await Runner.run(
                    repair_agent,
                    repair_prompt,
                    max_turns=1,
                    run_config=RunConfig(
                        model_settings=_diagnosis_model_settings(model)
                    ),
                )
            ).final_output
            skill_result = _parse_structured_result(
                repaired_result,
                GuidelineSkillResult,
            )

        if skill_result.skill_name != skill_name:
            raise ValueError(
                f"Skill executor returned {skill_result.skill_name!r} instead of "
                f"the selected skill {skill_name!r}."
            )
        return skill_result

    async def run_selected_guideline_skills(
        direct_matches: list[GuidelineDirectSkillMatch],
        expanded_matches: list[GuidelineExpandedSkillMatch],
        unused_reason: str | None = None,
        matching_failures: list[str] | None = None,
    ) -> GuidelineSearchResult:
        """Run each selected native guideline skill in an isolated child agent.

        Args:
            direct_matches: Skills whose primary disease directly matches a hypothesis.
            expanded_matches: One-hop forward differential expansions from direct matches.
            unused_reason: Specific reason no skill matched; null when skills were selected.
            matching_failures: Invalid direct or expanded matches rejected before execution.
        """
        failures = list(matching_failures or [])
        selected_names = list(
            dict.fromkeys(
                [
                    *(match.skill_name for match in direct_matches),
                    *(match.skill_name for match in expanded_matches),
                ]
            )
        )
        skill_semaphore = asyncio.Semaphore(10)

        async def run_limited_skill(skill_name: str) -> GuidelineSkillResult:
            async with skill_semaphore:
                return await run_one_skill(skill_name)

        skill_outcomes = await asyncio.gather(
            *(run_limited_skill(skill_name) for skill_name in selected_names),
            return_exceptions=True,
        )
        skill_results: list[GuidelineSkillResult] = []
        for skill_name, outcome in zip(selected_names, skill_outcomes):
            if isinstance(outcome, MaxTurnsExceeded):
                failures.append(
                    f"{skill_name}: "
                    f"{_max_turns_failure_reason('Guideline skill executor', outcome)}"
                )
            elif isinstance(outcome, Exception):
                failures.append(
                    f"{skill_name}: {type(outcome).__name__}: {outcome}"
                )
            else:
                skill_results.append(outcome)

        return GuidelineSearchResult(
            used_skill=bool(skill_results),
            unused_reason=(
                None
                if skill_results
                else (
                    "All selected guideline skills failed."
                    if selected_names
                    else unused_reason or "No guideline skills were selected."
                )
            ),
            direct_matches=direct_matches,
            expanded_matches=expanded_matches,
            skill_results=skill_results,
            reason=(
                f"Guideline skill failures: {'; '.join(failures)}"
                if failures
                else None
            ),
        )

    guideline_agent = build_guideline_orchestrator_agent(
        GuidelineDirectSkillSelection,
        model,
        native_structured_output=native_structured_output,
    )
    guideline_prompt = _prepare_structured_prompt(
        (
            "<DIAGNOSTIC_HYPOTHESES>\n"
            f"{_as_json(hypotheses)}\n"
            "</DIAGNOSTIC_HYPOTHESES>\n\n"
            "<AVAILABLE_SKILL_CATALOG>\n"
            f"{_as_json(catalog)}\n"
            "</AVAILABLE_SKILL_CATALOG>"
        ),
        GuidelineDirectSkillSelection,
        native_structured_output=native_structured_output,
    )
    _notify_agent_started(progress_callback, "Guideline Searcher Agent", round_index)
    try:
        raw_selection = (
            await Runner.run(
                guideline_agent,
                guideline_prompt,
                max_turns=1,
                run_config=RunConfig(model_settings=_diagnosis_model_settings(model)),
            )
        ).final_output
        selection = _parse_structured_result(
            raw_selection,
            GuidelineDirectSkillSelection,
        )
        direct_matches: list[GuidelineDirectSkillMatch] = []
        seen_direct_skill_names: set[str] = set()
        matching_failures: list[str] = []
        for match in selection.direct_matches:
            if match.skill_name not in available_skill_names:
                matching_failures.append(
                    f"{match.skill_name}: unknown direct skill name"
                )
                continue
            if match.skill_name in seen_direct_skill_names:
                continue
            seen_direct_skill_names.add(match.skill_name)
            direct_matches.append(match)

        expanded_matches: list[GuidelineExpandedSkillMatch] = []
        seen_expanded_skill_names: set[str] = set()
        source_skills: list[dict[str, object]] = []
        differential_diseases_by_source: dict[str, list[str]] = {}
        for direct_match in direct_matches:
            source_description = catalog_by_name[direct_match.skill_name]["description"]
            _, separator, differential_text = source_description.partition(
                "明确鉴别疾病："
            )
            differential_diseases = (
                []
                if not separator or differential_text.rstrip("。") == "无"
                else differential_text.rstrip("。").split("、")
            )
            if not differential_diseases:
                continue
            differential_diseases_by_source[direct_match.skill_name] = (
                differential_diseases
            )
            source_skills.append(
                {
                    "skill_name": direct_match.skill_name,
                    "explicit_differential_diseases": differential_diseases,
                }
            )

        if source_skills:
            expansion_agent = build_guideline_expansion_agent(
                GuidelineSkillExpansionSelection,
                model,
                native_structured_output=native_structured_output,
            )
            target_catalog = []
            for item in catalog:
                if item["name"] in seen_direct_skill_names:
                    continue
                primary_scope, _, _ = item["description"].partition(
                    "。明确鉴别疾病："
                )
                target_catalog.append(
                    {
                        "name": item["name"],
                        "primary_disease_scope": primary_scope.removeprefix(
                            "主要疾病及适用范围："
                        ),
                    }
                )
            expansion_prompt = _prepare_structured_prompt(
                (
                    "<SOURCE_DIRECT_SKILLS>\n"
                    f"{_as_json(source_skills)}\n"
                    "</SOURCE_DIRECT_SKILLS>\n\n"
                    "<AVAILABLE_TARGET_SKILL_CATALOG>\n"
                    f"{_as_json(target_catalog)}\n"
                    "</AVAILABLE_TARGET_SKILL_CATALOG>"
                ),
                GuidelineSkillExpansionSelection,
                native_structured_output=native_structured_output,
            )
            try:
                raw_expansion = (
                    await Runner.run(
                        expansion_agent,
                        expansion_prompt,
                        max_turns=1,
                        run_config=RunConfig(
                            model_settings=_diagnosis_model_settings(model)
                        ),
                    )
                ).final_output
                expansion_selection = _parse_structured_result(
                    raw_expansion,
                    GuidelineSkillExpansionSelection,
                )
            except Exception as exc:
                matching_failures.append(
                    f"Differential expansion failed: {type(exc).__name__}: {exc}"
                )
            else:
                for source_match in expansion_selection.source_matches:
                    differential_diseases = differential_diseases_by_source.get(
                        source_match.source_skill_name
                    )
                    if differential_diseases is None:
                        matching_failures.append(
                            f"{source_match.source_skill_name}: unknown differential "
                            "expansion source"
                        )
                        continue
                    for differential_match in source_match.differential_matches:
                        for skill_name in differential_match.skill_names:
                            if skill_name not in available_skill_names:
                                matching_failures.append(
                                    f"{skill_name}: invalid differential target from "
                                    f"{source_match.source_skill_name}"
                                )
                                continue
                            if skill_name in seen_direct_skill_names:
                                continue
                            if skill_name in seen_expanded_skill_names:
                                continue
                            seen_expanded_skill_names.add(skill_name)
                            expanded_matches.append(
                                GuidelineExpandedSkillMatch(
                                    skill_name=skill_name,
                                    source_skill_name=(
                                        source_match.source_skill_name
                                    ),
                                    differential_disease=(
                                        differential_match.differential_disease
                                    ),
                                )
                            )

        result = await run_selected_guideline_skills(
            direct_matches,
            expanded_matches,
            selection.unused_reason,
            matching_failures,
        )
    except MaxTurnsExceeded as exc:
        result = GuidelineSearchResult(
            used_skill=False,
            unused_reason="Guideline skill matching did not complete.",
            skill_results=[],
            reason=_max_turns_failure_reason("Guideline orchestrator", exc),
        )
    except Exception as exc:
        result = GuidelineSearchResult(
            used_skill=False,
            unused_reason="Guideline skill matching failed.",
            skill_results=[],
            reason=f"Guideline orchestrator failed: {type(exc).__name__}: {exc}",
        )
    source_locator_pattern = re.compile(
        r"\s*[（(]\s*原文\s*L\d{6}(?:\s*-\s*L\d{6})?"
        r"(?:\s*[、,，]\s*L\d{6}(?:\s*-\s*L\d{6})?)*\s*[）)]"
    )
    for skill_result in result.skill_results:
        skill_result.guideline_evidence = [
            source_locator_pattern.sub("", evidence).strip()
            for evidence in skill_result.guideline_evidence
        ]

    expanded_matches_by_name = {
        match.skill_name: match for match in result.expanded_matches
    }
    expanded_guideline_results = [
        {
            "skill_name": skill_result.skill_name,
            "source_skill_name": expanded_matches_by_name[
                skill_result.skill_name
            ].source_skill_name,
            "differential_disease": expanded_matches_by_name[
                skill_result.skill_name
            ].differential_disease,
            "disease_name": skill_result.disease_name,
            "guideline_diagnosis": skill_result.guideline_diagnosis,
            "guideline_evidence": skill_result.guideline_evidence,
        }
        for skill_result in result.skill_results
        if skill_result.skill_name in expanded_matches_by_name
    ]
    expanded_skill_names_before_filter = [
        str(item["skill_name"])
        for item in expanded_guideline_results
    ]
    if expanded_guideline_results:
        filter_agent = build_guideline_result_filter_agent(
            GuidelineExpandedResultSelection,
            model,
            native_structured_output=native_structured_output,
        )
        filter_prompt = _prepare_structured_prompt(
            (
                "<PATIENT_INFORMATION>\n"
                f"{case_text}\n"
                "</PATIENT_INFORMATION>\n\n"
                "<DIAGNOSTIC_HYPOTHESES>\n"
                f"{_as_json(hypotheses)}\n"
                "</DIAGNOSTIC_HYPOTHESES>\n\n"
                "<EXPANDED_GUIDELINE_RESULTS>\n"
                f"{_as_json(expanded_guideline_results)}\n"
                "</EXPANDED_GUIDELINE_RESULTS>\n\n"
                "## Task\n\n"
                "Return the exact skill names of the expanded guideline results that should be "
                "retained for final diagnosis."
            ),
            GuidelineExpandedResultSelection,
            native_structured_output=native_structured_output,
        )
        _notify_agent_started(
            progress_callback,
            "Guideline Result Filter Agent",
            round_index,
        )
        try:
            raw_filter_selection = (
                await Runner.run(
                    filter_agent,
                    filter_prompt,
                    max_turns=1,
                    run_config=RunConfig(
                        model_settings=_diagnosis_model_settings(model)
                    ),
                )
            ).final_output
            filter_selection = _parse_structured_result(
                raw_filter_selection,
                GuidelineExpandedResultSelection,
            )
            available_expanded_names = {
                item["skill_name"] for item in expanded_guideline_results
            }
            selected_expanded_names = (
                filter_selection.selected_expanded_skill_names
            )
            if len(selected_expanded_names) != len(set(selected_expanded_names)):
                raise ValueError(
                    "Guideline result filter returned duplicate expanded skill names."
                )
            unknown_skill_names = (
                set(selected_expanded_names) - available_expanded_names
            )
            if unknown_skill_names:
                raise ValueError(
                    "Guideline result filter returned unknown expanded skill names: "
                    f"{sorted(unknown_skill_names)}."
                )
        except Exception as exc:
            filter_error = f"{type(exc).__name__}: {exc}"
            filter_failure = (
                f"Guideline result filtering failed; all completed expanded skill results "
                f"were retained: {filter_error}"
            )
            result.filter_result = GuidelineResultFilterResult(
                status="failed",
                expanded_skill_names_before_filter=(
                    expanded_skill_names_before_filter
                ),
                retained_expanded_skill_names=(
                    expanded_skill_names_before_filter
                ),
                filtered_out_expanded_skill_names=[],
                reason=filter_error,
            )
            result.reason = (
                f"{result.reason}; {filter_failure}"
                if result.reason
                else filter_failure
            )
        else:
            selected_expanded_name_set = set(selected_expanded_names)
            retained_expanded_skill_names = [
                skill_name
                for skill_name in expanded_skill_names_before_filter
                if skill_name in selected_expanded_name_set
            ]
            filtered_out_expanded_skill_names = [
                skill_name
                for skill_name in expanded_skill_names_before_filter
                if skill_name not in selected_expanded_name_set
            ]
            result.filter_result = GuidelineResultFilterResult(
                status="completed",
                expanded_skill_names_before_filter=(
                    expanded_skill_names_before_filter
                ),
                retained_expanded_skill_names=(
                    retained_expanded_skill_names
                ),
                filtered_out_expanded_skill_names=(
                    filtered_out_expanded_skill_names
                ),
                reason=None,
            )
            result.expanded_matches = [
                match
                for match in result.expanded_matches
                if match.skill_name in selected_expanded_name_set
            ]
            result.skill_results = [
                skill_result
                for skill_result in result.skill_results
                if (
                    skill_result.skill_name not in available_expanded_names
                    or skill_result.skill_name in selected_expanded_name_set
                )
            ]
            result.used_skill = bool(result.skill_results)
            if not result.used_skill:
                result.unused_reason = (
                    "No completed guideline results passed final relevance filtering."
                )
    else:
        result.filter_result = GuidelineResultFilterResult(
            status="not_triggered",
            expanded_skill_names_before_filter=[],
            retained_expanded_skill_names=[],
            filtered_out_expanded_skill_names=[],
            reason=(
                "No completed expanded guideline results were available for filtering."
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
    llm_hypotheses_result: LlmHypothesesResult,
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
    llm_hypothesis_codes = {
        hypothesis.icd_code
        for hypothesis in llm_hypotheses_result.llm_hypotheses
    }
    similar_case_ranks = {
        similar_case.icd_code.strip().upper().replace(".", ""): rank
        for rank, similar_case in enumerate(
            similar_case_retrieval_result.rrf,
            start=1,
        )
    }
    candidate_diagnoses = []
    candidate_names: dict[str, str] = {}
    for hypothesis in search_planning_result.hypotheses:
        if hypothesis.icd_code in candidate_names:
            continue
        candidate_names[hypothesis.icd_code] = hypothesis.category_name
        sources: list[str] = []
        if hypothesis.icd_code in llm_hypothesis_codes:
            sources.append("initial_llm")
        if hypothesis.icd_code in similar_case_ranks:
            sources.append("similar_case_rrf")
        candidate: dict[str, object] = {
            "icd_code": hypothesis.icd_code,
            "category_name": hypothesis.category_name,
            "sources": sources,
        }
        if hypothesis.icd_code in similar_case_ranks:
            candidate["similar_case_rank"] = similar_case_ranks[hypothesis.icd_code]
        candidate_diagnoses.append(candidate)
    planning_candidates = {
        hypothesis.icd_code: hypothesis.category_name
        for hypothesis in search_planning_result.hypotheses
    }
    direct_skill_names = {
        match.skill_name for match in guideline_search_result.direct_matches
    }
    expanded_matches = {
        match.skill_name: match
        for match in guideline_search_result.expanded_matches
    }
    numbered_evidence: list[str] = []
    guideline_assessments: list[dict[str, object]] = []
    for skill_result in guideline_search_result.skill_results:
        skill_evidence: list[str] = []
        for evidence in skill_result.guideline_evidence:
            numbered_item = (
                f"[{len(numbered_evidence) + 1}] "
                f"{skill_result.skill_name}：{evidence}"
            )
            numbered_evidence.append(numbered_item)
            skill_evidence.append(numbered_item)
        assessment: dict[str, object] = {
            "skill_name": skill_result.skill_name,
            "disease_name": skill_result.disease_name,
            "match_type": (
                "direct"
                if skill_result.skill_name in direct_skill_names
                else "expanded"
            ),
            "guideline_diagnosis": skill_result.guideline_diagnosis,
            "guideline_evidence": skill_evidence,
        }
        expanded_match = expanded_matches.get(skill_result.skill_name)
        if expanded_match is not None:
            assessment["source_skill_name"] = expanded_match.source_skill_name
            assessment["differential_disease"] = (
                expanded_match.differential_disease
            )
        guideline_assessments.append(assessment)

    literature_evidence: list[str] = []
    for pubmed_result in _format_pubmed_results(knowledge_search_result):
        numbered_item = f"[{len(numbered_evidence) + 1}] {pubmed_result}"
        numbered_evidence.append(numbered_item)
        literature_evidence.append(numbered_item)
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
        "<GUIDELINE_ASSESSMENTS>\n"
        f"{_as_json(guideline_assessments)}\n"
        "</GUIDELINE_ASSESSMENTS>\n\n"
        "<LITERATURE_EVIDENCE>\n"
        f"{_as_json(literature_evidence)}\n"
        "</LITERATURE_EVIDENCE>\n\n"
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
    validation_error: ValueError | None = None
    for attempt in range(2):
        current_prompt = diagnosis_prompt
        if validation_error is not None:
            current_prompt = (
                f"{diagnosis_prompt}\n\n"
                "## Correction Required\n\n"
                f"The previous response failed validation: {type(validation_error).__name__}: "
                f"{validation_error}\n"
                "Generate the complete final diagnosis JSON again and correct that exact error. "
                "Return exactly five unique ICD-10-CM codes and continue to follow every candidate, "
                "citation, exclusion, and ranking requirement above."
            )
        raw_result = (
            await Runner.run(
                diagnosis_agent,
                current_prompt,
                run_config=RunConfig(model_settings=_diagnosis_model_settings(model)),
            )
        ).final_output
        try:
            diagnosis_content = _parse_structured_result(
                raw_result,
                FinalDiagnosisContent,
            )
            validated_excluded_candidates = _validate_final_diagnosis_content(
                diagnosis_content,
                candidate_names,
                planning_candidates,
            )
            break
        except ValueError as exc:
            if attempt == 1:
                raise DiagnosisStageError("final_diagnosis", exc) from exc
            validation_error = exc

    citation_pattern = re.compile(r"\[(\d+)\]")

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
    positive_features_result = preprocessing_result.positive_features
    try:
        similar_case_retrieval_result = await asyncio.to_thread(
            _run_similar_case_retrieval,
            positive_features_result,
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
                case_text,
                search_planning_result.hypotheses,
                positive_features_result,
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
            llm_hypotheses_result,
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
                llm_hypotheses_result,
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
                knowledge_search_result=knowledge_search_result,
                guideline_search_result=guideline_search_result,
                diagnosis_result=diagnosis_result,
                diagnostic_judgement_result=diagnostic_judgement_result,
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
