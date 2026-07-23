from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable
from typing import TypeVar

from agents import (
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
    DIAGNOSIS_TOPK,
    OPENAI_MODEL,
)
from diagnosis.agents.digestive_diagnosis_agent import build_digestive_diagnosis_agent
from diagnosis.agents.diagnostic_judgement_agent import build_diagnostic_judgement_agent
from diagnosis.agents.guideline_searcher_agent import SKILLS_DIR, build_guideline_searcher_agent
from diagnosis.agents.knowledge_searcher_agent import (
    build_knowledge_searcher_agent,
    search_pubmed_queries,
)
from diagnosis.agents.search_planning_agent import build_search_planning_agent
from diagnosis.agents.similar_case_retrieval_agent import retrieve_similar_cases
from schemas import (
    DiagnosisPipelineResult,
    DiagnosisResult,
    DiagnosticJudgementResult,
    GuidelineSearchResult,
    KnowledgeSearchResult,
    SearchPlanningResult,
    SimilarCaseRetrievalResult,
)


DiagnosisProgressCallback = Callable[[str, str, str | None], None]
StructuredResultT = TypeVar("StructuredResultT", bound=BaseModel)


def _read_case_text(args: argparse.Namespace) -> str:
    if args.case:
        return args.case.strip()

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    print("Enter the case information, then press Ctrl+D when finished:")
    return sys.stdin.read().strip()


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


def _deepseek_model_settings(model: str | Model) -> ModelSettings | None:
    if not isinstance(model, OpenAIChatCompletionsModel):
        return None
    thinking_type = "enabled" if DEEPSEEK_THINKING else "disabled"
    return ModelSettings(extra_body={"thinking": {"type": thinking_type}})


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
        "Return only one valid JSON object matching this JSON Schema. "
        "Do not wrap the JSON in Markdown fences or add explanatory text:\n"
        f"{json.dumps(output_type.model_json_schema(), ensure_ascii=False)}"
    )


def _parse_structured_result(
    result: object,
    output_type: type[StructuredResultT],
) -> StructuredResultT:
    if isinstance(result, output_type):
        return result
    stripped = str(result).strip()
    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        stripped,
        flags=re.DOTALL,
    )
    if fenced_match:
        stripped = fenced_match.group(1).strip()
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output.")
        stripped = stripped[start : end + 1]
    return output_type.model_validate_json(stripped)


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


def _run_search_planning(
    case_text: str,
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
    search_planning_prompt = (
        f"Patient information:\n{case_text}\n\n"
    )
    if previous_search_planning_result and previous_diagnosis_result and diagnostic_judgement_result:
        search_planning_prompt = (
            f"Patient information:\n{case_text}\n\n"
            f"Previous search planning result:\n{_as_json(previous_search_planning_result)}\n\n"
            f"Previous diagnosis result:\n{_as_json(previous_diagnosis_result)}\n\n"
            f"Diagnostic judgement result:\n{_as_json(diagnostic_judgement_result)}\n\n"
            f"Previous guideline evidence:\n{_as_json(previous_guideline_evidence or [])}\n\n"
            "The diagnostic judgement found that hypotheses were closer to the patient information "
            "than the previous topk_diagnoses. Regenerate improved search_queries for the next "
            "diagnosis round. Return the complete SearchPlanningResult required by the schema, but "
            "preserve the previous hypotheses and similar_case_queries unless they violate the agent "
            "instructions. Use the previous artifacts, including previous guideline evidence, only to "
            "improve the retrieval strategy, and do not treat their contents as new patient facts. "
        )

    search_planning_prompt = _prepare_structured_prompt(
        search_planning_prompt,
        SearchPlanningResult,
        native_structured_output=native_structured_output,
    )

    _notify_agent_started(progress_callback, "Search Planning Agent", round_index)
    raw_result = Runner.run_sync(
        search_planning_agent,
        search_planning_prompt,
        run_config=RunConfig(model_settings=_deepseek_model_settings(model)),
    ).final_output
    result = _parse_structured_result(raw_result, SearchPlanningResult)
    _publish_stage_result(
        f"Search Planning Result - Round {round_index}",
        result,
        debug=debug,
        progress_callback=progress_callback,
    )
    return result


def _run_knowledge_search(
    search_queries: list[str],
    *,
    model: str | Model,
    debug: bool = False,
    round_index: int | None = None,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> KnowledgeSearchResult:
    selected_queries = search_queries[:3]
    pubmed_results = search_pubmed_queries(selected_queries)
    native_structured_output = _uses_native_structured_output(model)
    knowledge_agent = build_knowledge_searcher_agent(
        model,
        native_structured_output=native_structured_output,
    )
    knowledge_prompt = (
        f"Search queries:\n{_as_json(selected_queries)}\n\n"
        f"PubMed search results:\n{_as_json(pubmed_results)}\n\n"
        "Keep search queries, publication titles, URLs, and quoted source text in their original language."
    )
    knowledge_prompt = _prepare_structured_prompt(
        knowledge_prompt,
        KnowledgeSearchResult,
        native_structured_output=native_structured_output,
    )
    _notify_agent_started(progress_callback, "Knowledge Searcher Agent", round_index)
    raw_result = Runner.run_sync(
        knowledge_agent,
        knowledge_prompt,
        run_config=RunConfig(model_settings=_deepseek_model_settings(model)),
    ).final_output
    result = _parse_structured_result(raw_result, KnowledgeSearchResult)
    _publish_stage_result(
        f"Knowledge Search Result - Round {round_index}",
        result,
        debug=debug,
        progress_callback=progress_callback,
    )
    return result


def _format_pubmed_evidence(
    knowledge_search_result: KnowledgeSearchResult,
) -> list[str]:
    return [
        f"PubMed PMID {item.pmid}（{item.title}）：{item.evidence}"
        for item in knowledge_search_result.pubmed_evidence
    ]


def _run_similar_case_retrieval(
    similar_case_queries: list[str],
    *,
    debug: bool = False,
    round_index: int | None = None,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> SimilarCaseRetrievalResult:
    _notify_agent_started(progress_callback, "Similar Case Retrieval Agent", round_index)
    ranking_details: list[dict[str, object]] = []
    result = retrieve_similar_cases(
        similar_case_queries,
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


def _run_guideline_search(
    search_queries: list[str],
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
        f"Search queries:\n{_as_json(search_queries)}\n\n"
        f"Available skills directory:\n{SKILLS_DIR}\n\n"
        "Search the local guideline skills for clinically relevant guideline evidence. "
        "Keep skill_names unchanged."
    )
    guideline_prompt = _prepare_structured_prompt(
        guideline_prompt,
        GuidelineSearchResult,
        native_structured_output=native_structured_output,
    )
    _notify_agent_started(progress_callback, "Guideline Searcher Agent", round_index)
    if native_structured_output:
        raw_result = Runner.run_sync(
            guideline_agent,
            guideline_prompt,
            run_config=RunConfig(
                model_settings=_deepseek_model_settings(model),
                sandbox=SandboxRunConfig(
                    client=UnixLocalSandboxClient(),
                ),
            ),
        ).final_output
    else:
        raw_result = Runner.run_sync(
            guideline_agent,
            guideline_prompt,
            max_turns=100,
            run_config=RunConfig(model_settings=_deepseek_model_settings(model)),
        ).final_output
    result = _parse_structured_result(raw_result, GuidelineSearchResult)
    _publish_stage_result(
        f"Guideline Search Result - Round {round_index}",
        result,
        debug=debug,
        progress_callback=progress_callback,
    )
    return result


def _run_final_diagnosis(
    case_text: str,
    knowledge_search_result: KnowledgeSearchResult,
    guideline_evidence: list[str],
    similar_case_retrieval_result: SimilarCaseRetrievalResult,
    *,
    model: str | Model,
    debug: bool = False,
    round_index: int | None = None,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> DiagnosisResult:
    native_structured_output = _uses_native_structured_output(model)
    diagnosis_agent = build_digestive_diagnosis_agent(
        DiagnosisResult,
        phase="final_diagnosis",
        model=model,
        native_structured_output=native_structured_output,
    )
    similar_case_diagnosis_evidence = {
        "discharge_disease": similar_case_retrieval_result.discharge_disease,
        "Sections": similar_case_retrieval_result.Sections,
    }
    pubmed_evidence = _format_pubmed_evidence(knowledge_search_result)
    combined_evidence = [
        *guideline_evidence,
        *pubmed_evidence,
    ]
    numbered_evidence = [
        f"[{index}] {evidence}"
        for index, evidence in enumerate(combined_evidence, start=1)
    ]
    diagnosis_prompt = (
        f"Case information:\n{case_text}\n\n"
        f"Knowledge search result:\n{_as_json(knowledge_search_result)}\n\n"
        f"Guideline evidence:\n{_as_json(guideline_evidence)}\n\n"
        f"Formatted PubMed evidence:\n{_as_json(pubmed_evidence)}\n\n"
        f"Numbered evidence:\n{_as_json(numbered_evidence)}\n\n"
        f"Similar case retrieval result:\n{_as_json(similar_case_diagnosis_evidence)}\n\n"
        "Set used_skill to whether guideline evidence is non-empty. Derive skill_names from the "
        "skill-name prefix before the full-width Chinese colon in each guideline evidence item. "
        "Copy the complete numbered evidence list into evidence exactly as provided. "
        f"Please output the top {DIAGNOSIS_TOPK} suspected diagnoses."
    )
    diagnosis_prompt = _prepare_structured_prompt(
        diagnosis_prompt,
        DiagnosisResult,
        native_structured_output=native_structured_output,
    )
    _notify_agent_started(progress_callback, "Digestive Diagnosis Agent", round_index)
    raw_result = Runner.run_sync(
        diagnosis_agent,
        diagnosis_prompt,
        run_config=RunConfig(model_settings=_deepseek_model_settings(model)),
    ).final_output
    result = _parse_structured_result(raw_result, DiagnosisResult)
    result = result.model_copy(update={"evidence": numbered_evidence})
    _publish_stage_result(
        f"Final Diagnosis Result - Round {round_index}",
        result,
        debug=debug,
        progress_callback=progress_callback,
    )
    return result


def _run_diagnostic_judgement(
    case_text: str,
    hypotheses: list[str],
    diagnosis_result: DiagnosisResult,
    knowledge_search_result: KnowledgeSearchResult,
    similar_case_retrieval_result: SimilarCaseRetrievalResult,
    guideline_evidence: list[str],
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
    diagnostic_judgement_prompt = (
        f"Patient information:\n{case_text}\n\n"
        f"Hypotheses from search planning:\n{_as_json(hypotheses)}\n\n"
        f"Top-K diagnoses from diagnosis stage:\n{_as_json(diagnosis_result.topk_diagnoses)}\n\n"
        f"Knowledge search result:\n{_as_json(knowledge_search_result)}\n\n"
        f"Similar case retrieval result:\n{_as_json(similar_case_retrieval_result)}\n\n"
        f"Guideline evidence:\n{_as_json(guideline_evidence)}\n\n"
        "Judge whether topk_diagnoses or hypotheses is closer to the patient information. "
        "Keep closer_result as the required enum value."
    )
    diagnostic_judgement_prompt = _prepare_structured_prompt(
        diagnostic_judgement_prompt,
        DiagnosticJudgementResult,
        native_structured_output=native_structured_output,
    )
    _notify_agent_started(progress_callback, "Diagnostic Judgement Agent", round_index)
    raw_result = Runner.run_sync(
        diagnostic_judgement_agent,
        diagnostic_judgement_prompt,
        run_config=RunConfig(model_settings=_deepseek_model_settings(model)),
    ).final_output
    result = _parse_structured_result(raw_result, DiagnosticJudgementResult)
    _publish_stage_result(
        f"Diagnostic Judgement Result - Round {round_index}",
        result,
        debug=debug,
        progress_callback=progress_callback,
    )
    return result


def make_diagnosis_pipeline(
    case_text: str,
    *,
    model: str | Model | None = None,
    debug: bool = False,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> DiagnosisPipelineResult:
    diagnosis_model = model or OPENAI_MODEL
    max_diagnosis_rounds = 2
    search_planning_result = _run_search_planning(
        case_text,
        model=diagnosis_model,
        debug=debug,
        round_index=1,
        progress_callback=progress_callback,
    )

    for round_index in range(1, max_diagnosis_rounds + 1):
        knowledge_search_result = _run_knowledge_search(
            search_planning_result.search_queries,
            model=diagnosis_model,
            debug=debug,
            round_index=round_index,
            progress_callback=progress_callback,
        )

        similar_case_retrieval_result = _run_similar_case_retrieval(
            search_planning_result.similar_case_queries,
            debug=debug,
            round_index=round_index,
            progress_callback=progress_callback,
        )

        guideline_search_result = _run_guideline_search(
            search_planning_result.search_queries,
            model=diagnosis_model,
            debug=debug,
            round_index=round_index,
            progress_callback=progress_callback,
        )

        diagnosis_result = _run_final_diagnosis(
            case_text,
            knowledge_search_result,
            guideline_search_result.guideline_evidence,
            similar_case_retrieval_result,
            model=diagnosis_model,
            debug=debug,
            round_index=round_index,
            progress_callback=progress_callback,
        )

        diagnostic_judgement_result = _run_diagnostic_judgement(
            case_text,
            search_planning_result.hypotheses,
            diagnosis_result,
            knowledge_search_result,
            similar_case_retrieval_result,
            guideline_search_result.guideline_evidence,
            model=diagnosis_model,
            debug=debug,
            round_index=round_index,
            progress_callback=progress_callback,
        )

        if (
            diagnostic_judgement_result.closer_result == "topk_diagnoses"
            or round_index == max_diagnosis_rounds
        ):
            return DiagnosisPipelineResult(
                search_planning_result=search_planning_result,
                similar_case_retrieval_result=similar_case_retrieval_result,
                diagnosis_result=diagnosis_result,
            )

        search_planning_result = _run_search_planning(
            case_text,
            model=diagnosis_model,
            previous_search_planning_result=search_planning_result,
            previous_diagnosis_result=diagnosis_result,
            diagnostic_judgement_result=diagnostic_judgement_result,
            previous_guideline_evidence=guideline_search_result.guideline_evidence,
            debug=debug,
            round_index=round_index + 1,
            progress_callback=progress_callback,
        )

    raise RuntimeError("Diagnosis loop ended without producing a diagnosis result.")


def make_diagnosis(
    case_text: str,
    *,
    model: str | Model | None = None,
    debug: bool = False,
    progress_callback: DiagnosisProgressCallback | None = None,
) -> DiagnosisResult:
    return make_diagnosis_pipeline(
        case_text,
        model=model,
        debug=debug,
        progress_callback=progress_callback,
    ).diagnosis_result


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


def _configure_cli_model(args: argparse.Namespace) -> Model:
    return build_diagnosis_model(
        args.model,
        openai_api_key=args.openai_apikey or "",
        openai_model=args.openai_model or "",
        deepseek_api_key=args.deepseek_apikey or "",
        deepseek_model=args.deepseek_model or "",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Gastroenterology medical diagnosis Agent demo")
    parser.add_argument(
        "--model",
        choices=("openai", "deepseek"),
        default="openai",
        help="LLM provider. Default: openai.",
    )
    parser.add_argument("--openai_apikey")
    parser.add_argument("--openai_model")
    parser.add_argument("--deepseek_apikey")
    parser.add_argument("--deepseek_model")
    parser.add_argument("--case", help="Case text. If omitted, input is read from standard input.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print search planning, retrieval, diagnosis, and judgement details.",
    )
    args = parser.parse_args()

    case_text = _read_case_text(args)
    if not case_text:
        print("Error: case information cannot be empty.", file=sys.stderr)
        return 1

    try:
        diagnosis_model = _configure_cli_model(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    result = make_diagnosis(case_text, model=diagnosis_model, debug=args.debug)
    # print(_as_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
