from __future__ import annotations

import argparse
import json
import sys

from agents import RunConfig, Runner
from agents.sandbox import SandboxRunConfig
from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

from config import DIAGNOSIS_TOPK
from diagnosis.agents.digestive_diagnosis_agent import (
    SKILLS_DIR,
    build_digestive_diagnosis_agent,
)
from diagnosis.agents.diagnostic_judgement_agent import build_diagnostic_judgement_agent
from diagnosis.agents.knowledge_searcher_agent import build_knowledge_searcher_agent
from diagnosis.agents.search_planning_agent import build_search_planning_agent
from diagnosis.agents.similar_case_retrieval_agent import (
    build_similar_case_retrieval_agent,
    build_similar_case_retrieval_prompt,
)
from schemas import (
    DiagnosisResult,
    DiagnosticJudgementResult,
    SearchPlanningResult,
    SimilarCaseRetrievalResult,
)


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


def _print_debug_section(title: str, model_object: object) -> None:
    print(f"\n===== {title} =====", file=sys.stderr)
    print(_as_json(model_object), file=sys.stderr)


def _run_search_planning(
    case_text: str,
    *,
    previous_search_planning_result: SearchPlanningResult | None = None,
    previous_diagnosis_result: DiagnosisResult | None = None,
    diagnostic_judgement_result: DiagnosticJudgementResult | None = None,
    debug: bool = False,
    round_index: int | None = None,
) -> SearchPlanningResult:
    search_planning_agent = build_search_planning_agent()
    search_planning_prompt = (
        f"Patient information:\n{case_text}\n\n"
        "Write every output field in English."
    )
    if previous_search_planning_result and previous_diagnosis_result and diagnostic_judgement_result:
        search_planning_prompt = (
            f"Patient information:\n{case_text}\n\n"
            f"Previous search planning result:\n{_as_json(previous_search_planning_result)}\n\n"
            f"Previous diagnosis result:\n{_as_json(previous_diagnosis_result)}\n\n"
            f"Diagnostic judgement result:\n{_as_json(diagnostic_judgement_result)}\n\n"
            "The diagnostic judgement found that hypotheses were closer to the patient information "
            "than the previous topk_diagnoses. Regenerate improved search_queries for the next "
            "diagnosis round while using only information present in the patient record. "
            "Write every output field in English."
        )

    result = Runner.run_sync(
        search_planning_agent,
        search_planning_prompt,
    ).final_output
    if debug:
        _print_debug_section(f"Search Planning Result - Round {round_index}", result)
    return result


def _run_knowledge_search(
    case_text: str,
    search_planning_result: SearchPlanningResult,
    *,
    debug: bool = False,
    round_index: int | None = None,
) -> object:
    knowledge_agent = build_knowledge_searcher_agent()
    knowledge_prompt = (
        f"Case information:\n{case_text}\n\n"
        f"Search queries:\n{_as_json(search_planning_result.search_queries)}\n\n"
        "Write every output field in English."
    )
    result = Runner.run_sync(knowledge_agent, knowledge_prompt).final_output
    if debug:
        _print_debug_section(f"Knowledge Search Result - Round {round_index}", result)
    return result


def _run_similar_case_retrieval(
    search_planning_result: SearchPlanningResult,
    *,
    debug: bool = False,
    round_index: int | None = None,
) -> SimilarCaseRetrievalResult:
    similar_case_agent = build_similar_case_retrieval_agent()
    similar_case_prompt = build_similar_case_retrieval_prompt(search_planning_result.search_queries)
    result = Runner.run_sync(similar_case_agent, similar_case_prompt).final_output
    if debug:
        _print_debug_section(f"Similar Case Retrieval Result - Round {round_index}", result)
    return result


def _run_final_diagnosis(
    case_text: str,
    search_planning_result: SearchPlanningResult,
    knowledge_search_result: object,
    similar_case_retrieval_result: SimilarCaseRetrievalResult | None = None,
    *,
    debug: bool = False,
    round_index: int | None = None,
) -> DiagnosisResult:
    diagnosis_agent = build_digestive_diagnosis_agent(
        DiagnosisResult,
        phase="final_diagnosis",
    )
    diagnosis_prompt = (
        f"Case information:\n{case_text}\n\n"
        f"Search planning result:\n{_as_json(search_planning_result)}\n\n"
        f"Knowledge search result:\n{_as_json(knowledge_search_result)}\n\n"
        f"Similar case retrieval result:\n{_as_json(similar_case_retrieval_result)}\n\n"
        f"Available skills directory:\n{SKILLS_DIR}\n\n"
        f"Please output the top {DIAGNOSIS_TOPK} suspected diagnoses. "
        "Write every output field in English."
    )
    run_config = RunConfig(
        sandbox=SandboxRunConfig(
            client=UnixLocalSandboxClient(),
        ),
    )
    result = Runner.run_sync(
        diagnosis_agent,
        diagnosis_prompt,
        run_config=run_config,
    ).final_output
    if debug:
        _print_debug_section(f"Final Diagnosis Result - Round {round_index}", result)
    return result


def _run_diagnostic_judgement(
    case_text: str,
    search_planning_result: SearchPlanningResult,
    diagnosis_result: DiagnosisResult,
    *,
    debug: bool = False,
    round_index: int | None = None,
) -> DiagnosticJudgementResult:
    diagnostic_judgement_agent = build_diagnostic_judgement_agent()
    diagnostic_judgement_prompt = (
        f"Patient information:\n{case_text}\n\n"
        f"Problem representation:\n{search_planning_result.problem_representation}\n\n"
        f"Hypotheses from search planning:\n{_as_json(search_planning_result.hypotheses)}\n\n"
        f"Top-K diagnoses from diagnosis stage:\n{_as_json(diagnosis_result.topk_diagnoses)}\n\n"
        "Judge whether topk_diagnoses or hypotheses is closer to the patient information. "
        "Write every output field in English."
    )
    result = Runner.run_sync(
        diagnostic_judgement_agent,
        diagnostic_judgement_prompt,
    ).final_output
    if debug:
        _print_debug_section(f"Diagnostic Judgement Result - Round {round_index}", result)
    return result


def make_diagnosis(case_text: str, *, debug: bool = False) -> DiagnosisResult:
    max_diagnosis_rounds = 2
    search_planning_result = _run_search_planning(case_text, debug=debug, round_index=1)

    for round_index in range(1, max_diagnosis_rounds + 1):
        knowledge_search_result = _run_knowledge_search(
            case_text,
            search_planning_result,
            debug=debug,
            round_index=round_index,
        )

        # similar_case_retrieval_result = _run_similar_case_retrieval(
        #     search_planning_result,
        #     debug=debug,
        #     round_index=round_index,
        # )

        diagnosis_result = _run_final_diagnosis(
            case_text,
            search_planning_result,
            knowledge_search_result,
            # similar_case_retrieval_result,
            debug=debug,
            round_index=round_index,
        )

        diagnostic_judgement_result = _run_diagnostic_judgement(
            case_text,
            search_planning_result,
            diagnosis_result,
            debug=debug,
            round_index=round_index,
        )

        if diagnostic_judgement_result.should_stop or round_index == max_diagnosis_rounds:
            return diagnosis_result

        search_planning_result = _run_search_planning(
            case_text,
            previous_search_planning_result=search_planning_result,
            previous_diagnosis_result=diagnosis_result,
            diagnostic_judgement_result=diagnostic_judgement_result,
            debug=debug,
            round_index=round_index + 1,
        )

    raise RuntimeError("Diagnosis loop ended without producing a diagnosis result.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gastroenterology medical diagnosis Agent demo")
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

    result = make_diagnosis(case_text, debug=args.debug)
    # print(_as_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
