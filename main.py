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
    TRIAGE_INSTRUCTIONS
)
from diagnosis.agents.knowledge_searcher_agent import build_knowledge_searcher_agent
from diagnosis.agents.phenotype_extraction_agent import build_phenotype_extraction_agent
from diagnosis.agents.search_planning_agent import build_search_planning_agent
from schemas import DiagnosisResult, PhenotypeExtractionResult, TriageResult


def _read_case_text(args: argparse.Namespace) -> str:
    if args.case:
        return args.case.strip()

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    print("Enter the case information, then press Ctrl+D when finished:")
    return sys.stdin.read().strip()


def _as_json(model_object: object) -> str:
    if hasattr(model_object, "model_dump"):
        return json.dumps(model_object.model_dump(), ensure_ascii=False, indent=2)
    return json.dumps(model_object, ensure_ascii=False, indent=2)


def _print_debug_section(title: str, model_object: object) -> None:
    print(f"\n===== {title} =====", file=sys.stderr)
    print(_as_json(model_object), file=sys.stderr)


def make_diagnosis(case_text: str, *, debug: bool = False) -> DiagnosisResult:
    # 1.1 Phenotype extraction stage:
    # phenotype_agent = build_phenotype_extraction_agent()
    # phenotype_prompt = f"Patient information:\n{case_text}"
    # phenotype_result: PhenotypeExtractionResult = Runner.run_sync(
    #     phenotype_agent,
    #     phenotype_prompt,
    # ).final_output
    # if debug:
    #     _print_debug_section("Phenotype Extraction Result", phenotype_result)

    # 1.2 Search planning stage:
    search_planning_agent = build_search_planning_agent()
    search_planning_prompt = f"Patient information:\n{case_text}"
    search_planning_result = Runner.run_sync(
        search_planning_agent,
        search_planning_prompt,
    ).final_output
    if debug:
        _print_debug_section("Search Planning Result", search_planning_result)

    # 2. Knowledge retrieval stage:
    knowledge_agent = build_knowledge_searcher_agent()
    knowledge_prompt = (
        f"Case information:\n{case_text}\n\n"
        f"Search planning result:\n{_as_json(search_planning_result)}"
    )
    knowledge_search_result = Runner.run_sync(knowledge_agent, knowledge_prompt).final_output
    if debug:
        _print_debug_section("Knowledge Search Result", knowledge_search_result)

    # 3. Triage stage:
    triage_agent = build_digestive_diagnosis_agent(TriageResult, phase="triage")
    triage_prompt = (
        # f"Case information:\n{case_text}\n\n"
        f"Search planning result:\n{_as_json(search_planning_result)}\n\n"
        # f"Knowledge search result:\n{_as_json(knowledge_search_result)}\n\n"
        f"{TRIAGE_INSTRUCTIONS}"
    )
    triage_result = Runner.run_sync(triage_agent, triage_prompt).final_output
    if debug:
        _print_debug_section("Triage Result", triage_result)

    # 4. Final diagnosis stage:
    diagnosis_agent = build_digestive_diagnosis_agent(
        DiagnosisResult,
        phase="final_diagnosis",
    )
    diagnosis_prompt = (
        # f"Case information:\n{case_text}\n\n"
        f"Search planning result:\n{_as_json(search_planning_result)}\n\n"
        f"Triage result:\n{_as_json(triage_result)}\n\n"
        f"Available skills directory:\n{SKILLS_DIR}\n\n"
        f"Please output the top {DIAGNOSIS_TOPK} suspected diagnoses."
    )
    run_config = None
    if True:
        run_config = RunConfig(
            sandbox=SandboxRunConfig(
                client=UnixLocalSandboxClient(),
            ),
        )
    diagnosis_result = Runner.run_sync(
        diagnosis_agent,
        diagnosis_prompt,
        run_config=run_config,
    ).final_output
    if debug:
        _print_debug_section("Final Diagnosis Result", diagnosis_result)
    return diagnosis_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Gastroenterology medical diagnosis Agent demo")
    parser.add_argument("--case", help="Case text. If omitted, input is read from standard input.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print intermediate phenotype extraction, knowledge retrieval, triage, and skill matching details.",
    )
    args = parser.parse_args()

    case_text = _read_case_text(args)
    if not case_text:
        print("Error: case information cannot be empty.", file=sys.stderr)
        return 1

    result = make_diagnosis(case_text, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
