from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from diagnosis.agents import similar_case_retrieval_agent as retrieval
from schemas import PositiveFeaturesResult


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "output"
    / "batch"
    / "sample5_test_nobhc_75_20260813_111639_669877.jsonl"
)
OUTPUT_DIR = PROJECT_ROOT / "output" / "similar_case"
METHODS = ("bm25", "embedding", "rrf")
METRIC_PREFIX_LENGTHS = {
    "icd3": 3,
    "icd4": 4,
    "exact": None,
}
CUTOFFS = (1, 3, 5)


def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed_value


def _dense_max_length(value: str) -> int:
    parsed_value = int(value)
    if not 512 <= parsed_value <= 8192:
        raise argparse.ArgumentTypeError("must be between 512 and 8192")
    return parsed_value


def _non_negative_float(value: str) -> float:
    parsed_value = float(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed_value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run similar-case retrieval against fixed positive_features_result "
            "values from an existing diagnosis batch."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--experiments",
        type=Path,
        help="JSON file containing named retrieval parameter combinations.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=_non_negative_float,
        default=retrieval.BM25_FUSION_WEIGHT,
    )
    parser.add_argument(
        "--dense-weight",
        type=_non_negative_float,
        default=retrieval.DENSE_FUSION_WEIGHT,
    )
    parser.add_argument(
        "--present-illness-weight",
        type=_non_negative_float,
        default=retrieval.SECTION_WEIGHTS["present_illness_history"],
    )
    parser.add_argument(
        "--past-medical-history-weight",
        type=_non_negative_float,
        default=retrieval.SECTION_WEIGHTS["past_medical_history"],
    )
    parser.add_argument(
        "--physical-exam-weight",
        type=_non_negative_float,
        default=retrieval.SECTION_WEIGHTS["physical_exam"],
    )
    parser.add_argument(
        "--family-history-weight",
        type=_non_negative_float,
        default=retrieval.SECTION_WEIGHTS["family_history"],
    )
    parser.add_argument(
        "--pertinent-results-weight",
        type=_non_negative_float,
        default=retrieval.SECTION_WEIGHTS["pertinent_results"],
    )
    parser.add_argument(
        "--bm25-candidate-k",
        type=_positive_int,
        default=retrieval.SIMILAR_CASE_BM25_CANDIDATE_K,
    )
    parser.add_argument(
        "--dense-candidate-k",
        type=_positive_int,
        default=retrieval.SIMILAR_CASE_DENSE_CANDIDATE_K,
    )
    parser.add_argument(
        "--rrf-candidate-k",
        type=_positive_int,
        default=retrieval.SIMILAR_CASE_RRF_CANDIDATE_K,
    )
    parser.add_argument(
        "--dense-max-length",
        type=_dense_max_length,
        default=retrieval.DENSE_MAX_LENGTH,
    )
    return parser.parse_args()


def _normalize_icd_code(icd_code: str) -> str:
    return icd_code.strip().upper().replace(".", "")


def _evaluate_ranks(
    predicted_icd_codes: list[str],
    golden_icd_code: str,
) -> dict[str, int | None]:
    normalized_golden_code = _normalize_icd_code(golden_icd_code)
    normalized_predicted_codes = [
        _normalize_icd_code(icd_code) for icd_code in predicted_icd_codes
    ]
    ranks = {}
    for metric, prefix_length in METRIC_PREFIX_LENGTHS.items():
        ranks[metric] = next(
            (
                rank
                for rank, predicted_icd_code in enumerate(
                    normalized_predicted_codes,
                    start=1,
                )
                if (
                    predicted_icd_code == normalized_golden_code
                    if prefix_length is None
                    else len(normalized_golden_code) >= prefix_length
                    and len(predicted_icd_code) >= prefix_length
                    and predicted_icd_code[:prefix_length]
                    == normalized_golden_code[:prefix_length]
                )
            ),
            None,
        )
    return ranks


def _base_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "bm25_weight": args.bm25_weight,
        "dense_weight": args.dense_weight,
        "section_weights": {
            "present_illness_history": args.present_illness_weight,
            "past_medical_history": args.past_medical_history_weight,
            "physical_exam": args.physical_exam_weight,
            "family_history": args.family_history_weight,
            "pertinent_results": args.pertinent_results_weight,
        },
        "bm25_candidate_k": args.bm25_candidate_k,
        "dense_candidate_k": args.dense_candidate_k,
        "rrf_candidate_k": args.rrf_candidate_k,
        "dense_max_length": args.dense_max_length,
    }


def _merge_parameters(
    base_parameters: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    unknown_keys = set(overrides) - set(base_parameters)
    if unknown_keys:
        raise ValueError(
            "Unknown experiment parameters: " + ", ".join(sorted(unknown_keys))
        )

    parameters = {
        **base_parameters,
        "section_weights": dict(base_parameters["section_weights"]),
    }
    if "section_weights" in overrides:
        unknown_sections = set(overrides["section_weights"]) - set(
            parameters["section_weights"]
        )
        if unknown_sections:
            raise ValueError(
                "Unknown section weights: " + ", ".join(sorted(unknown_sections))
            )
        parameters["section_weights"].update(overrides["section_weights"])
    for key, value in overrides.items():
        if key != "section_weights":
            parameters[key] = value

    parameters["bm25_weight"] = float(parameters["bm25_weight"])
    parameters["dense_weight"] = float(parameters["dense_weight"])
    parameters["section_weights"] = {
        field_name: float(weight)
        for field_name, weight in parameters["section_weights"].items()
    }
    for key in ("bm25_candidate_k", "dense_candidate_k", "rrf_candidate_k"):
        parameters[key] = int(parameters[key])
    parameters["dense_max_length"] = int(parameters["dense_max_length"])

    if parameters["bm25_weight"] < 0 or parameters["dense_weight"] < 0:
        raise ValueError("BM25 and Dense fusion weights cannot be negative.")
    fusion_weight_total = parameters["bm25_weight"] + parameters["dense_weight"]
    if fusion_weight_total == 0:
        raise ValueError("BM25 and Dense fusion weights cannot both be zero.")
    parameters["bm25_weight"] /= fusion_weight_total
    parameters["dense_weight"] /= fusion_weight_total
    if any(weight < 0 for weight in parameters["section_weights"].values()):
        raise ValueError("Section weights cannot be negative.")
    if sum(parameters["section_weights"].values()) == 0:
        raise ValueError("The five section weights cannot all be zero.")
    if any(
        parameters[key] <= 0
        for key in ("bm25_candidate_k", "dense_candidate_k", "rrf_candidate_k")
    ):
        raise ValueError("Candidate counts must be greater than zero.")
    if not 512 <= parameters["dense_max_length"] <= 8192:
        raise ValueError("Dense max length must be between 512 and 8192.")
    return parameters


def _apply_parameters(parameters: dict[str, Any]) -> None:
    retrieval.BM25_FUSION_WEIGHT = parameters["bm25_weight"]
    retrieval.DENSE_FUSION_WEIGHT = parameters["dense_weight"]
    retrieval.SECTION_WEIGHTS = parameters["section_weights"]
    retrieval.SIMILAR_CASE_BM25_CANDIDATE_K = parameters["bm25_candidate_k"]
    retrieval.SIMILAR_CASE_DENSE_CANDIDATE_K = parameters["dense_candidate_k"]
    retrieval.SIMILAR_CASE_RRF_CANDIDATE_K = parameters["rrf_candidate_k"]
    retrieval.DENSE_MAX_LENGTH = parameters["dense_max_length"]


def _load_records(
    input_path: Path,
) -> tuple[Path, list[tuple[dict[str, Any], PositiveFeaturesResult]]]:
    resolved_input_path = input_path.expanduser().resolve()
    if not resolved_input_path.is_file():
        raise FileNotFoundError(
            f"Diagnosis batch does not exist: {resolved_input_path}"
        )
    records = []
    with resolved_input_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            record = json.loads(line)
            positive_features_result = PositiveFeaturesResult.model_validate(
                record["positive_features_result"]
            )
            records.append((record, positive_features_result))
    if not records:
        raise ValueError(f"Diagnosis batch contains no records: {resolved_input_path}")
    return resolved_input_path, records


def _empty_recall_hits() -> dict[str, dict[str, dict[int, int]]]:
    return {
        method: {
            metric: {cutoff: 0 for cutoff in CUTOFFS}
            for metric in METRIC_PREFIX_LENGTHS
        }
        for method in METHODS
    }


def _recall_summary(
    total: int,
    recall_hits: dict[str, dict[str, dict[int, int]]],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        method: {
            metric: {
                f"recall{cutoff}": recall_hits[method][metric][cutoff] / total
                for cutoff in CUTOFFS
            }
            for metric in METRIC_PREFIX_LENGTHS
        }
        for method in METHODS
    }


def _print_summary(
    title: str,
    total: int,
    summary: dict[str, dict[str, dict[str, float]]],
) -> None:
    print(f"{title} (n={total})")
    print(
        f"{'Method':<12}  "
        f"{'3-digit Recall':^23}  "
        f"{'4-digit Recall':^23}  "
        f"{'Exact ICD Recall':^23}"
    )
    print(
        f"{'':<12}  "
        + "  ".join(f"{'R@1':>7} {'R@3':>7} {'R@5':>7}" for _ in range(3))
    )
    for method in METHODS:
        values = [
            summary[method][metric][f"recall{cutoff}"]
            for metric in METRIC_PREFIX_LENGTHS
            for cutoff in CUTOFFS
        ]
        print(
            f"{method:<12}  "
            + "  ".join(
                " ".join(f"{value:>7.1%}" for value in values[start : start + 3])
                for start in range(0, len(values), 3)
            )
        )


def _run_experiment(
    name: str,
    records: list[tuple[dict[str, Any], PositiveFeaturesResult]],
    parameters: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    _apply_parameters(parameters)
    print(
        f"Experiment {name}: " + json.dumps(parameters, ensure_ascii=False),
        file=sys.stderr,
    )
    recall_hits = _empty_recall_hits()
    with output_path.open("w", encoding="utf-8") as output_file:
        for record_index, (record, positive_features_result) in enumerate(
            records,
            start=1,
        ):
            print(
                f"[{name} {record_index}/{len(records)}] Retrieving "
                f"subject_id={record.get('subject_id')}, "
                f"hadm_id={record.get('hadm_id')} ...",
                file=sys.stderr,
            )
            result_data = retrieval.retrieve_similar_cases(
                positive_features_result
            ).model_dump(mode="json")
            evaluation = {}
            for method in METHODS:
                predicted_icd_codes = [
                    item["icd_code"] for item in result_data[method]
                ]
                evaluated_ranks = _evaluate_ranks(
                    predicted_icd_codes,
                    record["icd_code"],
                )
                evaluation[method] = {
                    "predicted_icd_codes": predicted_icd_codes,
                    "evaluated_ranks": evaluated_ranks,
                }
                for metric, rank in evaluated_ranks.items():
                    if rank is not None:
                        for cutoff in CUTOFFS:
                            recall_hits[method][metric][cutoff] += rank <= cutoff

            output_file.write(
                json.dumps(
                    {
                        "subject_id": record.get("subject_id"),
                        "hadm_id": record.get("hadm_id"),
                        "icd_code": record["icd_code"],
                        "long_title": record["long_title"],
                        "positive_features_result": (
                            positive_features_result.model_dump(mode="json")
                        ),
                        "retrieval_parameters": parameters,
                        "similar_case_retrieval_result": result_data,
                        "evaluation": evaluation,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_file.flush()

    summary = _recall_summary(len(records), recall_hits)
    print()
    _print_summary(f"Experiment: {name}", len(records), summary)
    print()
    return {
        "name": name,
        "parameters": parameters,
        "output": output_path.name,
        "metrics": summary,
    }


def _print_experiment_comparison(
    total: int,
    experiment_summaries: list[dict[str, Any]],
) -> None:
    name_width = max(
        len("Experiment"),
        *(len(experiment["name"]) for experiment in experiment_summaries),
    )
    print(f"RRF experiment comparison (n={total})")
    print(
        f"{'Experiment':<{name_width}}  "
        f"{'3-digit Recall':^23}  "
        f"{'4-digit Recall':^23}  "
        f"{'Exact ICD Recall':^23}"
    )
    print(
        f"{'':<{name_width}}  "
        + "  ".join(f"{'R@1':>7} {'R@3':>7} {'R@5':>7}" for _ in range(3))
    )
    for experiment in experiment_summaries:
        rrf_summary = experiment["metrics"]["rrf"]
        values = [
            rrf_summary[metric][f"recall{cutoff}"]
            for metric in METRIC_PREFIX_LENGTHS
            for cutoff in CUTOFFS
        ]
        print(
            f"{experiment['name']:<{name_width}}  "
            + "  ".join(
                " ".join(f"{value:>7.1%}" for value in values[start : start + 3])
                for start in range(0, len(values), 3)
            )
        )


def run(args: argparse.Namespace) -> Path:
    resolved_input_path, records = _load_records(args.input)
    base_parameters = _merge_parameters(_base_parameters(args), {})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if args.experiments is None:
        output_path = OUTPUT_DIR / f"{resolved_input_path.stem}_{timestamp}.jsonl"
        _run_experiment("single", records, base_parameters, output_path)
        print(f"Output: {output_path}")
        return output_path

    experiments_path = args.experiments.expanduser().resolve()
    if not experiments_path.is_file():
        raise FileNotFoundError(
            f"Experiment configuration does not exist: {experiments_path}"
        )
    experiment_config = json.loads(experiments_path.read_text(encoding="utf-8"))
    experiments = experiment_config["experiments"]
    if not experiments:
        raise ValueError("Experiment configuration contains no experiments.")

    run_directory = OUTPUT_DIR / f"{resolved_input_path.stem}_{timestamp}"
    run_directory.mkdir(parents=True)
    experiment_summaries = []
    seen_names = set()
    for experiment in experiments:
        experiment = dict(experiment)
        name = str(experiment.pop("name"))
        if not name or Path(name).name != name:
            raise ValueError(f"Invalid experiment name: {name!r}")
        if name in seen_names:
            raise ValueError(f"Duplicate experiment name: {name}")
        seen_names.add(name)
        parameters = _merge_parameters(base_parameters, experiment)
        experiment_summaries.append(
            _run_experiment(
                name,
                records,
                parameters,
                run_directory / f"{name}.jsonl",
            )
        )

    summary_path = run_directory / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "input": str(resolved_input_path),
                "total": len(records),
                "experiments": experiment_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _print_experiment_comparison(len(records), experiment_summaries)
    print(f"\nOutput: {run_directory}")
    return run_directory


def main() -> int:
    args = _parse_args()
    try:
        run(args)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
