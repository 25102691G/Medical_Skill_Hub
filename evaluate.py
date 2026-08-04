import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "evaluate"
METHODS = (
    "search_planning",
    "similar_case_bm25",
    "similar_case_embedding",
    "similar_case_rrf",
    "similar_case_rerank",
    "final_diagnosis",
)
METHOD_LABELS = {
    "search_planning": "Search planning",
    "similar_case_bm25": "Similar BM25",
    "similar_case_embedding": "Similar embedding",
    "similar_case_rrf": "Similar RRF",
    "similar_case_rerank": "Similar rerank",
    "final_diagnosis": "Final diagnosis",
}
METRIC_PREFIX_LENGTHS = {
    "disease": 3,
    "subcategory": 4,
}
METRICS = tuple(METRIC_PREFIX_LENGTHS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate six sets of top-5 principal-diagnosis ICD predictions."
    )
    parser.add_argument(
        "--input",
        type=Path,
    )
    return parser.parse_args()


def _normalize_icd_code(icd_code: str) -> str:
    return icd_code.strip().upper().replace(".", "")


def _evaluate_rank(
    predicted_icd_codes: list[str],
    golden_icd_code: str,
) -> dict[str, int | None]:
    normalized_golden_code = _normalize_icd_code(golden_icd_code)
    normalized_predicted_codes = [
        _normalize_icd_code(icd_code) for icd_code in predicted_icd_codes
    ]
    return {
        metric: next(
            (
                rank
                for rank, predicted_icd_code in enumerate(
                    normalized_predicted_codes,
                    start=1,
                )
                if len(normalized_golden_code) >= prefix_length
                and len(predicted_icd_code) >= prefix_length
                and predicted_icd_code[:prefix_length]
                == normalized_golden_code[:prefix_length]
            ),
            None,
        )
        for metric, prefix_length in METRIC_PREFIX_LENGTHS.items()
    }


def _print_recall_table(
    title: str,
    total: int,
    summary: dict[str, dict[str, dict[str, float]]],
) -> None:
    method_width = max(len("Method"), *(len(method) for method in METHODS))
    print(f"{title} (n={total})")
    print(
        f"{'Method':<{method_width}}  "
        f"{'3-digit Recall':^25}  "
        f"{'4-digit Recall':^25}"
    )
    print(
        f"{'':<{method_width}}  "
        f"{'R@1':>7} {'R@3':>7} {'R@5':>7}  "
        f"{'R@1':>7} {'R@3':>7} {'R@5':>7}"
    )
    for method in METHODS:
        three_digit_values = [
            summary[method]["disease"][f"recall{cutoff}"]
            for cutoff in (1, 3, 5)
        ]
        four_digit_values = [
            summary[method]["subcategory"][f"recall{cutoff}"]
            for cutoff in (1, 3, 5)
        ]
        print(
            f"{method:<{method_width}}  "
            + " ".join(f"{value:>7.1%}" for value in three_digit_values)
            + "  "
            + " ".join(f"{value:>7.1%}" for value in four_digit_values)
        )
    print()


def evaluate_file(input_path: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    output_path = DEFAULT_OUTPUT_DIR / f"{input_path.stem}_evaluation.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    final_recall_hits = {
        method: {
            metric: {1: 0, 3: 0, 5: 0}
            for metric in METRICS
        }
        for method in METHODS
    }
    round_totals: dict[int, int] = {}
    round_recall_hits: dict[
        int, dict[str, dict[str, dict[int, int]]]
    ] = {}
    used_skill_count = 0
    skill_counts: dict[str, int] = {}

    with (
        input_path.open("r", encoding="utf-8") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            golden_icd_code = record["icd_code"].strip()
            golden_diagnosis = record["long_title"].strip()
            multi_round_diagnosis = record["multi_round_diagnosis"]
            round_evaluations = []
            for round_result in multi_round_diagnosis["rounds"]:
                similar_case_result = round_result[
                    "similar_case_retrieval_result"
                ]
                predicted_icd_codes = {
                    "search_planning": [
                        hypothesis["icd_code"].strip()
                        for hypothesis in round_result[
                            "search_planning_result"
                        ]["hypotheses"][:5]
                    ],
                    "similar_case_bm25": [
                        item["icd_code"].strip()
                        for item in similar_case_result["bm25"][:5]
                    ],
                    "similar_case_embedding": [
                        item["icd_code"].strip()
                        for item in similar_case_result["embedding"][:5]
                    ],
                    "similar_case_rrf": [
                        item["icd_code"].strip()
                        for item in similar_case_result["rrf"][:5]
                    ],
                    "similar_case_rerank": [
                        item["icd_code"].strip()
                        for item in similar_case_result["rerank"][:5]
                    ],
                    "final_diagnosis": [
                        diagnosis["icd_code"].strip()
                        for diagnosis in round_result["diagnosis_result"][
                            "topk_diagnoses"
                        ][:5]
                    ],
                }
                evaluated_ranks = {
                    method: _evaluate_rank(
                        predicted_icd_codes[method],
                        golden_icd_code,
                    )
                    for method in METHODS
                }
                round_number = round_result["round"]
                round_evaluations.append(
                    {
                        "round": round_number,
                        **{
                            method: {
                                "predicted_icd_codes": predicted_icd_codes[method],
                                "evaluated_ranks": evaluated_ranks[method],
                            }
                            for method in METHODS
                        },
                    }
                )
                round_totals[round_number] = round_totals.get(round_number, 0) + 1
                round_hits = round_recall_hits.setdefault(
                    round_number,
                    {
                        method: {
                            metric: {1: 0, 3: 0, 5: 0}
                            for metric in METRICS
                        }
                        for method in METHODS
                    },
                )
                for method in METHODS:
                    for metric in METRICS:
                        evaluated_rank = evaluated_ranks[method][metric]
                        if evaluated_rank is not None:
                            for cutoff in (1, 3, 5):
                                round_hits[method][metric][cutoff] += (
                                    evaluated_rank <= cutoff
                                )

            evaluation_record = {
                "subject_id": record.get("subject_id"),
                "hadm_id": record.get("hadm_id"),
                "golden_icd_code": golden_icd_code,
                "golden_diagnosis": golden_diagnosis,
                "is_multi_round": multi_round_diagnosis["is_multi_round"],
                "round_evaluations": round_evaluations,
            }
            output_file.write(
                json.dumps(evaluation_record, ensure_ascii=False) + "\n"
            )

            total += 1
            final_round = multi_round_diagnosis["rounds"][-1]
            if final_round["diagnosis_result"]["used_skill"]:
                used_skill_count += 1
            for skill_name in final_round["diagnosis_result"]["skill_names"]:
                skill_counts[skill_name] = skill_counts.get(skill_name, 0) + 1
            final_evaluated_ranks = {
                method: round_evaluations[-1][method]["evaluated_ranks"]
                for method in METHODS
            }
            for method in METHODS:
                for metric in METRICS:
                    evaluated_rank = final_evaluated_ranks[method][metric]
                    if evaluated_rank is not None:
                        for cutoff in (1, 3, 5):
                            final_recall_hits[method][metric][cutoff] += (
                                evaluated_rank <= cutoff
                            )
            print(
                f"[{line_number}] Evaluated "
                f"subject_id={record.get('subject_id')}, "
                f"hadm_id={record.get('hadm_id')}\n"
                + "\n".join(
                    f"  Round {round_evaluation['round']} | "
                    + "\n          | ".join(
                        f"{METHOD_LABELS[method]}: ICD-3 rank="
                        f"{round_evaluation[method]['evaluated_ranks']['disease'] or 'not found'}, "
                        f"ICD-4 rank="
                        f"{round_evaluation[method]['evaluated_ranks']['subcategory'] or 'not found'}"
                        for method in METHODS
                    )
                    for round_evaluation in round_evaluations
                ),
                file=sys.stderr,
            )

        final_summary = {
            method: {
                metric: {
                    f"recall{cutoff}": (
                        final_recall_hits[method][metric][cutoff] / total
                    )
                    for cutoff in (1, 3, 5)
                }
                for metric in METRICS
            }
            for method in METHODS
        }
        round_summaries = [
            {
                "round": round_number,
                "total": round_totals[round_number],
                **{
                    method: {
                        metric: {
                            f"recall{cutoff}": (
                                round_recall_hits[round_number][method][metric][cutoff]
                                / round_totals[round_number]
                            )
                            for cutoff in (1, 3, 5)
                        }
                        for metric in METRICS
                    }
                    for method in METHODS
                },
            }
            for round_number in sorted(round_totals)
        ]
        summary_record = {
            "total": total,
            "final_result": final_summary,
            "rounds": round_summaries,
            "skill_usage": {
                "used_count": used_skill_count,
                "unused_count": total - used_skill_count,
                "usage_rate": used_skill_count / total,
                "skill_counts": skill_counts,
            },
        }
        output_file.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

    _print_recall_table("Final Results", total, final_summary)
    for round_summary in round_summaries:
        _print_recall_table(
            f"Round {round_summary['round']}",
            round_summary["total"],
            round_summary,
        )
    print(f"skill used: {used_skill_count}")
    print(f"skill unused: {total - used_skill_count}")
    print(f"skill usage rate: {used_skill_count / total:.6f}")
    for skill_name, count in skill_counts.items():
        print(f"skill {skill_name}: {count}")
    print(f"Evaluation details: {output_path}", file=sys.stderr)
    return output_path


def main() -> int:
    args = _parse_args()
    try:
        evaluate_file(args.input)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
