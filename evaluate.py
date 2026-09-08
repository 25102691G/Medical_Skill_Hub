import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "evaluate"
GUIDELINE_MAPPING_PATH = (
    PROJECT_ROOT / "database" / "mimic_test_icd_guideline_mapping.json"
)
METHODS = (
    "llm_hypotheses",
    "similar_case_bm25",
    "similar_case_embedding",
    "similar_case_rrf",
    "similar_case_rerank",
    "search_planning_result",
    "final_diagnosis",
)
RECALL_CUTOFFS = {
    method: (
        (1, 3, 5, 20)
        if method == "similar_case_rrf"
        else (1, 3, 5, 10)
        if method == "search_planning_result"
        else (1, 3, 5)
    )
    for method in METHODS
}
METHOD_LABELS = {
    "llm_hypotheses": "LLM hypotheses",
    "similar_case_bm25": "Similar BM25",
    "similar_case_embedding": "Similar embedding",
    "similar_case_rrf": "Similar RRF",
    "similar_case_rerank": "Similar LLM reranker",
    "search_planning_result": "Search planning result",
    "final_diagnosis": "Final diagnosis",
}
SIMILAR_CASE_METHODS = {
    "bm25": "similar_case_bm25",
    "embedding": "similar_case_embedding",
    "rrf": "similar_case_rrf",
    "rerank": "similar_case_rerank",
}
METRIC_PREFIX_LENGTHS = {
    "disease": 3,
    "subcategory": 4,
}
METRICS = tuple(METRIC_PREFIX_LENGTHS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate principal-diagnosis ICD predictions from seven stages."
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
    methods: tuple[str, ...] = METHODS,
) -> None:
    method_width = max(len("Method"), *(len(method) for method in methods))
    print(f"{title} (n={total})")
    print(
        f"{'Method':<{method_width}}  "
        f"{'3-digit Recall':^41}  "
        f"{'4-digit Recall':^41}"
    )
    print(
        f"{'':<{method_width}}  "
        f"{'R@1':>7} {'R@3':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7}  "
        f"{'R@1':>7} {'R@3':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7}"
    )
    for method in methods:
        three_digit_values = [
            summary[method]["disease"].get(f"recall{cutoff}")
            for cutoff in (1, 3, 5, 10, 20)
        ]
        four_digit_values = [
            summary[method]["subcategory"].get(f"recall{cutoff}")
            for cutoff in (1, 3, 5, 10, 20)
        ]
        print(
            f"{method:<{method_width}}  "
            + " ".join(
                f"{value:>7.2%}" if value is not None else f"{'-':>7}"
                for value in three_digit_values
            )
            + "  "
            + " ".join(
                f"{value:>7.2%}" if value is not None else f"{'-':>7}"
                for value in four_digit_values
            )
        )
    print()


def _evaluate_llm_hypotheses_file(input_path: Path, output_path: Path) -> Path:
    total = 0
    recall_hits = {
        metric: {cutoff: 0 for cutoff in RECALL_CUTOFFS["llm_hypotheses"]}
        for metric in METRICS
    }

    with (
        input_path.open("r", encoding="utf-8") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            golden_icd_code = record["icd_code"].strip()
            predicted_icd_codes = [
                hypothesis["icd_code"].strip()
                for hypothesis in record["llm_hypotheses_result"][
                    "llm_hypotheses"
                ][:5]
            ]
            evaluated_ranks = _evaluate_rank(
                predicted_icd_codes,
                golden_icd_code,
            )
            for metric in METRICS:
                evaluated_rank = evaluated_ranks[metric]
                if evaluated_rank is not None:
                    for cutoff in RECALL_CUTOFFS["llm_hypotheses"]:
                        recall_hits[metric][cutoff] += evaluated_rank <= cutoff

            output_file.write(
                json.dumps(
                    {
                        "subject_id": record.get("subject_id"),
                        "hadm_id": record.get("hadm_id"),
                        "golden_icd_code": golden_icd_code,
                        "golden_diagnosis": record["long_title"].strip(),
                        "llm_hypotheses": {
                            "predicted_icd_codes": predicted_icd_codes,
                            "evaluated_ranks": evaluated_ranks,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            total += 1
            print(
                f"[{line_number}] Evaluated "
                f"subject_id={record.get('subject_id')}, "
                f"hadm_id={record.get('hadm_id')} | "
                f"LLM hypotheses: ICD-3 rank="
                f"{evaluated_ranks['disease'] or 'not found'}, ICD-4 rank="
                f"{evaluated_ranks['subcategory'] or 'not found'}",
                file=sys.stderr,
            )

        if total == 0:
            raise ValueError("Input JSONL contains no result records.")
        summary = {
            "llm_hypotheses": {
                metric: {
                    f"recall{cutoff}": recall_hits[metric][cutoff] / total
                    for cutoff in RECALL_CUTOFFS["llm_hypotheses"]
                }
                for metric in METRICS
            }
        }
        output_file.write(
            json.dumps(
                {"total": total, "final_result": summary},
                ensure_ascii=False,
            )
            + "\n"
        )

    _print_recall_table(
        "LLM Hypotheses Results",
        total,
        summary,
        methods=("llm_hypotheses",),
    )
    print(f"Evaluation details: {output_path}", file=sys.stderr)
    return output_path


def _evaluate_rag_baseline_file(input_path: Path, output_path: Path) -> Path:
    with GUIDELINE_MAPPING_PATH.open("r", encoding="utf-8") as mapping_file:
        guideline_mapping_data = json.load(mapping_file)
    applicable_guidelines_by_icd = {
        _normalize_icd_code(item["icd_code"]): item["applicable_guidelines"]
        for item in guideline_mapping_data["records"]
    }

    cutoffs = (1, 3, 5)
    total = 0
    recall_hits = {
        metric: {cutoff: 0 for cutoff in cutoffs}
        for metric in METRICS
    }
    applicable_case_count = 0
    hit_case_count = 0
    retrieved_guideline_count = 0
    applicable_retrieved_guideline_count = 0
    case_recall_sum = 0.0

    with (
        input_path.open("r", encoding="utf-8") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            golden_icd_code = record["icd_code"].strip()
            applicable_guidelines = applicable_guidelines_by_icd[
                _normalize_icd_code(golden_icd_code)
            ]
            retrieved_guidelines = list(
                dict.fromkeys(record["rag_baseline_result"]["skill_names"])
            )
            applicable_guideline_set = set(applicable_guidelines)
            matched_applicable_guidelines = [
                skill_name
                for skill_name in retrieved_guidelines
                if skill_name in applicable_guideline_set
            ]
            rag_guideline_hit = int(bool(matched_applicable_guidelines))
            rag_guideline_recall = (
                len(matched_applicable_guidelines) / len(retrieved_guidelines)
                if retrieved_guidelines
                else 0.0
            )
            applicable_case_count += bool(applicable_guidelines)
            hit_case_count += rag_guideline_hit
            retrieved_guideline_count += len(retrieved_guidelines)
            applicable_retrieved_guideline_count += len(
                matched_applicable_guidelines
            )
            case_recall_sum += rag_guideline_recall
            predicted_icd_codes = [
                diagnosis["icd_code"].strip()
                for diagnosis in record["rag_baseline_result"][
                    "topk_diagnoses"
                ][:5]
            ]
            evaluated_ranks = _evaluate_rank(
                predicted_icd_codes,
                golden_icd_code,
            )
            for metric in METRICS:
                evaluated_rank = evaluated_ranks[metric]
                if evaluated_rank is not None:
                    for cutoff in cutoffs:
                        recall_hits[metric][cutoff] += evaluated_rank <= cutoff

            output_file.write(
                json.dumps(
                    {
                        "subject_id": record.get("subject_id"),
                        "hadm_id": record.get("hadm_id"),
                        "golden_icd_code": golden_icd_code,
                        "golden_diagnosis": record["long_title"].strip(),
                        "rag_baseline": {
                            "predicted_icd_codes": predicted_icd_codes,
                            "evaluated_ranks": evaluated_ranks,
                        },
                        "rag_search_result": {
                            "applicable_guidelines": applicable_guidelines,
                            "retrieved_guidelines": retrieved_guidelines,
                            "matched_applicable_guidelines": (
                                matched_applicable_guidelines
                            ),
                            "has_applicable_guideline": bool(
                                applicable_guidelines
                            ),
                            "rag_guideline_hit": rag_guideline_hit,
                            "rag_guideline_recall": rag_guideline_recall,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            total += 1
            print(
                f"[{line_number}] Evaluated "
                f"subject_id={record.get('subject_id')}, "
                f"hadm_id={record.get('hadm_id')} | "
                f"RAG baseline: ICD-3 rank="
                f"{evaluated_ranks['disease'] or 'not found'}, ICD-4 rank="
                f"{evaluated_ranks['subcategory'] or 'not found'}",
                file=sys.stderr,
            )

        if total == 0:
            raise ValueError("Input JSONL contains no result records.")
        summary = {
            "rag_baseline": {
                metric: {
                    f"recall{cutoff}": recall_hits[metric][cutoff] / total
                    for cutoff in cutoffs
                }
                for metric in METRICS
            },
            "rag_search_result": {
                "applicable_case_count": applicable_case_count,
                "hit_case_count": hit_case_count,
                "retrieved_guideline_count": retrieved_guideline_count,
                "applicable_retrieved_guideline_count": (
                    applicable_retrieved_guideline_count
                ),
                "applicable_guideline_hit_rate": (
                    hit_case_count / applicable_case_count
                    if applicable_case_count
                    else None
                ),
                "applicable_guideline_recall_rate": case_recall_sum / total,
            },
        }
        output_file.write(
            json.dumps(
                {"total": total, "final_result": summary},
                ensure_ascii=False,
            )
            + "\n"
        )

    _print_recall_table(
        "DeepSeek + Guideline RAG Baseline Results",
        total,
        summary,
        methods=("rag_baseline",),
    )
    rag_search_summary = summary["rag_search_result"]
    rag_guideline_hit_rate = rag_search_summary[
        "applicable_guideline_hit_rate"
    ]
    print("RAG Search Results (macro average)")
    print(
        "applicable guideline hit rate: "
        + (
            f"{rag_guideline_hit_rate:.2%}"
            if rag_guideline_hit_rate is not None
            else "N/A"
        )
        + " "
        f"({rag_search_summary['hit_case_count']}/"
        f"{rag_search_summary['applicable_case_count']})"
    )
    print(
        "applicable guideline recall rate: "
        f"{rag_search_summary['applicable_guideline_recall_rate']:.2%} "
        f"(average of {total} case recall values)"
    )
    print()
    print(f"Evaluation details: {output_path}", file=sys.stderr)
    return output_path


def evaluate_file(input_path: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    output_path = DEFAULT_OUTPUT_DIR / f"{input_path.stem}_evaluation.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as input_file:
        first_record = next(
            (json.loads(line) for line in input_file if line.strip()),
            None,
        )
    if first_record is None:
        raise ValueError("Input JSONL contains no result records.")
    if "rag_baseline_result" in first_record:
        return _evaluate_rag_baseline_file(input_path, output_path)
    if "multi_round_diagnosis" not in first_record:
        return _evaluate_llm_hypotheses_file(input_path, output_path)

    with GUIDELINE_MAPPING_PATH.open("r", encoding="utf-8") as mapping_file:
        guideline_mapping_data = json.load(mapping_file)
    applicable_guidelines_by_icd = {
        _normalize_icd_code(item["icd_code"]): item["applicable_guidelines"]
        for item in guideline_mapping_data["records"]
    }

    total = 0
    final_recall_hits = {
        method: {
            metric: {cutoff: 0 for cutoff in RECALL_CUTOFFS[method]}
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
    final_guideline_counts = {
        "applicable_case_count": 0,
        "hit_case_count": 0,
        "retrieved_guideline_count": 0,
        "applicable_retrieved_guideline_count": 0,
        "case_recall_sum": 0.0,
    }
    round_guideline_counts: dict[int, dict[str, int | float]] = {}

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
            applicable_guidelines = applicable_guidelines_by_icd[
                _normalize_icd_code(golden_icd_code)
            ]
            applicable_guideline_set = set(applicable_guidelines)
            multi_round_diagnosis = record["multi_round_diagnosis"]
            round_evaluations = []
            for round_result in multi_round_diagnosis["rounds"]:
                similar_case_result = round_result[
                    "similar_case_retrieval_result"
                ]
                predicted_icd_codes = {
                    "llm_hypotheses": [
                        hypothesis["icd_code"].strip()
                        for hypothesis in record["llm_hypotheses_result"][
                            "llm_hypotheses"
                        ][:5]
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
                        for item in similar_case_result["rrf"]
                    ],
                    "similar_case_rerank": [
                        item["icd_code"].strip()
                        for item in (
                            similar_case_result.get("rerank")
                            or similar_case_result["rrf"][:5]
                        )
                    ],
                    "search_planning_result": [
                        hypothesis["icd_code"].strip()
                        for hypothesis in round_result["search_planning_result"][
                            "hypotheses"
                        ]
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
                retrieved_guidelines = list(
                    dict.fromkeys(
                        round_result["guideline_search_result"]["skill_names"]
                    )
                )
                matched_applicable_guidelines = [
                    skill_name
                    for skill_name in retrieved_guidelines
                    if skill_name in applicable_guideline_set
                ]
                applicable_guideline_hit = int(
                    bool(matched_applicable_guidelines)
                )
                applicable_guideline_recall = (
                    len(matched_applicable_guidelines) / len(retrieved_guidelines)
                    if retrieved_guidelines
                    else 0.0
                )
                guideline_evaluation = {
                    "applicable_guidelines": applicable_guidelines,
                    "retrieved_guidelines": retrieved_guidelines,
                    "matched_applicable_guidelines": matched_applicable_guidelines,
                    "has_applicable_guideline": bool(applicable_guidelines),
                    "applicable_guideline_hit": applicable_guideline_hit,
                    "applicable_guideline_recall": applicable_guideline_recall,
                }
                round_number = round_result["round"]
                round_evaluations.append(
                    {
                        "round": round_number,
                        "llm_hypotheses": {
                            "predicted_icd_codes": predicted_icd_codes[
                                "llm_hypotheses"
                            ],
                            "evaluated_ranks": evaluated_ranks["llm_hypotheses"],
                        },
                        "similar_case_retrieval": {
                            stage: {
                                "predicted_icd_codes": predicted_icd_codes[method],
                                "evaluated_ranks": evaluated_ranks[method],
                            }
                            for stage, method in SIMILAR_CASE_METHODS.items()
                        },
                        "search_planning_result": {
                            "predicted_icd_codes": predicted_icd_codes[
                                "search_planning_result"
                            ],
                            "evaluated_ranks": evaluated_ranks[
                                "search_planning_result"
                            ],
                        },
                        "final_diagnosis": {
                            "predicted_icd_codes": predicted_icd_codes[
                                "final_diagnosis"
                            ],
                            "evaluated_ranks": evaluated_ranks["final_diagnosis"],
                        },
                        "guideline_search_result": guideline_evaluation,
                    }
                )
                round_totals[round_number] = round_totals.get(round_number, 0) + 1
                round_hits = round_recall_hits.setdefault(
                    round_number,
                    {
                        method: {
                            metric: {
                                cutoff: 0 for cutoff in RECALL_CUTOFFS[method]
                            }
                            for metric in METRICS
                        }
                        for method in METHODS
                    },
                )
                for method in METHODS:
                    for metric in METRICS:
                        evaluated_rank = evaluated_ranks[method][metric]
                        if evaluated_rank is not None:
                            for cutoff in RECALL_CUTOFFS[method]:
                                round_hits[method][metric][cutoff] += (
                                    evaluated_rank <= cutoff
                                )
                guideline_counts = round_guideline_counts.setdefault(
                    round_number,
                    {
                        "applicable_case_count": 0,
                        "hit_case_count": 0,
                        "retrieved_guideline_count": 0,
                        "applicable_retrieved_guideline_count": 0,
                        "case_recall_sum": 0.0,
                    },
                )
                guideline_counts["applicable_case_count"] += bool(
                    applicable_guidelines
                )
                guideline_counts["hit_case_count"] += applicable_guideline_hit
                guideline_counts["retrieved_guideline_count"] += len(
                    retrieved_guidelines
                )
                guideline_counts["applicable_retrieved_guideline_count"] += len(
                    matched_applicable_guidelines
                )
                guideline_counts["case_recall_sum"] += applicable_guideline_recall

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
            final_guideline_evaluation = round_evaluations[-1][
                "guideline_search_result"
            ]
            final_guideline_counts["applicable_case_count"] += (
                final_guideline_evaluation["has_applicable_guideline"]
            )
            final_guideline_counts["hit_case_count"] += (
                final_guideline_evaluation["applicable_guideline_hit"]
            )
            final_guideline_counts["retrieved_guideline_count"] += len(
                final_guideline_evaluation["retrieved_guidelines"]
            )
            final_guideline_counts[
                "applicable_retrieved_guideline_count"
            ] += len(final_guideline_evaluation["matched_applicable_guidelines"])
            final_guideline_counts["case_recall_sum"] += final_guideline_evaluation[
                "applicable_guideline_recall"
            ]
            if final_round["diagnosis_result"]["used_skill"]:
                used_skill_count += 1
            for skill_name in final_round["diagnosis_result"]["skill_names"]:
                skill_counts[skill_name] = skill_counts.get(skill_name, 0) + 1
            final_evaluated_ranks = {
                "llm_hypotheses": round_evaluations[-1]["llm_hypotheses"][
                    "evaluated_ranks"
                ],
                **{
                    method: round_evaluations[-1]["similar_case_retrieval"][stage][
                        "evaluated_ranks"
                    ]
                    for stage, method in SIMILAR_CASE_METHODS.items()
                },
                "search_planning_result": round_evaluations[-1][
                    "search_planning_result"
                ]["evaluated_ranks"],
                "final_diagnosis": round_evaluations[-1]["final_diagnosis"][
                    "evaluated_ranks"
                ],
            }
            for method in METHODS:
                for metric in METRICS:
                    evaluated_rank = final_evaluated_ranks[method][metric]
                    if evaluated_rank is not None:
                        for cutoff in RECALL_CUTOFFS[method]:
                            final_recall_hits[method][metric][cutoff] += (
                                evaluated_rank <= cutoff
                            )
            evaluation_messages = []
            for round_evaluation in round_evaluations:
                round_evaluated_ranks = {
                    "llm_hypotheses": round_evaluation["llm_hypotheses"][
                        "evaluated_ranks"
                    ],
                    **{
                        method: round_evaluation["similar_case_retrieval"][stage][
                            "evaluated_ranks"
                        ]
                        for stage, method in SIMILAR_CASE_METHODS.items()
                    },
                    "search_planning_result": round_evaluation[
                        "search_planning_result"
                    ]["evaluated_ranks"],
                    "final_diagnosis": round_evaluation["final_diagnosis"][
                        "evaluated_ranks"
                    ],
                }
                evaluation_messages.append(
                    f"  Round {round_evaluation['round']} | "
                    + "\n          | ".join(
                        f"{METHOD_LABELS[method]}: ICD-3 rank="
                        f"{round_evaluated_ranks[method]['disease'] or 'not found'}, "
                        f"ICD-4 rank="
                        f"{round_evaluated_ranks[method]['subcategory'] or 'not found'}"
                        for method in METHODS
                    )
                )
            print(
                f"[{line_number}] Evaluated "
                f"subject_id={record.get('subject_id')}, "
                f"hadm_id={record.get('hadm_id')}\n"
                + "\n".join(evaluation_messages),
                file=sys.stderr,
            )

        flat_final_summary = {
            method: {
                metric: {
                    f"recall{cutoff}": (
                        final_recall_hits[method][metric][cutoff] / total
                    )
                    for cutoff in RECALL_CUTOFFS[method]
                }
                for metric in METRICS
            }
            for method in METHODS
        }
        final_summary = {
            "llm_hypotheses": flat_final_summary["llm_hypotheses"],
            "similar_case_retrieval": {
                stage: flat_final_summary[method]
                for stage, method in SIMILAR_CASE_METHODS.items()
            },
            "search_planning_result": flat_final_summary[
                "search_planning_result"
            ],
            "final_diagnosis": flat_final_summary["final_diagnosis"],
            "guideline_search_result": {
                "applicable_case_count": final_guideline_counts[
                    "applicable_case_count"
                ],
                "hit_case_count": final_guideline_counts["hit_case_count"],
                "retrieved_guideline_count": final_guideline_counts[
                    "retrieved_guideline_count"
                ],
                "applicable_retrieved_guideline_count": final_guideline_counts[
                    "applicable_retrieved_guideline_count"
                ],
                "applicable_guideline_hit_rate": (
                    final_guideline_counts["hit_case_count"]
                    / final_guideline_counts["applicable_case_count"]
                    if final_guideline_counts["applicable_case_count"]
                    else None
                ),
                "applicable_guideline_recall_rate": (
                    final_guideline_counts["case_recall_sum"] / total
                ),
            },
        }
        flat_round_summaries = [
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
                            for cutoff in RECALL_CUTOFFS[method]
                        }
                        for metric in METRICS
                    }
                    for method in METHODS
                },
            }
            for round_number in sorted(round_totals)
        ]
        round_summaries = [
            {
                "round": round_summary["round"],
                "total": round_summary["total"],
                "llm_hypotheses": round_summary["llm_hypotheses"],
                "similar_case_retrieval": {
                    stage: round_summary[method]
                    for stage, method in SIMILAR_CASE_METHODS.items()
                },
                "search_planning_result": round_summary[
                    "search_planning_result"
                ],
                "final_diagnosis": round_summary["final_diagnosis"],
                "guideline_search_result": {
                    "applicable_case_count": round_guideline_counts[
                        round_summary["round"]
                    ]["applicable_case_count"],
                    "hit_case_count": round_guideline_counts[
                        round_summary["round"]
                    ]["hit_case_count"],
                    "retrieved_guideline_count": round_guideline_counts[
                        round_summary["round"]
                    ]["retrieved_guideline_count"],
                    "applicable_retrieved_guideline_count": (
                        round_guideline_counts[round_summary["round"]][
                            "applicable_retrieved_guideline_count"
                        ]
                    ),
                    "applicable_guideline_hit_rate": (
                        round_guideline_counts[round_summary["round"]][
                            "hit_case_count"
                        ]
                        / round_guideline_counts[round_summary["round"]][
                            "applicable_case_count"
                        ]
                        if round_guideline_counts[round_summary["round"]][
                            "applicable_case_count"
                        ]
                        else None
                    ),
                    "applicable_guideline_recall_rate": (
                        round_guideline_counts[round_summary["round"]][
                            "case_recall_sum"
                        ]
                        / round_summary["total"]
                    ),
                },
            }
            for round_summary in flat_round_summaries
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

    _print_recall_table("Final Results", total, flat_final_summary)
    guideline_summary = final_summary["guideline_search_result"]
    guideline_hit_rate = guideline_summary["applicable_guideline_hit_rate"]
    print("Guideline Search Results (macro average)")
    print(
        "applicable guideline hit rate: "
        + (f"{guideline_hit_rate:.2%}" if guideline_hit_rate is not None else "N/A")
        + " "
        f"({guideline_summary['hit_case_count']}/"
        f"{guideline_summary['applicable_case_count']})"
    )
    print(
        "applicable guideline recall rate: "
        f"{guideline_summary['applicable_guideline_recall_rate']:.2%} "
        f"(average of {total} case recall values)"
    )
    print()
    for round_summary in flat_round_summaries:
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
