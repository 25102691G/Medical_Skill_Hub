import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, OPENAI_MODEL


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "evaluate"
VALID_RESULTS = {"No", "1", "2", "3", "4", "5"}
METHODS = (
    "search_planning",
    "similar_case_retrieval",
    "final_diagnosis",
)

PROMPT = """You are a specialist in gastroenterology. Identify the rank of the gold-standard diagnosis among the five predicted diseases based only on the underlying ICD-10-CM disease subcategory or disease type.

Ignore all complication information, including whether a diagnosis is with or without complications and any specific complication type. Diagnoses that differ only in complication status or complication type must be treated as a match. Accept synonymous clinical wording and differences in word order when the underlying disease subcategory or disease type is equivalent. A broad disease family, symptom, or related condition does not match a more specific disease subcategory. If a predicted disease contains multiple conditions, use its highest matching rank.

Output only "No" or one number from 1 to 5.

Predicted diseases:
{predict_diagnosis}

Gold-standard diagnosis: {golden_diagnosis}"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate three sets of top-5 diagnosis results with an LLM judge."
    )
    parser.add_argument(
        "--model",
        choices=("openai", "deepseek"),
        default="openai",
        help="Evaluation model provider. Default: openai.",
    )
    parser.add_argument(
        "--input",
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Evaluation JSONL path. By default, write to output/evaluate and append "
            "'_evaluation' to the input file name."
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def _evaluate_rank(
    client: OpenAI,
    model_name: str,
    predicted_diseases: list[str],
    golden_diagnosis: str,
) -> int | None:
    numbered_diseases = "\n".join(
        f"{rank}. {disease}"
        for rank, disease in enumerate(predicted_diseases, start=1)
    )
    prompt = PROMPT.format(
        predict_diagnosis=numbered_diseases,
        golden_diagnosis=golden_diagnosis,
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        seed=42,
    )
    result = (response.choices[0].message.content or "").strip()
    if result not in VALID_RESULTS:
        raise ValueError(f"The model returned an invalid result: {result!r}")
    return None if result == "No" else int(result)


def evaluate_file(
    input_path: Path,
    output_path: Path | None = None,
    model: str = "openai",
    workers: int = 1,
) -> Path:
    input_path = input_path.expanduser().resolve()

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{input_path.stem}_evaluation.jsonl"
    else:
        output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model == "deepseek":
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        model_name = DEEPSEEK_MODEL
    else:
        client = OpenAI()
        model_name = OPENAI_MODEL
    total = 0
    final_recall_hits = {
        method: {1: 0, 3: 0, 5: 0}
        for method in METHODS
    }
    round_totals: dict[int, int] = {}
    round_recall_hits: dict[int, dict[str, dict[int, int]]] = {}
    used_skill_count = 0
    skill_counts: dict[str, int] = {}

    with (
        input_path.open("r", encoding="utf-8") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            cases = []
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                golden_diagnosis = record["long_title"].strip()
                multi_round_diagnosis = record["multi_round_diagnosis"]
                print(
                    f"[{line_number}] Queued "
                    f"subject_id={record.get('subject_id')}, "
                    f"hadm_id={record.get('hadm_id')}.",
                    file=sys.stderr,
                )
                round_jobs = []
                for round_result in multi_round_diagnosis["rounds"]:
                    predicted_diseases = {
                        "search_planning": [
                            disease.strip()
                            for disease in round_result["search_planning_result"][
                                "hypotheses"
                            ][:5]
                        ],
                        "similar_case_retrieval": [
                            disease.strip()
                            for disease in round_result[
                                "similar_case_retrieval_result"
                            ]["discharge_disease"][:5]
                        ],
                        "final_diagnosis": [
                            diagnosis["disease"].strip()
                            for diagnosis in round_result["diagnosis_result"][
                                "topk_diagnoses"
                            ][:5]
                        ],
                    }
                    futures = {
                        method: executor.submit(
                            _evaluate_rank,
                            client,
                            model_name,
                            predicted_diseases[method],
                            golden_diagnosis,
                        )
                        for method in METHODS
                    }
                    round_jobs.append(
                        (
                            round_result["round"],
                            predicted_diseases,
                            futures,
                        )
                    )
                cases.append(
                    (
                        line_number,
                        record,
                        golden_diagnosis,
                        multi_round_diagnosis,
                        round_jobs,
                    )
                )

            for (
                line_number,
                record,
                golden_diagnosis,
                multi_round_diagnosis,
                round_jobs,
            ) in cases:
                round_evaluations = []
                for round_number, predicted_diseases, futures in round_jobs:
                    evaluated_ranks = {
                        method: futures[method].result()
                        for method in METHODS
                    }
                    round_evaluations.append(
                        {
                            "round": round_number,
                            **{
                                method: {
                                    "predicted_diseases": predicted_diseases[method],
                                    "evaluated_rank": evaluated_ranks[method],
                                }
                                for method in METHODS
                            },
                        }
                    )
                    round_totals[round_number] = round_totals.get(round_number, 0) + 1
                    round_hits = round_recall_hits.setdefault(
                        round_number,
                        {
                            method: {1: 0, 3: 0, 5: 0}
                            for method in METHODS
                        },
                    )
                    for method in METHODS:
                        evaluated_rank = evaluated_ranks[method]
                        if evaluated_rank is not None:
                            for cutoff in (1, 3, 5):
                                round_hits[method][cutoff] += evaluated_rank <= cutoff

                evaluation_record = {
                    "subject_id": record.get("subject_id"),
                    "hadm_id": record.get("hadm_id"),
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
                    method: round_evaluations[-1][method]["evaluated_rank"]
                    for method in METHODS
                }
                for method in METHODS:
                    evaluated_rank = final_evaluated_ranks[method]
                    if evaluated_rank is not None:
                        for cutoff in (1, 3, 5):
                            final_recall_hits[method][cutoff] += (
                                evaluated_rank <= cutoff
                            )
                print(
                    f"[{line_number}] Completed "
                    f"subject_id={record.get('subject_id')}, "
                    f"hadm_id={record.get('hadm_id')}: "
                    + "; ".join(
                        f"round {round_evaluation['round']} "
                        + ", ".join(
                            f"{method}="
                            f"{round_evaluation[method]['evaluated_rank'] or 'No'}"
                            for method in METHODS
                        )
                        for round_evaluation in round_evaluations
                    )
                    + ".",
                    file=sys.stderr,
                )

        final_summary = {
            method: {
                f"recall{cutoff}": final_recall_hits[method][cutoff] / total
                for cutoff in (1, 3, 5)
            }
            for method in METHODS
        }
        round_summaries = [
            {
                "round": round_number,
                "total": round_totals[round_number],
                **{
                    method: {
                        f"recall{cutoff}": (
                            round_recall_hits[round_number][method][cutoff]
                            / round_totals[round_number]
                        )
                        for cutoff in (1, 3, 5)
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

    print(f"total: {total}")
    for method in METHODS:
        print(f"final {method} recall1: {final_summary[method]['recall1']:.6f}")
        print(f"final {method} recall3: {final_summary[method]['recall3']:.6f}")
        print(f"final {method} recall5: {final_summary[method]['recall5']:.6f}")
    for round_summary in round_summaries:
        print(f"round {round_summary['round']} total: {round_summary['total']}")
        for method in METHODS:
            print(
                f"round {round_summary['round']} {method} recall1: "
                f"{round_summary[method]['recall1']:.6f}"
            )
            print(
                f"round {round_summary['round']} {method} recall3: "
                f"{round_summary[method]['recall3']:.6f}"
            )
            print(
                f"round {round_summary['round']} {method} recall5: "
                f"{round_summary[method]['recall5']:.6f}"
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
        evaluate_file(args.input, args.output, args.model, args.workers)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
