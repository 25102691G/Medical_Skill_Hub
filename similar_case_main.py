from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from agents import Model

from config import DIAGNOSIS_PROVIDER
from diagnosis.agents.similar_case_retrieval_agent import retrieve_similar_cases
from main import _run_search_planning, build_diagnosis_model


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "similar_case"
CASE_TEXT_COLUMN = "discharge_text_before_disposition"
INPUT_COLUMNS = ("subject_id", "hadm_id", "long_title", CASE_TEXT_COLUMN)


def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed_value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run search planning and similar-case retrieval for CSV cases."
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Maximum number of CSV rows to process. If omitted, process all rows.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=50,
        help="Number of cases to process concurrently. Default: 1.",
    )
    return parser.parse_args()


def _process_case(case_text: str, diagnosis_model: Model) -> dict[str, object]:
    search_planning_result = _run_search_planning(
        case_text,
        model=diagnosis_model,
        round_index=1,
    )
    ranking_details: list[dict[str, object]] = []
    similar_case_retrieval_result = retrieve_similar_cases(
        search_planning_result.similar_case_queries,
        debug=True,
        ranking_callback=ranking_details.append,
    )
    return {
        "search_planning_result": {
            "similar_case_queries": search_planning_result.similar_case_queries,
        },
        "similar_case_retrieval_rankings": {
            "rankings": ranking_details,
        },
        "similar_case_retrieval_result": {
            "discharge_disease": similar_case_retrieval_result.discharge_disease,
            "hadm_id": similar_case_retrieval_result.hadm_id,
        },
    }


def main() -> int:
    args = _parse_args()
    input_value = os.getenv("INPUT", "").strip()
    if not input_value:
        print("Error: INPUT is not configured in .env.", file=sys.stderr)
        return 1

    input_path = Path(input_value).expanduser()
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not input_path.is_file():
        print(f"Error: input CSV does not exist: {input_path}", file=sys.stderr)
        return 1

    diagnosis_model = build_diagnosis_model(DIAGNOSIS_PROVIDER)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / (
        "similar_case_results_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
    )

    attempted_count = 0
    success_count = 0
    with (
        input_path.open("r", encoding="utf-8-sig", newline="") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        reader = csv.DictReader(input_file)
        missing_columns = set(INPUT_COLUMNS).difference(reader.fieldnames or [])
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input CSV is missing required columns: {missing_text}")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            cases = []
            for row_number, row in enumerate(reader, start=2):
                if args.limit is not None and attempted_count >= args.limit:
                    break
                attempted_count += 1
                case_text = (row[CASE_TEXT_COLUMN] or "").strip()
                case_label = (
                    f"subject_id={row['subject_id']}, hadm_id={row['hadm_id']}"
                )
                if not case_text:
                    print(
                        f"[{attempted_count}] Skipped CSV row {row_number} "
                        f"({case_label}): {CASE_TEXT_COLUMN} is empty.",
                        file=sys.stderr,
                    )
                    continue

                print(
                    f"[{attempted_count}] Queued {case_label}.",
                    file=sys.stderr,
                )
                future = executor.submit(_process_case, case_text, diagnosis_model)
                cases.append(
                    (attempted_count, row_number, row, case_label, future)
                )

            for case_number, row_number, row, case_label, future in cases:
                try:
                    case_result = future.result()
                except Exception as exc:
                    print(
                        f"[{case_number}] Failed CSV row {row_number} "
                        f"({case_label}): {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                output_record = {
                    "subject_id": row["subject_id"],
                    "hadm_id": row["hadm_id"],
                    "long_title": row["long_title"],
                    **case_result,
                }
                output_file.write(
                    json.dumps(output_record, ensure_ascii=False) + "\n"
                )
                output_file.flush()
                success_count += 1
                print(f"[{case_number}] Completed {case_label}.", file=sys.stderr)

    print(
        f"Similar-case module completed: attempted={attempted_count}, "
        f"succeeded={success_count}, output={output_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
