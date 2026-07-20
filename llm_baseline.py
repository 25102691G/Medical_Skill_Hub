import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from interface import LLM_handler


PROJECT_ROOT = Path(__file__).resolve().parent
CASE_TEXT_COLUMN = "discharge_text_before_disposition"
PROMPT = (
    "You are a specialist in the field of gastroenterology. You will be provided "
    "and asked about a complicated clinical case; read it carefully and then "
    "provide a diverse and comprehensive differential diagnosis.  Patient’s "
    "discharge information before disposition: {patient_info} Enumerate the top "
    "5 most likely diagnoses. Be precise, listing one diagnosis per line, and "
    "try to cover many unique possibilities (at least 5). The top 5 diagnoses are:"
)


def parse_diagnoses(text: str) -> list[str]:
    diagnoses = []
    for raw_line in text.splitlines():
        disease = re.sub(
            r"^\s*(?:[-*•]\s+|\d+\s*[.)：:\-]\s*)",
            "",
            raw_line,
        ).strip("*_` ")
        if (
            disease
            and not disease.casefold().startswith("the top 5 diagnoses are")
            and disease.casefold() not in {item.casefold() for item in diagnoses}
        ):
            diagnoses.append(disease)
        if len(diagnoses) == 5:
            return diagnoses
    raise ValueError("The model did not return five unique diagnosis lines.")


def generate_diagnoses(llm_handler: LLM_handler, patient_info: str) -> list[str]:
    prompt = PROMPT.format(patient_info=patient_info)
    response = llm_handler.handler.get_completion(
        (
            "You are a specialist in gastroenterology. Return exactly five "
            "distinct differential diagnoses, one diagnosis per line."
        ),
        prompt,
    )
    return parse_diagnoses(response)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the pure-LLM baseline.")
    parser.add_argument(
        "--model",
        choices=("openai", "gemini", "deepseek", "claude"),
        default=os.getenv("LLM_BASELINE_PROVIDER", "openai").strip().lower(),
        help="LLM provider. Defaults to LLM_BASELINE_PROVIDER or openai.",
    )
    parser.add_argument("--openai_apikey")
    parser.add_argument("--openai_model")
    parser.add_argument("--deepseek_apikey")
    parser.add_argument("--deepseek_model")
    parser.add_argument("--gemini_apikey")
    parser.add_argument("--gemini_model")
    parser.add_argument("--claude_apikey")
    parser.add_argument("--claude_model")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"mimiv_iv_llm_baseline_results_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
    )

    attempted = 0
    succeeded = 0
    try:
        llm_handler = LLM_handler(args)
        with (
            args.csv.expanduser().resolve().open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as input_file,
            output_path.open("w", encoding="utf-8") as output_file,
        ):
            reader = csv.DictReader(input_file)
            required = {"subject_id", "hadm_id", "long_title", CASE_TEXT_COLUMN}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

            for row in reader:
                if args.limit is not None and attempted >= args.limit:
                    break
                attempted += 1
                patient_info = (row[CASE_TEXT_COLUMN] or "").strip()
                if not patient_info:
                    continue

                case_label = (
                    f"subject_id={row['subject_id']}, hadm_id={row['hadm_id']}"
                )
                print(f"[{attempted}] Diagnosing {case_label} ...", file=sys.stderr)
                try:
                    diseases = generate_diagnoses(llm_handler, patient_info)
                except Exception as exc:
                    print(f"[{attempted}] Failed {case_label}: {exc}", file=sys.stderr)
                    continue

                record = {
                    "subject_id": row["subject_id"],
                    "hadm_id": row["hadm_id"],
                    "long_title": row["long_title"],
                    "diagnosis_result": {
                        "topk_diagnoses": [
                            {"rank": rank, "disease": disease}
                            for rank, disease in enumerate(diseases, start=1)
                        ]
                    },
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
                succeeded += 1
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Baseline completed: attempted={attempted}, succeeded={succeeded}, "
        f"output={output_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
