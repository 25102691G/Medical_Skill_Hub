import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

from config import OPENAI_MODEL


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "evaluate"
VALID_RESULTS = {"No", "1", "2", "3", "4", "5"}

PROMPT = """You are a specialist in gastroenterology. Identify the rank of the gold-standard diagnosis among the five predicted diseases. If a predicted disease contains multiple conditions, use its highest matching rank. Output only "No" or one number from 1 to 5.

Predicted diseases:
{predict_diagnosis}

Gold-standard diagnosis: {golden_diagnosis}"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate top-5 diagnosis recall with an OpenAI judge."
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
    return parser.parse_args()


def _evaluate_rank(
    client: OpenAI,
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
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    result = (response.choices[0].message.content or "").strip()
    if result not in VALID_RESULTS:
        raise ValueError(f"OpenAI returned an invalid result: {result!r}")
    return None if result == "No" else int(result)


def evaluate_file(input_path: Path, output_path: Path | None = None) -> Path:
    input_path = input_path.expanduser().resolve()

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{input_path.stem}_evaluation.jsonl"
    else:
        output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    total = 0
    recall1_hits = 0
    recall3_hits = 0
    recall5_hits = 0

    with (
        input_path.open("r", encoding="utf-8") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            golden_diagnosis = record["long_title"].strip()
            predicted_diseases = [
                diagnosis["disease"].strip()
                for diagnosis in record["diagnosis_result"]["topk_diagnoses"][:5]
            ]
            print(
                f"[{line_number}] Evaluating {golden_diagnosis!r} ...",
                file=sys.stderr,
            )
            evaluated_rank = _evaluate_rank(
                client,
                predicted_diseases,
                golden_diagnosis,
            )
            evaluation_record = {
                "subject_id": record.get("subject_id"),
                "hadm_id": record.get("hadm_id"),
                "golden_diagnosis": golden_diagnosis,
                "predicted_diseases": predicted_diseases,
                "evaluated_rank": evaluated_rank,
            }
            output_file.write(
                json.dumps(evaluation_record, ensure_ascii=False) + "\n"
            )

            total += 1
            if evaluated_rank is not None:
                recall1_hits += evaluated_rank <= 1
                recall3_hits += evaluated_rank <= 3
                recall5_hits += evaluated_rank <= 5
            print(
                f"[{line_number}] Completed: rank={evaluated_rank or 'No'}.",
                file=sys.stderr,
            )

        recall1 = recall1_hits / total
        recall3 = recall3_hits / total
        recall5 = recall5_hits / total
        summary_record = {
            "total": total,
            "recall1": recall1,
            "recall3": recall3,
            "recall5": recall5,
        }
        output_file.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

    print(f"total: {total}")
    print(f"recall1: {recall1:.6f}")
    print(f"recall3: {recall3:.6f}")
    print(f"recall5: {recall5:.6f}")
    print(f"Evaluation details: {output_path}", file=sys.stderr)
    return output_path


def main() -> int:
    args = _parse_args()
    try:
        evaluate_file(args.input, args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
