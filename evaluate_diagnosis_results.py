from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "mimiv_iv_diagnosis_results_20260720_132958_713012.jsonl"
)
MAX_RETRIES = int(os.getenv("DEEPSEEK_EVALUATION_RETRIES", "5"))
VALID_RESULTS = {"No", "1", "2", "3", "4", "5"}

PROMPT = """You are a specialist in the field of gastroenterology.  I will now give you five predicted diseases. Please identify the rank of the following gold-standard diagnosis.  Please output the predicted rank; otherwise, output "No". Only output "No" or "1-5" numbers. If the predicted disease has multiple conditions, only output the top rank. Output only "No" or one number, no additional output.  Predicted diseases: {predict_diagnosis} Standard diagnosis: {golden_diagnosis}."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate top-5 diagnosis recall with a DeepSeek judge."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Diagnosis JSONL path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Evaluation JSONL path. By default, append '_evaluation' to the input "
            "file name."
        ),
    )
    return parser.parse_args()


def _resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()
    return input_path.with_name(f"{input_path.stem}_evaluation.jsonl")


def _extract_case(record: object, line_number: int) -> tuple[str, list[str]]:
    if not isinstance(record, dict):
        raise ValueError(f"JSONL line {line_number} is not a JSON object.")

    golden_diagnosis = record.get("long_title")
    if not isinstance(golden_diagnosis, str) or not golden_diagnosis.strip():
        raise ValueError(
            f"JSONL line {line_number} has no non-empty string 'long_title'."
        )

    diagnosis_result = record.get("diagnosis_result")
    if not isinstance(diagnosis_result, dict):
        raise ValueError(
            f"JSONL line {line_number} has no object 'diagnosis_result'."
        )

    topk_diagnoses = diagnosis_result.get("topk_diagnoses")
    if not isinstance(topk_diagnoses, list) or len(topk_diagnoses) < 5:
        raise ValueError(
            f"JSONL line {line_number} must contain at least five "
            "'diagnosis_result.topk_diagnoses' items."
        )

    predicted_diseases: list[str] = []
    for diagnosis_index, diagnosis in enumerate(topk_diagnoses[:5], start=1):
        if not isinstance(diagnosis, dict):
            raise ValueError(
                f"JSONL line {line_number} top-{diagnosis_index} diagnosis "
                "is not an object."
            )
        disease = diagnosis.get("disease")
        if not isinstance(disease, str) or not disease.strip():
            raise ValueError(
                f"JSONL line {line_number} top-{diagnosis_index} diagnosis "
                "has no non-empty string 'disease'."
            )
        predicted_diseases.append(disease.strip())

    return golden_diagnosis.strip(), predicted_diseases


def _format_prompt(
    predicted_diseases: list[str],
    golden_diagnosis: str,
) -> str:
    numbered_diseases = "\n".join(
        f"{rank}. {disease}"
        for rank, disease in enumerate(predicted_diseases, start=1)
    )
    return PROMPT.format(
        predict_diagnosis=f"\n{numbered_diseases}\n",
        golden_diagnosis=golden_diagnosis,
    )


def _evaluate_rank(
    client: OpenAI,
    predicted_diseases: list[str],
    golden_diagnosis: str,
) -> str:
    last_error: Exception | None = None
    prompt = _format_prompt(predicted_diseases, golden_diagnosis)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            choice = response.choices[0]
            if choice.finish_reason == "length":
                raise RuntimeError("DeepSeek evaluation output was truncated.")

            result = (choice.message.content or "").strip()
            if result not in VALID_RESULTS:
                raise ValueError(
                    f"DeepSeek returned an invalid evaluation result: {result!r}"
                )
            return result
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 30))

    raise RuntimeError(
        f"Evaluation failed after {MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def evaluate_file(input_path: Path, output_path: Path | None = None) -> Path:
    resolved_input_path = input_path.expanduser().resolve()
    if not resolved_input_path.is_file():
        raise FileNotFoundError(
            f"Input JSONL does not exist: {resolved_input_path}"
        )
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not configured in the project .env file."
        )
    if MAX_RETRIES <= 0:
        raise ValueError("DEEPSEEK_EVALUATION_RETRIES must be greater than 0.")

    resolved_output_path = _resolve_output_path(
        resolved_input_path,
        output_path,
    )
    if resolved_output_path == resolved_input_path:
        raise ValueError("Input and output JSONL paths must be different.")
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    total = 0
    recall1_hits = 0
    recall3_hits = 0
    recall5_hits = 0

    with (
        resolved_input_path.open("r", encoding="utf-8") as input_file,
        resolved_output_path.open("w", encoding="utf-8") as output_file,
    ):
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on JSONL line {line_number}: {exc}"
                ) from exc

            golden_diagnosis, predicted_diseases = _extract_case(
                record,
                line_number,
            )
            print(
                f"[{line_number}] Evaluating {golden_diagnosis!r} ...",
                file=sys.stderr,
            )
            result = _evaluate_rank(
                client,
                predicted_diseases,
                golden_diagnosis,
            )
            evaluated_rank = None if result == "No" else int(result)
            evaluation_record = {
                "subject_id": record.get("subject_id"),
                "hadm_id": record.get("hadm_id"),
                "golden_diagnosis": golden_diagnosis,
                "predicted_diseases": predicted_diseases,
                "evaluation_result": result,
                "evaluated_rank": evaluated_rank,
            }
            output_file.write(
                json.dumps(evaluation_record, ensure_ascii=False) + "\n"
            )
            output_file.flush()

            total += 1
            if evaluated_rank is not None:
                recall1_hits += evaluated_rank <= 1
                recall3_hits += evaluated_rank <= 3
                recall5_hits += evaluated_rank <= 5
            print(
                f"[{line_number}] Completed: rank={result}.",
                file=sys.stderr,
            )

    if total == 0:
        raise ValueError(f"Input JSONL contains no records: {resolved_input_path}")

    print(f"total: {total}")
    print(f"recall1: {recall1_hits / total:.6f}")
    print(f"recall3: {recall3_hits / total:.6f}")
    print(f"recall5: {recall5_hits / total:.6f}")
    print(f"Evaluation details: {resolved_output_path}", file=sys.stderr)
    return resolved_output_path


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
