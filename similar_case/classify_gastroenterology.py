from __future__ import annotations

import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


INPUT_PATH = Path(__file__).with_name("icd10.csv")
OUTPUT_PATH = Path(__file__).with_name("icd10_Gastroenterology.csv")
PROGRESS_PATH = Path(__file__).with_name("icd10_Gastroenterology.progress.csv")
DISEASE_COLUMN = "long_title"
CLASSIFICATION_COLUMN = "is_gastroenterology"

MAX_WORKERS = int(os.getenv("DEEPSEEK_CLASSIFICATION_WORKERS", "5"))
MAX_RETRIES = int(os.getenv("DEEPSEEK_CLASSIFICATION_RETRIES", "5"))

PROMPT = """You are a specialist in gastroenterology.

Given a disease term, determine whether it belongs to the field of gastroenterology.

You must make a strict binary classification, even if the term is ambiguous or you are uncertain.

Output:
- 1: The term represents a disease or clinically meaningful diagnostic condition within the scope of gastroenterology.
- 0: The term does not represent a gastroenterology disease or condition.

Only answer with 1 or 0. Do not provide any explanation, punctuation, or additional text.

Example 1:
Disease: Crohn disease
Classification: 1

Example 2:
Disease: Acute pancreatitis
Classification: 1

Example 3:
Disease: Asthma
Classification: 0

Example 4:
Disease: Chronic kidney disease
Classification: 0

You can refer to ICD10 to determine whether it is a gastroenterology disease.

Now, given the following disease or diagnostic term, determine whether it belongs to the field of gastroenterology:

Disease: {disease}
Classification:"""

_thread_local = threading.local()


def _get_client() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        _thread_local.client = client
    return client


def _classify_disease(disease: str) -> str:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _get_client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT.format(disease=disease),
                    }
                ],
                temperature=0,
            )
            choice = response.choices[0]
            if choice.finish_reason == "length":
                raise RuntimeError(
                    f"DeepSeek output was truncated for {disease!r} "
                    f"using model {DEEPSEEK_MODEL!r}."
                )

            result = (choice.message.content or "").strip()
            if not result:
                raise RuntimeError(
                    f"DeepSeek returned empty content for {disease!r} "
                    f"using model {DEEPSEEK_MODEL!r}; "
                    f"finish_reason={choice.finish_reason!r}."
                )
            if result not in {"0", "1"}:
                raise ValueError(f"DeepSeek returned an invalid classification: {result!r}")
            return result
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 30))

    raise RuntimeError(
        f"Failed to classify {disease!r} after {MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def _read_unique_diseases() -> tuple[list[str], list[str]]:
    with INPUT_PATH.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {INPUT_PATH}")
        if DISEASE_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"Input CSV is missing required column {DISEASE_COLUMN!r}: {INPUT_PATH}"
            )
        if CLASSIFICATION_COLUMN in reader.fieldnames:
            raise ValueError(
                f"Input CSV already contains output column {CLASSIFICATION_COLUMN!r}."
            )

        diseases = list(dict.fromkeys(row[DISEASE_COLUMN] for row in reader))
        return reader.fieldnames, diseases


def _load_progress() -> dict[str, str]:
    if not PROGRESS_PATH.exists():
        return {}

    classifications: dict[str, str] = {}
    with PROGRESS_PATH.open("r", encoding="utf-8", newline="") as progress_file:
        reader = csv.DictReader(progress_file)
        expected_columns = [DISEASE_COLUMN, CLASSIFICATION_COLUMN]
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"Unexpected progress file columns in {PROGRESS_PATH}: "
                f"expected {expected_columns}, got {reader.fieldnames}"
            )
        for row in reader:
            result = row[CLASSIFICATION_COLUMN]
            if result not in {"0", "1"}:
                raise ValueError(
                    f"Invalid cached classification for {row[DISEASE_COLUMN]!r}: {result!r}"
                )
            classifications[row[DISEASE_COLUMN]] = result
    return classifications


def _classify_missing_diseases(
    diseases: list[str], classifications: dict[str, str]
) -> None:
    missing_diseases = [disease for disease in diseases if disease not in classifications]
    if not missing_diseases:
        print("All unique diseases are already present in the progress file.")
        return

    progress_exists = PROGRESS_PATH.exists()
    failures: list[tuple[str, Exception]] = []
    completed_now = 0

    with PROGRESS_PATH.open("a", encoding="utf-8", newline="") as progress_file:
        writer = csv.DictWriter(
            progress_file,
            fieldnames=[DISEASE_COLUMN, CLASSIFICATION_COLUMN],
        )
        if not progress_exists:
            writer.writeheader()
            progress_file.flush()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_disease = {
                executor.submit(_classify_disease, disease): disease
                for disease in missing_diseases
            }
            for future in as_completed(future_to_disease):
                disease = future_to_disease[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failures.append((disease, exc))
                    print(f"Classification failed for {disease!r}: {exc}")
                    continue

                classifications[disease] = result
                writer.writerow(
                    {
                        DISEASE_COLUMN: disease,
                        CLASSIFICATION_COLUMN: result,
                    }
                )
                progress_file.flush()
                completed_now += 1
                if completed_now % 100 == 0 or completed_now == len(missing_diseases):
                    total_completed = len(classifications)
                    print(
                        f"Classified {total_completed}/{len(diseases)} unique diseases "
                        f"({completed_now} in this run)."
                    )

    if failures:
        raise RuntimeError(
            f"{len(failures)} disease(s) could not be classified. "
            "Run the script again to retry only the missing diseases."
        )


def _write_output(fieldnames: list[str], classifications: dict[str, str]) -> None:
    temporary_output_path = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    output_fieldnames = [*fieldnames, CLASSIFICATION_COLUMN]

    with (
        INPUT_PATH.open("r", encoding="utf-8", newline="") as input_file,
        temporary_output_path.open("w", encoding="utf-8", newline="") as output_file,
    ):
        reader = csv.DictReader(input_file)
        writer = csv.DictWriter(output_file, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in reader:
            disease = row[DISEASE_COLUMN]
            try:
                row[CLASSIFICATION_COLUMN] = classifications[disease]
            except KeyError as exc:
                raise RuntimeError(f"Missing classification for {disease!r}") from exc
            writer.writerow(row)

    temporary_output_path.replace(OUTPUT_PATH)
    PROGRESS_PATH.unlink(missing_ok=True)


def main() -> None:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured in the project .env file.")
    if MAX_WORKERS < 1:
        raise ValueError("DEEPSEEK_CLASSIFICATION_WORKERS must be at least 1.")
    if MAX_RETRIES < 1:
        raise ValueError("DEEPSEEK_CLASSIFICATION_RETRIES must be at least 1.")

    fieldnames, diseases = _read_unique_diseases()
    classifications = _load_progress()
    print(
        f"Found {len(diseases)} unique diseases; "
        f"{len(classifications)} already classified."
    )
    _classify_missing_diseases(diseases, classifications)
    _write_output(fieldnames, classifications)
    print(f"Completed output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
