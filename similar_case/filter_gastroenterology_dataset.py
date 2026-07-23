#!/usr/bin/env python3
"""Filter exclusion-type ICD diagnoses from the gastroenterology CSV.

The script classifies each unique (icd_version, icd_code, long_title) once,
caches the result, and writes only classification=0 (Retain) rows to the final
dataset. It never sends subject_id, hadm_id, or discharge_text to the API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "icd_is_Gastroenterology_with_discharge_cleaned(1).csv"
DEFAULT_OUTPUT = "final_dataset.csv"
DEFAULT_AUDIT = "icd_exclusion_classifications.csv"
DEFAULT_CACHE = ".icd_exclusion_cache.json"
DEFAULT_MODEL = "gpt-5.6-luna"

REQUIRED_COLUMNS = {"icd_code", "icd_version", "long_title"}
FINAL_DATASET_EXCLUDED_COLUMNS = {"is_gastroenterology"}

PROMPT_TEMPLATE = """You are reviewing ICD-10-CM labels for a curated dataset of specific gastroenterology diseases.

Classify the ICD label itself. Do not infer a more specific diagnosis from the medical record.

Return 1 (EXCLUDE) if the label is:
- a symptom, sign, abnormal test, screening, encounter, status, aftercare, injury, poisoning, foreign body, or pregnancy-related secondary condition; or
- a generic or catch-all category that does not name a disease, pathological process, or clinically meaningful disease family; or
- a nonspecific complication or disorder without stating its type.

Return 0 (RETAIN) if the label identifies a specific gastrointestinal, liver, biliary, pancreatic, or peritoneal disease, a meaningful disease family, or a specific complication.

Important:
- Do not decide solely from the words "other specified" or "unspecified".
- Retain when the underlying disease remains identifiable, such as "acute viral hepatitis, unspecified".
- Exclude generic organ-level labels such as "liver disease, unspecified", "other specified diseases of pancreas", or "other postprocedural complications of digestive system".

Examples:
Other specified diseases of liver (K76.89) -> 1
Acute viral hepatitis, unspecified -> 0
Exocrine pancreatic insufficiency (K86.81) -> 0
Other specified diseases of pancreas (K86.89) -> 1
Postprocedural seroma of a digestive system organ (K91.872) -> 0
Other postprocedural complications and disorders of digestive system (K91.89) -> 1

Only output 1 or 0.

ICD label: {disease}
Classification:"""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify unique ICD diagnoses with the OpenAI Responses API and "
            "retain only classification=0 rows."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Source CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Retained-row CSV path")
    parser.add_argument(
        "--audit",
        default=DEFAULT_AUDIT,
        help="Unique-ICD classification audit CSV path",
    )
    parser.add_argument(
        "--cache",
        default=DEFAULT_CACHE,
        help="Resume cache path for completed API classifications",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"OpenAI model (default: OPENAI_MODEL or {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="none",
        help="Reasoning effort for GPT-5-family models (default: none)",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent API calls")
    parser.add_argument("--timeout", type=float, default=60.0, help="API timeout in seconds")
    parser.add_argument("--retries", type=int, default=5, help="Attempts per diagnosis")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing output and audit CSVs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count the input without API calls or output files",
    )
    return parser.parse_args()


def set_csv_field_limit() -> None:
    """Raise the CSV field limit for long discharge notes, portably."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def normalized_term(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        (row.get("icd_version") or "").strip(),
        (row.get("icd_code") or "").strip(),
        (row.get("long_title") or "").strip(),
    )


def term_cache_key(term: tuple[str, str, str]) -> str:
    return json.dumps(term, ensure_ascii=False, separators=(",", ":"))


def scan_input(path: Path) -> tuple[list[str], Counter[tuple[str, str, str]], int]:
    terms: Counter[tuple[str, str, str]] = Counter()
    row_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("The input CSV has no header row.")

        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            term = normalized_term(row)
            if not all(term):
                raise ValueError(
                    f"Row {row_number} has a blank icd_version, icd_code, or long_title."
                )
            terms[term] += 1
            row_count += 1

    return list(reader.fieldnames), terms, row_count


def cache_metadata(model: str, reasoning_effort: str) -> dict[str, str]:
    prompt_hash = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    return {
        "prompt_sha256": prompt_hash,
        "model": model,
        "reasoning_effort": reasoning_effort,
    }


def load_cache(
    path: Path, model: str, reasoning_effort: str
) -> tuple[dict[str, str], dict[str, int]]:
    expected = cache_metadata(model, reasoning_effort)
    if not path.exists():
        return expected, {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    actual = payload.get("metadata")
    if actual != expected:
        raise ValueError(
            "The cache was created with a different prompt, model, or reasoning "
            "effort. Use another --cache path to avoid mixing classifications."
        )

    raw_items = payload.get("items", {})
    items: dict[str, int] = {}
    for key, value in raw_items.items():
        decision = int(value)
        if decision not in (0, 1):
            raise ValueError(f"Invalid cached classification for {key!r}: {value!r}")
        items[key] = decision
    return expected, items


def save_cache(path: Path, metadata: dict[str, str], items: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": metadata, "items": items},
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    os.replace(temporary, path)


_thread_local = threading.local()


def get_openai_client(timeout: float) -> Any:
    client = getattr(_thread_local, "openai_client", None)
    if client is not None:
        return client

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is not installed. Run: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    client = OpenAI(timeout=timeout, max_retries=0)
    _thread_local.openai_client = client
    return client


def extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    raise ValueError("The API response did not contain output_text.")


def classify_term(
    term: tuple[str, str, str],
    model: str,
    reasoning_effort: str,
    timeout: float,
    retries: int,
) -> int:
    version, code, title = term
    disease = f"{title} (ICD-{version}: {code})"
    prompt = PROMPT_TEMPLATE.replace("{disease}", disease)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            client = get_openai_client(timeout)
            request: dict[str, Any] = {
                "model": model,
                "input": prompt,
                "max_output_tokens": 16,
                "store": False,
            }
            if model.startswith("gpt-5"):
                request["reasoning"] = {"effort": reasoning_effort}

            response = client.responses.create(**request)
            text = extract_output_text(response).strip()
            if not re.fullmatch(r"[01]", text):
                raise ValueError(f"Expected exactly 0 or 1, received {text!r}")
            return int(text)
        except Exception as exc:  # Retries cover rate limits and transient API errors.
            last_error = exc
            if attempt == retries:
                break
            delay = min(30.0, (2 ** (attempt - 1)) + random.random())
            time.sleep(delay)

    raise RuntimeError(f"Classification failed for ICD-{version} {code}: {title}") from last_error


def classify_missing_terms(
    terms: Counter[tuple[str, str, str]],
    cache_path: Path,
    model: str,
    reasoning_effort: str,
    workers: int,
    timeout: float,
    retries: int,
) -> dict[str, int]:
    metadata, cached = load_cache(cache_path, model, reasoning_effort)
    missing = [term for term in terms if term_cache_key(term) not in cached]

    print(
        f"Unique ICD terms: {len(terms):,}; cached: {len(terms) - len(missing):,}; "
        f"API calls needed: {len(missing):,}"
    )
    if not missing:
        return cached

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                classify_term,
                term,
                model,
                reasoning_effort,
                timeout,
                retries,
            ): term
            for term in missing
        }
        try:
            for future in as_completed(futures):
                term = futures[future]
                cached[term_cache_key(term)] = future.result()
                completed += 1
                save_cache(cache_path, metadata, cached)
                if completed % 25 == 0 or completed == len(missing):
                    print(f"Classified {completed:,}/{len(missing):,} new ICD terms")
        except Exception:
            for pending in futures:
                pending.cancel()
            raise

    return cached


def ensure_output_paths(
    input_path: Path, output_path: Path, audit_path: Path, overwrite: bool
) -> None:
    resolved_input = input_path.resolve()
    for path in (output_path, audit_path):
        if path.resolve() == resolved_input:
            raise ValueError(f"Refusing to overwrite the source CSV: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Re-run with --overwrite to replace it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)


def write_audit(
    path: Path,
    terms: Counter[tuple[str, str, str]],
    classifications: dict[str, int],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "icd_version",
                "icd_code",
                "long_title",
                "classification",
                "action",
                "source_row_count",
            ]
        )
        for term in sorted(terms):
            decision = classifications[term_cache_key(term)]
            writer.writerow(
                [*term, decision, "exclude" if decision == 1 else "retain", terms[term]]
            )
    os.replace(temporary, path)


def write_filtered_dataset(
    input_path: Path,
    output_path: Path,
    fieldnames: list[str],
    classifications: dict[str, int],
) -> tuple[int, int]:
    retained = 0
    excluded = 0
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    output_fieldnames = [
        name for name in fieldnames if name not in FINAL_DATASET_EXCLUDED_COLUMNS
    ]

    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source, temporary.open(
            "w", encoding="utf-8", newline=""
        ) as destination:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(destination, fieldnames=output_fieldnames)
            writer.writeheader()

            for row in reader:
                decision = classifications[term_cache_key(normalized_term(row))]
                if decision == 0:
                    writer.writerow({name: row.get(name, "") for name in output_fieldnames})
                    retained += 1
                else:
                    excluded += 1
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return retained, excluded


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.retries < 1:
        raise ValueError("--retries must be at least 1.")

    set_csv_field_limit()
    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_path = Path(args.audit)
    cache_path = Path(args.cache)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    fieldnames, terms, source_rows = scan_input(input_path)
    print(f"Validated {source_rows:,} rows and {len(terms):,} unique ICD terms.")

    if args.dry_run:
        print("Dry run complete: no API calls were made and no files were written.")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY before running the classifier.")

    ensure_output_paths(input_path, output_path, audit_path, args.overwrite)
    classifications = classify_missing_terms(
        terms=terms,
        cache_path=cache_path,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
    )

    expected_keys = {term_cache_key(term) for term in terms}
    missing_keys = expected_keys.difference(classifications)
    if missing_keys:
        raise RuntimeError(f"Missing {len(missing_keys)} classifications; no output was written.")

    write_audit(audit_path, terms, classifications)
    retained, excluded = write_filtered_dataset(
        input_path, output_path, fieldnames, classifications
    )

    if retained + excluded != source_rows:
        raise RuntimeError(
            f"Integrity check failed: {retained} + {excluded} != {source_rows}"
        )

    unique_retained = sum(
        1 for term in terms if classifications[term_cache_key(term)] == 0
    )
    unique_excluded = len(terms) - unique_retained
    print(
        "Complete. "
        f"Rows retained: {retained:,}; rows excluded: {excluded:,}; "
        f"unique ICD terms retained: {unique_retained:,}; "
        f"unique ICD terms excluded: {unique_excluded:,}."
    )
    print(f"Final dataset: {output_path}")
    print(f"Audit table: {audit_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Completed classifications remain in the cache.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
