from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from agents import Model

from config import DIAGNOSIS_PROVIDER
from main import build_diagnosis_model, make_diagnosis_pipeline_async


PROJECT_ROOT = Path(__file__).absolute().parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "batch"
CASE_TEXT_COLUMN = "discharge_text_before_disposition"
OUTPUT_COLUMNS = ("subject_id", "hadm_id", "icd_code", "long_title")
MAX_DIAGNOSIS_ATTEMPTS = 3


def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed_value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the diagnosis pipeline for cases in a MIMIC-IV CSV file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Maximum number of cases to process. If omitted, process all rows.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Maximum number of cases to diagnose concurrently. Default: 1.",
    )
    parser.add_argument(
        "--model",
        choices=("openai", "deepseek", "qwen"),
        default=DIAGNOSIS_PROVIDER,
        help=f"LLM provider. Default: {DIAGNOSIS_PROVIDER}.",
    )
    parser.add_argument("--openai_apikey")
    parser.add_argument("--openai_model")
    parser.add_argument("--deepseek_apikey")
    parser.add_argument("--deepseek_model")
    return parser.parse_args()


def _validate_columns(fieldnames: list[str] | None) -> None:
    required_columns = {*OUTPUT_COLUMNS, CASE_TEXT_COLUMN}
    missing_columns = required_columns.difference(fieldnames or [])
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV is missing required columns: {missing_text}")


async def _run_batch_async(
    csv_path: Path,
    limit: int | None,
    workers: int,
    diagnosis_model: Model,
) -> Path:
    resolved_csv_path = csv_path.expanduser().resolve()
    if not resolved_csv_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {resolved_csv_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    limit_label = limit if limit is not None else "all"
    output_path = (
        OUTPUT_DIR / f"{resolved_csv_path.stem}_{limit_label}_{timestamp}.jsonl"
    )

    attempted_count = 0
    success_count = 0
    failed_count = 0

    async def diagnose_row(
        attempted_index: int,
        row_number: int,
        row: dict[str, str | None],
    ) -> dict[str, object] | None:
        case_text = (row.get(CASE_TEXT_COLUMN) or "").strip()
        case_label = (
            f"subject_id={row.get('subject_id', '')}, "
            f"hadm_id={row.get('hadm_id', '')}"
        )
        if not case_text:
            print(
                f"[{attempted_index}] Skipped CSV row {row_number} ({case_label}): "
                f"{CASE_TEXT_COLUMN} is empty.",
                file=sys.stderr,
            )
            return None

        for attempt in range(1, MAX_DIAGNOSIS_ATTEMPTS + 1):
            action = "Diagnosing" if attempt == 1 else "Retrying"
            print(
                f"[{attempted_index}] {action} {case_label} "
                f"(attempt {attempt}/{MAX_DIAGNOSIS_ATTEMPTS}) ...",
                file=sys.stderr,
            )
            try:
                pipeline_result = await make_diagnosis_pipeline_async(
                    case_text,
                    model=diagnosis_model,
                )
                break
            except Exception as exc:
                error_stage = getattr(exc, "stage", "diagnosis_pipeline")
                print(
                    f"[{attempted_index}] Attempt {attempt}/{MAX_DIAGNOSIS_ATTEMPTS} "
                    f"failed for CSV row {row_number} ({case_label}) at {error_stage}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
        else:
            print(
                f"[{attempted_index}] Failed {case_label} after "
                f"{MAX_DIAGNOSIS_ATTEMPTS} attempts.",
                file=sys.stderr,
            )
            return None

        print(
            f"[{attempted_index}] Completed {case_label} on attempt "
            f"{attempt}/{MAX_DIAGNOSIS_ATTEMPTS}.",
            file=sys.stderr,
        )
        return {
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "icd_code": row["icd_code"],
            "long_title": row["long_title"],
            **pipeline_result.model_dump(mode="json"),
        }

    with (
        resolved_csv_path.open("r", encoding="utf-8-sig", newline="") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        reader = csv.DictReader(input_file)
        _validate_columns(reader.fieldnames)

        row_queue: asyncio.Queue[
            tuple[int, int, dict[str, str | None]] | None
        ] = asyncio.Queue(maxsize=workers)

        async def worker() -> None:
            nonlocal success_count, failed_count
            while True:
                pending_row = await row_queue.get()
                if pending_row is None:
                    return
                attempted_index, pending_row_number, row = pending_row
                output_record = await diagnose_row(
                    attempted_index,
                    pending_row_number,
                    row,
                )
                if output_record is not None:
                    output_file.write(
                        json.dumps(output_record, ensure_ascii=False) + "\n"
                    )
                    output_file.flush()
                    success_count += 1
                else:
                    failed_count += 1

        worker_tasks = [
            asyncio.create_task(worker())
            for _ in range(workers)
        ]
        for row_number, row in enumerate(reader, start=2):
            if limit is not None and attempted_count >= limit:
                break

            attempted_count += 1
            await row_queue.put((attempted_count, row_number, row))

        for _ in worker_tasks:
            await row_queue.put(None)
        await asyncio.gather(*worker_tasks)

    print(
        f"Batch completed: attempted={attempted_count}, succeeded={success_count}, "
        f"failed={failed_count}, output={output_path.relative_to(PROJECT_ROOT)}",
        file=sys.stderr,
    )
    return output_path


def run_batch(
    csv_path: Path,
    limit: int | None,
    diagnosis_model: Model,
    workers: int = 1,
) -> Path:
    return asyncio.run(
        _run_batch_async(
            csv_path,
            limit,
            workers,
            diagnosis_model,
        )
    )


def main() -> int:
    args = _parse_args()
    try:
        diagnosis_model = build_diagnosis_model(
            args.model,
            openai_api_key=args.openai_apikey or "",
            openai_model=args.openai_model or "",
            deepseek_api_key=args.deepseek_apikey or "",
            deepseek_model=args.deepseek_model or "",
        )
        run_batch(
            args.input,
            args.limit,
            diagnosis_model,
            workers=args.workers,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
