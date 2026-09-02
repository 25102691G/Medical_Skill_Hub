from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

from main import VllmChatCompletionsModel, make_llm_hypotheses_async


PROJECT_ROOT = Path(__file__).absolute().parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "batch"
CASE_TEXT_COLUMN = "discharge_text_before_disposition"
OUTPUT_COLUMNS = ("subject_id", "hadm_id", "icd_code", "long_title")
MAX_ATTEMPTS = 3


def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed_value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM hypotheses through a vLLM OpenAI-compatible endpoint."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Model name supplied to vLLM with --served-model-name.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="vLLM OpenAI-compatible API base URL.",
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
        help="Maximum number of cases to process concurrently. Default: 1.",
    )
    return parser.parse_args()


async def _run_batch_async(args: argparse.Namespace) -> Path:
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    limit_label = args.limit if args.limit is not None else "all"
    model_label = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model).strip("_")
    output_path = (
        OUTPUT_DIR
        / f"{input_path.stem}_{model_label}_{limit_label}_{timestamp}.jsonl"
    )
    model = VllmChatCompletionsModel(
        model=args.model,
        openai_client=AsyncOpenAI(api_key="EMPTY", base_url=args.base_url),
    )

    async def diagnose_row(
        index: int,
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
                f"[{index}] Skipped CSV row {row_number} ({case_label}): "
                f"{CASE_TEXT_COLUMN} is empty.",
                file=sys.stderr,
            )
            return None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(
                f"[{index}] Generating hypotheses for {case_label} "
                f"(attempt {attempt}/{MAX_ATTEMPTS}) ...",
                file=sys.stderr,
            )
            try:
                result = await make_llm_hypotheses_async(
                    case_text,
                    model=model,
                )
                break
            except Exception as exc:
                print(
                    f"[{index}] Attempt {attempt}/{MAX_ATTEMPTS} failed for "
                    f"CSV row {row_number} ({case_label}): "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
        else:
            return None

        print(f"[{index}] Completed {case_label}.", file=sys.stderr)
        return {
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "icd_code": row["icd_code"],
            "long_title": row["long_title"],
            "llm_hypotheses_result": result.model_dump(mode="json"),
        }

    attempted_count = 0
    success_count = 0
    failed_count = 0
    with (
        input_path.open("r", encoding="utf-8-sig", newline="") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        reader = csv.DictReader(input_file)
        required_columns = {*OUTPUT_COLUMNS, CASE_TEXT_COLUMN}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input CSV is missing required columns: {missing_text}")

        row_queue: asyncio.Queue[
            tuple[int, int, dict[str, str | None]] | None
        ] = asyncio.Queue(maxsize=args.workers)

        async def worker() -> None:
            nonlocal success_count, failed_count
            while True:
                pending_row = await row_queue.get()
                if pending_row is None:
                    return
                index, row_number, row = pending_row
                result = await diagnose_row(index, row_number, row)
                if result is None:
                    failed_count += 1
                    continue
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()
                success_count += 1

        worker_tasks = [asyncio.create_task(worker()) for _ in range(args.workers)]
        for row_number, row in enumerate(reader, start=2):
            if args.limit is not None and attempted_count >= args.limit:
                break
            attempted_count += 1
            await row_queue.put((attempted_count, row_number, row))

        for _ in worker_tasks:
            await row_queue.put(None)
        await asyncio.gather(*worker_tasks)

    print(
        f"Batch completed: attempted={attempted_count}, succeeded={success_count}, "
        f"failed={failed_count}, "
        f"output={output_path.relative_to(PROJECT_ROOT)}",
        file=sys.stderr,
    )
    return output_path


def main() -> int:
    args = _parse_args()
    try:
        asyncio.run(_run_batch_async(args))
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
