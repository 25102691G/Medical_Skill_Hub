from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import DIAGNOSIS_PROVIDER
from evaluate import evaluate_file
from main import _run_final_diagnosis_async, build_diagnosis_model
from schemas import (
    GuidelineSearchResult,
    KnowledgeSearchResult,
    LlmHypothesesResult,
    SearchPlanningResult,
    SimilarCaseRetrievalResult,
)


PROJECT_ROOT = Path(__file__).resolve().parent
BATCH_INPUT_PATH = (
    PROJECT_ROOT
    / "output/batch/sample5_test_nobhc_75_20260814_145123_037575.jsonl"
)
CASE_INPUT_PATH = PROJECT_ROOT / "database/sample5_test_nobhc.csv"
OUTPUT_DIR = PROJECT_ROOT / "output/final_diagnosis_experiments"
CASE_TEXT_COLUMN = "discharge_text_before_disposition"
WORKERS = 25


async def run_experiment() -> Path:
    with CASE_INPUT_PATH.open(encoding="utf-8-sig", newline="") as input_file:
        case_rows = {
            (row["subject_id"], row["hadm_id"]): row
            for row in csv.DictReader(input_file)
        }

    batch_records = [
        json.loads(line)
        for line in BATCH_INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    diagnosis_model = build_diagnosis_model(DIAGNOSIS_PROVIDER)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUT_DIR / f"{BATCH_INPUT_PATH.stem}_{timestamp}.jsonl"

    semaphore = asyncio.Semaphore(WORKERS)

    async def diagnose_record(record: dict[str, Any]) -> dict[str, object]:
        async with semaphore:
            case_row = case_rows[(record["subject_id"], record["hadm_id"])]
            round_result = record["multi_round_diagnosis"]["rounds"][0]
            diagnosis_result = await _run_final_diagnosis_async(
                case_row[CASE_TEXT_COLUMN],
                LlmHypothesesResult.model_validate(
                    record["llm_hypotheses_result"]
                ),
                SearchPlanningResult.model_validate(
                    round_result["search_planning_result"]
                ),
                KnowledgeSearchResult.model_validate(
                    round_result["knowledge_search_result"]
                ),
                GuidelineSearchResult.model_validate(
                    round_result["guideline_search_result"]
                ),
                SimilarCaseRetrievalResult.model_validate(
                    round_result["similar_case_retrieval_result"]
                ),
                model=diagnosis_model,
                previous_diagnosis_result=None,
                diagnostic_judgement_result=None,
                corrective=False,
                round_index=round_result["round"],
            )
            return {
                "subject_id": record["subject_id"],
                "hadm_id": record["hadm_id"],
                "icd_code": record["icd_code"],
                "long_title": record["long_title"],
                "llm_hypotheses_result": record["llm_hypotheses_result"],
                "multi_round_diagnosis": {
                    "is_multi_round": False,
                    "rounds": [
                        {
                            "round": round_result["round"],
                            "search_planning_result": round_result[
                                "search_planning_result"
                            ],
                            "similar_case_retrieval_result": round_result[
                                "similar_case_retrieval_result"
                            ],
                            "diagnosis_result": diagnosis_result.model_dump(
                                mode="json"
                            ),
                        }
                    ],
                },
            }

    diagnosis_tasks = [
        asyncio.create_task(diagnose_record(record))
        for record in batch_records
    ]
    with output_path.open("w", encoding="utf-8") as output_file:
        for diagnosis_task in diagnosis_tasks:
            output_record = await diagnosis_task
            output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            output_file.flush()

    evaluate_file(output_path)
    return output_path


if __name__ == "__main__":
    result_path = asyncio.run(run_experiment())
    print(result_path.relative_to(PROJECT_ROOT))
