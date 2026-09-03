from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agents import Agent, RunConfig, Runner
from rank_bm25 import BM25Okapi

from diagnosis.agents.similar_case_retrieval_agent import (
    _encode_texts,
    _load_dense_model,
    _require_dense_dependencies,
)
from main import (
    _diagnosis_model_settings,
    _parse_structured_result,
    _prepare_structured_prompt,
    build_diagnosis_model,
)
from schemas import DiagnosisResult, FinalDiagnosisContent


PROJECT_ROOT = Path(__file__).resolve().parent
SKILLS_DIR = PROJECT_ROOT / "skills"
OUTPUT_DIR = PROJECT_ROOT / "output" / "batch"
CASE_TEXT_COLUMN = "discharge_text_before_disposition"
OUTPUT_COLUMNS = ("subject_id", "hadm_id", "icd_code", "long_title")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
RRF_K = 60


@dataclass(frozen=True)
class GuidelineChunk:
    skill_name: str
    content: str

    @property
    def retrieval_text(self) -> str:
        return f"{self.skill_name}\n{self.content}"


@dataclass
class GuidelineIndex:
    chunks: list[GuidelineChunk]
    bm25: BM25Okapi
    embeddings: Any


def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed_value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the configured DeepSeek model plus all-guideline RAG baseline."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--workers", type=_positive_int, default=1)
    return parser.parse_args()


def _bm25_tokens(text: str) -> list[str]:
    return text.lower().split()


def _guideline_paths() -> list[Path]:
    paths = sorted(
        SKILLS_DIR.glob("*/references/guideline-full-text.md"),
        key=lambda path: path.as_posix(),
    )
    if not paths:
        raise FileNotFoundError(
            f"No guideline-full-text.md files were found below {SKILLS_DIR}."
        )
    return paths


def _split_guideline(path: Path, tokenizer: Any) -> list[GuidelineChunk]:
    content = path.read_text(encoding="utf-8")
    offsets = tokenizer(
        content,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )["offset_mapping"]
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(offsets), step):
        chunk_offsets = offsets[start : start + CHUNK_SIZE]
        if not chunk_offsets:
            break
        chunk_content = content[chunk_offsets[0][0] : chunk_offsets[-1][1]].strip()
        if chunk_content:
            chunks.append(
                GuidelineChunk(
                    skill_name=path.parents[1].name,
                    content=chunk_content,
                )
            )
    return chunks


def _build_guideline_index() -> GuidelineIndex:
    _require_dense_dependencies()
    tokenizer, _, _ = _load_dense_model()
    paths = _guideline_paths()
    chunks = [chunk for path in paths for chunk in _split_guideline(path, tokenizer)]
    print(
        f"Indexing {len(chunks)} chunks from {len(paths)} guideline full texts...",
        file=sys.stderr,
    )
    embeddings = _encode_texts([chunk.retrieval_text for chunk in chunks])

    return GuidelineIndex(
        chunks=chunks,
        bm25=BM25Okapi([_bm25_tokens(chunk.retrieval_text) for chunk in chunks]),
        embeddings=embeddings,
    )


def _retrieve_chunks(index: GuidelineIndex, case_text: str) -> list[dict[str, object]]:
    torch, _, _, _ = _require_dense_dependencies()
    bm25_scores = index.bm25.get_scores(_bm25_tokens(case_text))
    query_embedding = _encode_texts([case_text])[0]
    dense_scores = torch.matmul(index.embeddings, query_embedding)

    bm25_ranking = sorted(
        range(len(index.chunks)),
        key=lambda chunk_index: (-float(bm25_scores[chunk_index]), chunk_index),
    )
    dense_ranking = torch.argsort(dense_scores, descending=True).tolist()
    bm25_ranks = {
        chunk_index: rank for rank, chunk_index in enumerate(bm25_ranking, start=1)
    }
    dense_ranks = {
        chunk_index: rank for rank, chunk_index in enumerate(dense_ranking, start=1)
    }
    rrf_scores = {
        chunk_index: 1 / (RRF_K + bm25_ranks[chunk_index])
        + 1 / (RRF_K + dense_ranks[chunk_index])
        for chunk_index in range(len(index.chunks))
    }
    retrieved_indices = sorted(
        rrf_scores,
        key=lambda chunk_index: (-rrf_scores[chunk_index], chunk_index),
    )[:TOP_K]

    return [
        {
            "chunk_id": f"G{chunk_index + 1:05d}",
            "skill_name": index.chunks[chunk_index].skill_name,
            "content": index.chunks[chunk_index].content,
            "bm25_rank": bm25_ranks[chunk_index],
            "dense_rank": dense_ranks[chunk_index],
            "rrf_score": round(rrf_scores[chunk_index], 8),
        }
        for chunk_index in retrieved_indices
    ]


async def _diagnose_case(
    case_text: str,
    retrieved_chunks: list[dict[str, object]],
    model: Any,
) -> DiagnosisResult:
    evidence = [
        f"[{number}] {chunk['skill_name']}：{chunk['content']}"
        for number, chunk in enumerate(retrieved_chunks, start=1)
    ]
    guideline_context = "\n\n".join(evidence)
    prompt = (
        "Diagnose the principal condition responsible for this hospitalization.\n"
        "Return exactly five unique, ranked ICD-10-CM candidates in English. "
        "Use the patient record for patient facts and the retrieved guideline chunks "
        "only as external medical knowledge. Set excluded_planning_candidates to [].\n\n"
        "<PATIENT_RECORD>\n"
        f"{case_text}\n"
        "</PATIENT_RECORD>\n\n"
        "<RETRIEVED_GUIDELINES>\n"
        f"{guideline_context}\n"
        "</RETRIEVED_GUIDELINES>"
    )
    diagnosis_agent = Agent(
        name="Basic RAG Diagnosis Agent",
        model=model,
        instructions=(
            "You are a gastroenterology diagnosis model. Follow the user request and "
            "return valid JSON only. Do not invent patient findings."
        ),
    )
    structured_prompt = _prepare_structured_prompt(
        prompt,
        FinalDiagnosisContent,
        native_structured_output=False,
    )
    run = await Runner.run(
        diagnosis_agent,
        structured_prompt,
        run_config=RunConfig(model_settings=_diagnosis_model_settings(model)),
    )
    diagnosis_content = _parse_structured_result(
        run.final_output,
        FinalDiagnosisContent,
    )
    if diagnosis_content.excluded_planning_candidates:
        raise ValueError(
            "RAG baseline must return an empty excluded_planning_candidates array."
        )

    skill_names = list(
        dict.fromkeys(str(chunk["skill_name"]) for chunk in retrieved_chunks)
    )
    return DiagnosisResult(
        used_skill=bool(skill_names),
        skill_names=skill_names,
        topk_diagnoses=diagnosis_content.topk_diagnoses,
        excluded_planning_candidates=[],
        summary=diagnosis_content.summary,
        evidence=evidence,
    )


async def _run_batch_async(args: argparse.Namespace) -> Path:
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    guideline_index = _build_guideline_index()
    model = build_diagnosis_model("deepseek-v4-pro")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    limit_label = args.limit if args.limit is not None else "all"
    output_path = (
        OUTPUT_DIR / f"{input_path.stem}_rag_baseline_{limit_label}_{timestamp}.jsonl"
    )

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
            raise ValueError(
                f"Input CSV is missing required columns: {', '.join(sorted(missing_columns))}"
            )

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
                    failed_count += 1
                    continue
                try:
                    retrieved_chunks = _retrieve_chunks(guideline_index, case_text)
                    result = await _diagnose_case(case_text, retrieved_chunks, model)
                except Exception as exc:
                    print(
                        f"[{index}] Failed {case_label}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    failed_count += 1
                    continue

                output_file.write(
                    json.dumps(
                        {
                            "subject_id": row["subject_id"],
                            "hadm_id": row["hadm_id"],
                            "icd_code": row["icd_code"],
                            "long_title": row["long_title"],
                            "retrieved_guideline_chunks": retrieved_chunks,
                            "rag_baseline_result": result.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_file.flush()
                success_count += 1
                print(f"[{index}] Completed {case_label}.", file=sys.stderr)

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
        f"RAG baseline completed: attempted={attempted_count}, "
        f"succeeded={success_count}, failed={failed_count}, output={output_path}",
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
