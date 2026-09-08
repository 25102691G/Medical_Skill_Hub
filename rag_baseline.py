from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agents import Agent, RunConfig, Runner

from config import (
    DIAGNOSIS_MODELS,
    DIAGNOSIS_PROVIDER,
    SIMILAR_CASE_EMBEDDING_BATCH_SIZE,
)
from diagnosis.agents.similar_case_retrieval_agent import (
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
CHUNK_OVERLAP = 64
TOP_K = 10
QUERY_MAX_LENGTH = 8192


RAG_DIAGNOSIS_INSTRUCTIONS = """
You are a gastroenterology clinical decision-support model using retrieval-augmented generation.

Your task is to identify and rank exactly five unique ICD-10-CM candidates for the principal diagnosis
responsible for the current hospitalization.

Use the complete patient case as the only source of facts about the current patient. Retrieved guideline
chunks are external medical knowledge: use them to interpret patient findings and support diagnostic or
management reasoning, but never present their content as findings observed in the patient. Do not invent
missing symptoms, examinations, test results, or history.

Rank the diagnosis chiefly responsible for admission first. Prefer principal diseases over chronic
comorbidities, incidental findings, symptoms, aftercare codes, and secondary complications. Keep lower-ranked
diagnoses clinically plausible when the available evidence is uncertain.

For every diagnosis, use a complete ICD-10-CM code without a decimal point and its canonical English
description. Do not infer a subtype, cause, site, or complication that the patient case does not support.
Use unique codes and ranks 1 through 5 in list order.

Each supporting_evidence item must be anchored in the patient case. When a retrieved chunk supports the
interpretation, append its citation number, such as [1] or [1][2]. Apply the same citation format to
recommended_next_steps that use retrieved knowledge. Cite only the supplied numbers and do not invent
references. Set excluded_planning_candidates to an empty array.

Write all output in English. Use integer confidence values from 0 to 100, keep the summary concise, and
return only valid JSON matching the requested schema. Do not output Markdown or a reasoning trace.
""".strip()


@dataclass(frozen=True)
class GuidelineChunk:
    skill_name: str
    content: str


@dataclass(frozen=True)
class VectorKnowledgeBase:
    chunks: list[GuidelineChunk]
    embeddings: Any


def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed_value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a basic LLM plus guideline vector-RAG diagnosis baseline."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument(
        "--model",
        choices=tuple(DIAGNOSIS_MODELS),
        default=DIAGNOSIS_PROVIDER,
    )
    return parser.parse_args()


def _encode_texts(texts: list[str], *, max_length: int) -> Any:
    torch, functional, _, _ = _require_dense_dependencies()
    tokenizer, model, device = _load_dense_model()
    embeddings = []
    for start in range(0, len(texts), SIMILAR_CASE_EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + SIMILAR_CASE_EMBEDDING_BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        embeddings.append(
            functional.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1)
        )
    return torch.cat(embeddings, dim=0)


def _guideline_paths() -> list[Path]:
    paths = sorted(SKILLS_DIR.glob("*/references/guideline-full-text.md"))
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
        chunk_content = content[
            chunk_offsets[0][0] : chunk_offsets[-1][1]
        ].strip()
        if chunk_content:
            chunks.append(
                GuidelineChunk(
                    skill_name=path.parents[1].name,
                    content=chunk_content,
                )
            )
    return chunks


def _build_vector_knowledge_base() -> VectorKnowledgeBase:
    tokenizer, _, _ = _load_dense_model()
    paths = _guideline_paths()
    chunks = [chunk for path in paths for chunk in _split_guideline(path, tokenizer)]
    print(
        f"Building vector knowledge base from {len(chunks)} chunks "
        f"across {len(paths)} guidelines...",
        file=sys.stderr,
    )
    embeddings = _encode_texts(
        [f"{chunk.skill_name}\n{chunk.content}" for chunk in chunks],
        max_length=CHUNK_SIZE,
    )
    return VectorKnowledgeBase(chunks=chunks, embeddings=embeddings)


def _retrieve_chunks(
    knowledge_base: VectorKnowledgeBase,
    case_text: str,
) -> list[dict[str, object]]:
    torch, _, _, _ = _require_dense_dependencies()
    query_embedding = _encode_texts(
        [case_text],
        max_length=QUERY_MAX_LENGTH,
    )[0]
    scores = torch.matmul(knowledge_base.embeddings, query_embedding)
    indices = torch.argsort(scores, descending=True)[:TOP_K].tolist()
    return [
        {
            "chunk_id": f"G{chunk_index + 1:05d}",
            "skill_name": knowledge_base.chunks[chunk_index].skill_name,
            "content": knowledge_base.chunks[chunk_index].content,
            "similarity": round(float(scores[chunk_index]), 6),
        }
        for chunk_index in indices
    ]


async def _diagnose_case(
    case_text: str,
    retrieved_chunks: list[dict[str, object]],
    model: Any,
) -> DiagnosisResult:
    numbered_evidence = [
        f"[{number}] {chunk['skill_name']}：{chunk['content']}"
        for number, chunk in enumerate(retrieved_chunks, start=1)
    ]
    prompt = (
        "<PATIENT_CASE>\n"
        f"{case_text}\n"
        "</PATIENT_CASE>\n\n"
        "<RETRIEVED_GUIDELINE_CONTEXT>\n"
        f"{json.dumps(numbered_evidence, ensure_ascii=False, indent=2)}\n"
        "</RETRIEVED_GUIDELINE_CONTEXT>\n\n"
        "## Task\n\n"
        "Analyze the complete patient case together with the retrieved external context and return "
        "the five most likely principal-diagnosis candidates with patient-grounded evidence, "
        "appropriate citations, and recommended next steps."
    )
    agent = Agent(
        name="Basic Vector RAG Diagnosis Agent",
        model=model,
        instructions=RAG_DIAGNOSIS_INSTRUCTIONS,
    )
    structured_prompt = _prepare_structured_prompt(
        prompt,
        FinalDiagnosisContent,
        native_structured_output=False,
    )
    run = await Runner.run(
        agent,
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

    citation_pattern = re.compile(r"\[(\d+)\]")
    referenced_numbers = {
        int(reference)
        for diagnosis in diagnosis_content.topk_diagnoses
        for text in [
            *diagnosis.supporting_evidence,
            *diagnosis.recommended_next_steps,
        ]
        for reference in citation_pattern.findall(text)
        if 1 <= int(reference) <= len(numbered_evidence)
    }
    ordered_numbers = [
        number
        for number in range(1, len(numbered_evidence) + 1)
        if number in referenced_numbers
    ]
    citation_mapping = {
        old_number: new_number
        for new_number, old_number in enumerate(ordered_numbers, start=1)
    }
    evidence = [
        citation_pattern.sub(
            f"[{citation_mapping[old_number]}]",
            numbered_evidence[old_number - 1],
            count=1,
        )
        for old_number in ordered_numbers
    ]
    for diagnosis in diagnosis_content.topk_diagnoses:
        diagnosis.supporting_evidence = [
            citation_pattern.sub(
                lambda match: (
                    f"[{citation_mapping[int(match.group(1))]}]"
                    if int(match.group(1)) in citation_mapping
                    else ""
                ),
                text,
            ).strip()
            for text in diagnosis.supporting_evidence
        ]
        diagnosis.recommended_next_steps = [
            citation_pattern.sub(
                lambda match: (
                    f"[{citation_mapping[int(match.group(1))]}]"
                    if int(match.group(1)) in citation_mapping
                    else ""
                ),
                text,
            ).strip()
            for text in diagnosis.recommended_next_steps
        ]
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

    knowledge_base = _build_vector_knowledge_base()
    model = build_diagnosis_model(args.model)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    limit_label = args.limit if args.limit is not None else "all"
    output_path = (
        OUTPUT_DIR
        / f"{input_path.stem}_rag_baseline_{limit_label}_{timestamp}.jsonl"
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
                    retrieved_chunks = _retrieve_chunks(knowledge_base, case_text)
                    result = await _diagnose_case(case_text, retrieved_chunks, model)
                except Exception as exc:
                    print(
                        f"[{index}] Failed {case_label}: "
                        f"{type(exc).__name__}: {exc}",
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
