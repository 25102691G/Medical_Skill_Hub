from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from config import (
    MIMIC_IV_CASE_PATH,
    SIMILAR_CASE_EMBEDDING_BATCH_SIZE,
    SIMILAR_CASE_EMBEDDING_CACHE_PATH,
    SIMILAR_CASE_EMBEDDING_DEVICE,
    SIMILAR_CASE_EMBEDDING_MODEL,
    SIMILAR_CASE_TOP_K,
)
from schemas import SimilarCaseRetrievalResult


REQUIRED_COLUMNS = {
    "hadm_id",
    "long_title",
    "discharge_text",
}
RRF_K = 60
_TEXT_PART_PATTERN = re.compile(r"[a-z0-9]+")
RankingDetails = dict[str, object]
RankingCallback = Callable[[RankingDetails], None]


@dataclass(frozen=True)
class _CaseRecord:
    hadm_id: str
    discharge_disease: str
    discharge_text: str
    similar_case_content: str


def _require_rank_bm25() -> Any:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError(
            "Similar-case retrieval requires rank-bm25. "
            "Install the dependencies from requirements.txt in the project virtual environment."
        ) from exc
    return BM25Okapi


def _require_dense_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Dense similar-case retrieval requires torch and transformers. "
            "Install the dependencies from requirements.txt in the project virtual environment."
        ) from exc
    return torch, F, AutoModel, AutoTokenizer


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


@lru_cache(maxsize=1)
def _load_case_records(path: Path = MIMIC_IV_CASE_PATH) -> tuple[_CaseRecord, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Similar-case database does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Similar-case database has no header row: {path}")

        columns = {
            _cell_text(column_name): column_name
            for column_name in reader.fieldnames
            if _cell_text(column_name)
        }
        missing_columns = sorted(REQUIRED_COLUMNS - columns.keys())
        if missing_columns:
            raise ValueError(
                f"Similar-case database is missing required columns {missing_columns}: {path}"
            )

        records: list[_CaseRecord] = []
        for row in reader:
            hadm_id = _cell_text(row.get(columns["hadm_id"]))
            discharge_disease = _cell_text(row.get(columns["long_title"]))
            discharge_text = _cell_text(row.get(columns["discharge_text"]))
            if not discharge_text:
                continue
            records.append(
                _CaseRecord(
                    hadm_id=hadm_id,
                    discharge_disease=discharge_disease,
                    discharge_text=discharge_text,
                    similar_case_content=discharge_text,
                )
            )
        return tuple(records)


def _join_query(items: list[str]) -> str:
    return " ".join(item.strip() for item in items if item.strip())


def _tokenize(text: str) -> list[str]:
    return _TEXT_PART_PATTERN.findall(text.lower())


def _empty_retrieval_result() -> SimilarCaseRetrievalResult:
    return SimilarCaseRetrievalResult(
        discharge_disease=[],
        hadm_id=[],
        discharge_texts=[],
    )


def _bm25_ranking(query: str, corpus: list[str]) -> list[tuple[int, float]]:
    query_tokens = _tokenize(query)
    eligible_indices = [index for index, text in enumerate(corpus) if text]
    tokenized_corpus = [_tokenize(corpus[index]) for index in eligible_indices]
    if not query_tokens or not tokenized_corpus:
        return []
    corpus_tokens = {token for document in tokenized_corpus for token in document}
    if corpus_tokens.isdisjoint(query_tokens):
        return []

    BM25Okapi = _require_rank_bm25()
    scores = BM25Okapi(tokenized_corpus).get_scores(query_tokens)
    ranked_positions = sorted(
        range(len(eligible_indices)),
        key=lambda position: float(scores[position]),
        reverse=True,
    )
    return [
        (eligible_indices[position], float(scores[position]))
        for position in ranked_positions
    ]


def _database_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as database_file:
        for chunk in iter(lambda: database_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_dense_model() -> tuple[Any, Any, str]:
    torch, _, AutoModel, AutoTokenizer = _require_dense_dependencies()
    if SIMILAR_CASE_EMBEDDING_DEVICE not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "SIMILAR_CASE_EMBEDDING_DEVICE must be one of: auto, cpu, cuda."
        )
    if SIMILAR_CASE_EMBEDDING_DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = SIMILAR_CASE_EMBEDDING_DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "SIMILAR_CASE_EMBEDDING_DEVICE is set to cuda, but CUDA is unavailable."
        )
    tokenizer = AutoTokenizer.from_pretrained(SIMILAR_CASE_EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(SIMILAR_CASE_EMBEDDING_MODEL)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _encode_texts(texts: list[str]) -> Any:
    torch, F, _, _ = _require_dense_dependencies()
    tokenizer, model, device = _load_dense_model()
    embeddings = []
    for start in range(0, len(texts), SIMILAR_CASE_EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + SIMILAR_CASE_EMBEDDING_BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        batch_embeddings = F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1)
        embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0)


def _load_or_build_corpus_embeddings(
    records: tuple[_CaseRecord, ...],
) -> Any:
    torch, _, _, _ = _require_dense_dependencies()
    fingerprint = _database_fingerprint(MIMIC_IV_CASE_PATH)
    cache_path = SIMILAR_CASE_EMBEDDING_CACHE_PATH

    if cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            cache.get("database_fingerprint") == fingerprint
            and cache.get("model_name") == SIMILAR_CASE_EMBEDDING_MODEL
            and cache.get("record_count") == len(records)
            and cache.get("content_schema") == "discharge_text_v1"
        ):
            return cache["combined_embeddings"]

    combined_embeddings = _encode_texts(
        [record.similar_case_content for record in records]
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "database_fingerprint": fingerprint,
            "model_name": SIMILAR_CASE_EMBEDDING_MODEL,
            "record_count": len(records),
            "content_schema": "discharge_text_v1",
            "combined_embeddings": combined_embeddings,
        },
        cache_path,
    )
    return combined_embeddings


def _dense_ranking(
    query: str,
    corpus_embeddings: Any,
    eligible_indices: list[int],
) -> list[tuple[int, float]]:
    torch, _, _, _ = _require_dense_dependencies()
    query_embedding = _encode_texts([query])[0]
    eligible_embeddings = corpus_embeddings[eligible_indices]
    scores = torch.matmul(eligible_embeddings, query_embedding)
    ranked_positions = torch.argsort(scores, descending=True).tolist()
    return [
        (eligible_indices[position], float(scores[position]))
        for position in ranked_positions
    ]


def _build_ranking_details(
    query_field: str,
    method: str,
    query: str,
    ranking: list[tuple[int, float]],
    records: tuple[_CaseRecord, ...],
    *,
    skipped_reason: str | None = None,
) -> RankingDetails:
    ranking_items = [
        {
            "rank": rank,
            "hadm_id": records[record_index].hadm_id,
            "discharge_disease": records[record_index].discharge_disease,
            "score": score,
        }
        for rank, (record_index, score) in enumerate(
            ranking[:SIMILAR_CASE_TOP_K],
            start=1,
        )
    ]
    return {
        "query_field": query_field,
        "method": method,
        "query": query,
        "status": "skipped" if skipped_reason is not None else "completed",
        "skipped_reason": skipped_reason,
        "ranking": ranking_items,
    }


def _print_ranking_debug(details: RankingDetails) -> None:
    print(
        (
            "\n===== Similar Case Retrieval "
            f"{details['query_field']} {details['method']} Ranking ====="
        ),
        file=sys.stderr,
    )
    print(
        json.dumps(details, ensure_ascii=False, indent=2),
        file=sys.stderr,
    )


def _report_ranking(
    query_field: str,
    method: str,
    query: str,
    ranking: list[tuple[int, float]],
    records: tuple[_CaseRecord, ...],
    *,
    debug: bool,
    ranking_callback: RankingCallback | None,
    skipped_reason: str | None = None,
) -> None:
    details = _build_ranking_details(
        query_field,
        method,
        query,
        ranking,
        records,
        skipped_reason=skipped_reason,
    )
    if debug:
        _print_ranking_debug(details)
    if ranking_callback is not None:
        ranking_callback(details)


def _report_skipped_rankings(
    query_field: str,
    query: str,
    records: tuple[_CaseRecord, ...],
    reason: str,
    *,
    debug: bool,
    ranking_callback: RankingCallback | None,
) -> None:
    for method in ("BM25", "Dense"):
        _report_ranking(
            query_field,
            method,
            query,
            [],
            records,
            debug=debug,
            ranking_callback=ranking_callback,
            skipped_reason=reason,
        )


def _rrf_rank(rankings: list[list[int]], result_count: int) -> list[int]:
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranking in rankings:
        for rank, record_index in enumerate(ranking, start=1):
            scores[record_index] = scores.get(record_index, 0.0) + 1.0 / (RRF_K + rank)
            best_rank[record_index] = min(best_rank.get(record_index, rank), rank)
    return sorted(
        scores,
        key=lambda record_index: (
            -scores[record_index],
            best_rank[record_index],
            record_index,
        ),
    )[:result_count]


def retrieve_similar_cases(
    similar_case_queries: list[str],
    *,
    debug: bool = False,
    ranking_callback: RankingCallback | None = None,
) -> SimilarCaseRetrievalResult:

    query = _join_query(similar_case_queries)
    if not query:
        _report_skipped_rankings(
            "similar_case_queries",
            query,
            (),
            "The similar-case query is empty.",
            debug=debug,
            ranking_callback=ranking_callback,
        )
        return _empty_retrieval_result()

    records = _load_case_records()
    if not records:
        _report_skipped_rankings(
            "similar_case_queries",
            query,
            records,
            "The similar-case corpus has no usable records.",
            debug=debug,
            ranking_callback=ranking_callback,
        )
        return _empty_retrieval_result()

    corpus = [record.similar_case_content for record in records]
    corpus_embeddings = _load_or_build_corpus_embeddings(records)

    rankings: list[list[int]] = []
    eligible_indices = [index for index, text in enumerate(corpus) if text]
    bm25_ranking = _bm25_ranking(query, corpus)
    _report_ranking(
        "similar_case_queries",
        "BM25",
        query,
        bm25_ranking,
        records,
        debug=debug,
        ranking_callback=ranking_callback,
        skipped_reason=(
            None
            if bm25_ranking
            else "The query and corpus have no shared BM25 tokens."
        ),
    )
    if bm25_ranking:
        rankings.append([record_index for record_index, _ in bm25_ranking])
    dense_ranking = _dense_ranking(
        query,
        corpus_embeddings,
        eligible_indices,
    )
    _report_ranking(
        "similar_case_queries",
        "Dense",
        query,
        dense_ranking,
        records,
        debug=debug,
        ranking_callback=ranking_callback,
    )
    rankings.append([record_index for record_index, _ in dense_ranking])

    if not rankings:
        return _empty_retrieval_result()

    top_indices = _rrf_rank(rankings, min(SIMILAR_CASE_TOP_K, len(records)))
    return SimilarCaseRetrievalResult(
        discharge_disease=[
            records[index].discharge_disease for index in top_indices
        ],
        hadm_id=[records[index].hadm_id for index in top_indices],
        discharge_texts=[records[index].discharge_text for index in top_indices]
    )
