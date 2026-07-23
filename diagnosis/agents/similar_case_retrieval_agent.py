from __future__ import annotations

import csv
import hashlib
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from config import (
    MIMIC_IV_CASE_PATH,
    SIMILAR_CASE_BM25_CANDIDATE_K,
    SIMILAR_CASE_DENSE_CANDIDATE_K,
    SIMILAR_CASE_EMBEDDING_BATCH_SIZE,
    SIMILAR_CASE_EMBEDDING_CACHE_PATH,
    SIMILAR_CASE_EMBEDDING_DEVICE,
    SIMILAR_CASE_EMBEDDING_MODEL,
    SIMILAR_CASE_RERANKER_BATCH_SIZE,
    SIMILAR_CASE_RERANKER_DEVICE,
    SIMILAR_CASE_RERANKER_MODEL,
    SIMILAR_CASE_RRF_CANDIDATE_K,
    SIMILAR_CASE_TOP_K,
)
from schemas import SimilarCaseRetrievalResult


SECTION_COLUMNS = (
    "chief_complaint",
    "major_surgical_or_invasive_procedure",
    "history_of_present_illness",
    "past_medical_history",
    "family_history",
    "physical_exam",
    "pertinent_results",
    "brief_hospital_course",
    "medications_on_admission",
)
REQUIRED_COLUMNS = {"hadm_id", "long_title", *SECTION_COLUMNS}
RRF_K = 60
logger = logging.getLogger(__name__)
BM25_CACHE_PATH = MIMIC_IV_CASE_PATH.with_name(
    f"{MIMIC_IV_CASE_PATH.stem}_bm25.pkl"
)
RankingDetails = dict[str, object]
RankingCallback = Callable[[RankingDetails], None]
_bm25_index: tuple[Any, list[int]] | None = None
_bm25_index_lock = Lock()
_dense_dependencies: tuple[Any, Any, Any, Any] | None = None
_dense_dependencies_lock = Lock()
_dense_model: tuple[Any, Any, str] | None = None
_dense_model_lock = Lock()
_reranker_model: tuple[Any, Any, str] | None = None
_reranker_model_lock = Lock()
_corpus_embeddings: Any | None = None
_corpus_embeddings_lock = Lock()


@dataclass(frozen=True)
class _CaseRecord:
    hadm_id: str
    discharge_disease: str


@dataclass(frozen=True)
class _SectionRecord:
    case_index: int
    name: str
    content: str


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
    global _dense_dependencies
    if _dense_dependencies is not None:
        return _dense_dependencies

    with _dense_dependencies_lock:
        if _dense_dependencies is not None:
            return _dense_dependencies
        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Dense similar-case retrieval requires torch and transformers. "
                "Install the dependencies from requirements.txt in the project virtual environment."
            ) from exc
        _dense_dependencies = (torch, F, AutoModel, AutoTokenizer)
        return _dense_dependencies


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


@lru_cache(maxsize=1)
def _load_case_records(
    path: Path = MIMIC_IV_CASE_PATH,
) -> tuple[tuple[_CaseRecord, ...], tuple[_SectionRecord, ...]]:
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

        cases: list[_CaseRecord] = []
        sections: list[_SectionRecord] = []
        case_indices: dict[str, int] = {}
        for row in reader:
            hadm_id = _cell_text(row.get(columns["hadm_id"]))
            discharge_disease = _cell_text(row.get(columns["long_title"]))
            section_values = [
                (section_name, _cell_text(row.get(columns[section_name])))
                for section_name in SECTION_COLUMNS
            ]
            section_values = [
                (section_name, content)
                for section_name, content in section_values
                if content
            ]
            if not section_values:
                continue
            if hadm_id not in case_indices:
                case_indices[hadm_id] = len(cases)
                cases.append(
                    _CaseRecord(
                        hadm_id=hadm_id,
                        discharge_disease=discharge_disease,
                    )
                )
            case_index = case_indices[hadm_id]
            sections.extend(
                _SectionRecord(
                    case_index=case_index,
                    name=section_name,
                    content=content,
                )
                for section_name, content in section_values
            )
        return tuple(cases), tuple(sections)


def _join_query(items: list[str]) -> str:
    return " ".join(item.strip() for item in items if item.strip())


@lru_cache(maxsize=1)
def _load_scispacy_nlp() -> Any:
    try:
        import spacy
        from scispacy.custom_tokenizer import combined_rule_tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "BM25 similar-case retrieval requires spaCy and scispaCy. "
            "Install the dependencies from requirements.txt in the project virtual environment."
        ) from exc

    nlp = spacy.blank("en")
    nlp.tokenizer = combined_rule_tokenizer(nlp)
    return nlp


def _tokenize(text: str) -> list[str]:
    doc = _load_scispacy_nlp().make_doc(text)
    return [
        token.lower_
        for token in doc
        if not token.is_space and not token.is_punct
    ]


def _empty_retrieval_result() -> SimilarCaseRetrievalResult:
    return SimilarCaseRetrievalResult(
        discharge_disease=[],
        Sections=[],
    )


def _load_or_build_bm25_index() -> tuple[Any, list[int]]:
    global _bm25_index
    if _bm25_index is not None:
        return _bm25_index

    with _bm25_index_lock:
        if _bm25_index is not None:
            return _bm25_index

        cases, sections = _load_case_records()
        fingerprint = _database_fingerprint(MIMIC_IV_CASE_PATH)
        if BM25_CACHE_PATH.is_file():
            with BM25_CACHE_PATH.open("rb") as cache_file:
                cache = pickle.load(cache_file)
            if (
                cache.get("database_fingerprint") == fingerprint
                and cache.get("case_count") == len(cases)
                and cache.get("section_count") == len(sections)
                and cache.get("content_schema") == "section_text_scispacy_v1"
            ):
                _bm25_index = (cache["bm25"], cache["eligible_indices"])
                return _bm25_index

        eligible_indices = [
            index for index, section in enumerate(sections) if section.content
        ]
        tokenized_corpus = [
            _tokenize(sections[index].content)
            for index in eligible_indices
        ]
        BM25Okapi = _require_rank_bm25()
        bm25 = BM25Okapi(tokenized_corpus)
        BM25_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with BM25_CACHE_PATH.open("wb") as cache_file:
            pickle.dump(
                {
                    "database_fingerprint": fingerprint,
                    "case_count": len(cases),
                    "section_count": len(sections),
                    "content_schema": "section_text_scispacy_v1",
                    "eligible_indices": eligible_indices,
                    "bm25": bm25,
                },
                cache_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        _bm25_index = (bm25, eligible_indices)
        return _bm25_index


def _bm25_ranking(query: str) -> list[tuple[int, float]]:
    query_tokens = _tokenize(query)
    bm25, eligible_indices = _load_or_build_bm25_index()
    if not query_tokens or not eligible_indices:
        return []
    if set(bm25.idf).isdisjoint(query_tokens):
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_positions = sorted(
        (
            position
            for position in range(len(eligible_indices))
            if float(scores[position]) > 0
        ),
        key=lambda position: float(scores[position]),
        reverse=True,
    )[:SIMILAR_CASE_BM25_CANDIDATE_K]
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


def _load_dense_model() -> tuple[Any, Any, str]:
    global _dense_model
    if _dense_model is not None:
        return _dense_model

    with _dense_model_lock:
        if _dense_model is not None:
            return _dense_model
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
        _dense_model = (tokenizer, model, device)
        return _dense_model


def _load_reranker_model() -> tuple[Any, Any, str]:
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model

    with _reranker_model_lock:
        if _reranker_model is not None:
            return _reranker_model
        torch, _, _, AutoTokenizer = _require_dense_dependencies()
        from transformers import AutoModelForSequenceClassification

        if SIMILAR_CASE_RERANKER_DEVICE not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                "SIMILAR_CASE_RERANKER_DEVICE must be one of: auto, cpu, cuda."
            )
        if SIMILAR_CASE_RERANKER_DEVICE == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = SIMILAR_CASE_RERANKER_DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "SIMILAR_CASE_RERANKER_DEVICE is set to cuda, but CUDA is unavailable."
            )
        tokenizer = AutoTokenizer.from_pretrained(SIMILAR_CASE_RERANKER_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            SIMILAR_CASE_RERANKER_MODEL
        )
        model.to(device)
        model.eval()
        _reranker_model = (tokenizer, model, device)
        return _reranker_model


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
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)


def _load_or_build_corpus_embeddings(
    cases: tuple[_CaseRecord, ...],
    sections: tuple[_SectionRecord, ...],
) -> Any:
    global _corpus_embeddings
    if _corpus_embeddings is not None:
        return _corpus_embeddings

    with _corpus_embeddings_lock:
        if _corpus_embeddings is not None:
            return _corpus_embeddings
        torch, _, _, _ = _require_dense_dependencies()
        _, _, device = _load_dense_model()
        fingerprint = _database_fingerprint(MIMIC_IV_CASE_PATH)
        cache_path = SIMILAR_CASE_EMBEDDING_CACHE_PATH

        if cache_path.is_file():
            cache = torch.load(cache_path, map_location=device, weights_only=False)
            if (
                cache.get("database_fingerprint") == fingerprint
                and cache.get("model_name") == SIMILAR_CASE_EMBEDDING_MODEL
                and cache.get("case_count") == len(cases)
                and cache.get("section_count") == len(sections)
                and cache.get("content_schema") == "section_text_v1"
            ):
                _corpus_embeddings = cache["section_embeddings"]
                return _corpus_embeddings

        section_embeddings = _encode_texts(
            [section.content for section in sections]
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "database_fingerprint": fingerprint,
                "model_name": SIMILAR_CASE_EMBEDDING_MODEL,
                "case_count": len(cases),
                "section_count": len(sections),
                "content_schema": "section_text_v1",
                "section_embeddings": section_embeddings.cpu(),
            },
            cache_path,
        )
        _corpus_embeddings = section_embeddings
        return _corpus_embeddings


def _dense_ranking(
    query: str,
    corpus_embeddings: Any,
    eligible_indices: list[int],
) -> list[tuple[int, float]]:
    torch, _, _, _ = _require_dense_dependencies()
    query_embedding = _encode_texts([query])[0]
    eligible_embeddings = corpus_embeddings[eligible_indices]
    scores = torch.matmul(eligible_embeddings, query_embedding)
    ranked_positions = torch.topk(
        scores,
        k=min(SIMILAR_CASE_DENSE_CANDIDATE_K, len(eligible_indices)),
    ).indices.tolist()
    return [
        (eligible_indices[position], float(scores[position]))
        for position in ranked_positions
    ]


def _aggregate_section_ranking(
    section_ranking: list[tuple[int, float]],
    sections: tuple[_SectionRecord, ...],
) -> tuple[list[tuple[int, float]], dict[int, list[tuple[int, float]]]]:
    case_hits: dict[int, list[tuple[int, float]]] = {}
    for section_index, score in section_ranking:
        hits = case_hits.setdefault(sections[section_index].case_index, [])
        if len(hits) < 2:
            hits.append((section_index, score))

    case_ranking = [
        (
            case_index,
            hits[0][1] + (0.2 * hits[1][1] if len(hits) > 1 else 0.0),
        )
        for case_index, hits in case_hits.items()
    ]
    case_ranking.sort(key=lambda item: (-item[1], item[0]))
    return case_ranking, case_hits


def _section_hit_details(
    hits: list[tuple[int, float]],
    sections: tuple[_SectionRecord, ...],
) -> list[dict[str, object]]:
    return [
        {
            "section": sections[section_index].name,
            "content": sections[section_index].content,
            "score": score,
        }
        for section_index, score in hits
    ]


def _select_reranker_sections(
    candidate_ranking: list[tuple[int, float]],
    bm25_section_ranking: list[tuple[int, float]],
    dense_section_ranking: list[tuple[int, float]],
    sections: tuple[_SectionRecord, ...],
) -> dict[int, list[tuple[int, float]]]:
    candidate_indices = {case_index for case_index, _ in candidate_ranking}
    section_scores: dict[int, float] = {}
    best_ranks: dict[int, int] = {}
    for section_ranking in (bm25_section_ranking, dense_section_ranking):
        for rank, (section_index, _) in enumerate(section_ranking, start=1):
            if sections[section_index].case_index not in candidate_indices:
                continue
            section_scores[section_index] = section_scores.get(
                section_index,
                0.0,
            ) + 1.0 / (RRF_K + rank)
            best_ranks[section_index] = min(
                best_ranks.get(section_index, rank),
                rank,
            )

    selected: dict[int, list[tuple[int, float]]] = {}
    for case_index, _ in candidate_ranking:
        case_section_indices = [
            section_index
            for section_index in section_scores
            if sections[section_index].case_index == case_index
        ]
        case_section_indices.sort(
            key=lambda section_index: (
                -section_scores[section_index],
                best_ranks[section_index],
                section_index,
            )
        )
        selected[case_index] = [
            (section_index, section_scores[section_index])
            for section_index in case_section_indices[:2]
        ]
    return selected


def _rerank_candidates(
    query: str,
    candidate_ranking: list[tuple[int, float]],
    reranker_hits: dict[int, list[tuple[int, float]]],
    sections: tuple[_SectionRecord, ...],
) -> list[tuple[int, float]]:
    torch, _, _, _ = _require_dense_dependencies()
    tokenizer, model, device = _load_reranker_model()
    documents = [
        "\n\n".join(
            (
                f"Section: {sections[section_index].name}\n"
                f"{sections[section_index].content}"
            )
            for section_index, _ in reranker_hits[case_index]
        )
        for case_index, _ in candidate_ranking
    ]
    scores: list[float] = []
    for start in range(0, len(documents), SIMILAR_CASE_RERANKER_BATCH_SIZE):
        batch_documents = documents[
            start : start + SIMILAR_CASE_RERANKER_BATCH_SIZE
        ]
        pairs = [[query, document] for document in batch_documents]
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            batch_scores = model(**inputs).logits.squeeze(dim=1)
        scores.extend(float(score) for score in batch_scores)

    rrf_ranks = {
        case_index: rank
        for rank, (case_index, _) in enumerate(candidate_ranking, start=1)
    }
    reranked = [
        (case_index, score)
        for (case_index, _), score in zip(candidate_ranking, scores)
    ]
    reranked.sort(key=lambda item: (-item[1], rrf_ranks[item[0]]))
    return reranked[:SIMILAR_CASE_TOP_K]


def _build_ranking_details(
    query_field: str,
    method: str,
    query: str,
    ranking: list[tuple[int, float]],
    cases: tuple[_CaseRecord, ...],
    sections: tuple[_SectionRecord, ...],
    *,
    bm25_ranks: dict[int, int] | None = None,
    dense_ranks: dict[int, int] | None = None,
    bm25_hits: dict[int, list[tuple[int, float]]] | None = None,
    dense_hits: dict[int, list[tuple[int, float]]] | None = None,
    reranker_hits: dict[int, list[tuple[int, float]]] | None = None,
    skipped_reason: str | None = None,
) -> RankingDetails:
    ranking_items = []
    ranking_limit = (
        SIMILAR_CASE_RRF_CANDIDATE_K
        if method == "RRF"
        else SIMILAR_CASE_TOP_K
    )
    for rank, (record_index, score) in enumerate(
        ranking[:ranking_limit],
        start=1,
    ):
        item: dict[str, object] = {
            "rank": rank,
            "hadm_id": cases[record_index].hadm_id,
            "discharge_disease": cases[record_index].discharge_disease,
        }
        if method == "RRF":
            item.update(
                {
                    "rrf_score": score,
                    "bm25_rank": bm25_ranks.get(record_index) if bm25_ranks else None,
                    "dense_rank": dense_ranks.get(record_index) if dense_ranks else None,
                    "bm25_top_sections": _section_hit_details(
                        bm25_hits.get(record_index, []) if bm25_hits else [],
                        sections,
                    ),
                    "dense_top_sections": _section_hit_details(
                        dense_hits.get(record_index, []) if dense_hits else [],
                        sections,
                    ),
                }
            )
        elif method == "Reranker":
            item.update(
                {
                    "reranker_score": score,
                    "reranker_sections": _section_hit_details(
                        (
                            reranker_hits.get(record_index, [])
                            if reranker_hits
                            else []
                        ),
                        sections,
                    ),
                }
            )
        else:
            item["score"] = score
            hits = bm25_hits if method == "BM25" else dense_hits
            item["top_sections"] = _section_hit_details(
                hits.get(record_index, []) if hits else [],
                sections,
            )
        ranking_items.append(item)
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
    cases: tuple[_CaseRecord, ...],
    sections: tuple[_SectionRecord, ...],
    *,
    debug: bool,
    ranking_callback: RankingCallback | None,
    bm25_ranks: dict[int, int] | None = None,
    dense_ranks: dict[int, int] | None = None,
    bm25_hits: dict[int, list[tuple[int, float]]] | None = None,
    dense_hits: dict[int, list[tuple[int, float]]] | None = None,
    reranker_hits: dict[int, list[tuple[int, float]]] | None = None,
    skipped_reason: str | None = None,
) -> None:
    details = _build_ranking_details(
        query_field,
        method,
        query,
        ranking,
        cases,
        sections,
        bm25_ranks=bm25_ranks,
        dense_ranks=dense_ranks,
        bm25_hits=bm25_hits,
        dense_hits=dense_hits,
        reranker_hits=reranker_hits,
        skipped_reason=skipped_reason,
    )
    if debug:
        _print_ranking_debug(details)
    if ranking_callback is not None:
        ranking_callback(details)


def _report_skipped_rankings(
    query_field: str,
    query: str,
    cases: tuple[_CaseRecord, ...],
    sections: tuple[_SectionRecord, ...],
    reason: str,
    *,
    debug: bool,
    ranking_callback: RankingCallback | None,
) -> None:
    for method in ("BM25", "Dense", "RRF", "Reranker"):
        _report_ranking(
            query_field,
            method,
            query,
            [],
            cases,
            sections,
            debug=debug,
            ranking_callback=ranking_callback,
            skipped_reason=reason,
        )


def _rrf_rank(
    bm25_ranking: list[tuple[int, float]],
    dense_ranking: list[tuple[int, float]],
    result_count: int,
) -> tuple[list[tuple[int, float]], dict[int, int], dict[int, int]]:
    bm25_ranks = {
        record_index: rank
        for rank, (record_index, _) in enumerate(bm25_ranking, start=1)
    }
    dense_ranks = {
        record_index: rank
        for rank, (record_index, _) in enumerate(dense_ranking, start=1)
    }
    scores = {
        record_index: 1.0 / (RRF_K + rank)
        for record_index, rank in bm25_ranks.items()
    }
    for record_index, rank in dense_ranks.items():
        scores[record_index] = scores.get(record_index, 0.0) + 1.0 / (
            RRF_K + rank
        )
    ranked_indices = sorted(
        scores,
        key=lambda record_index: (
            -scores[record_index],
            min(
                rank
                for rank in (
                    bm25_ranks.get(record_index),
                    dense_ranks.get(record_index),
                )
                if rank is not None
            ),
            record_index,
        ),
    )[:result_count]
    return (
        [(record_index, scores[record_index]) for record_index in ranked_indices],
        bm25_ranks,
        dense_ranks,
    )


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
            (),
            "The similar-case query is empty.",
            debug=debug,
            ranking_callback=ranking_callback,
        )
        return _empty_retrieval_result()

    cases, sections = _load_case_records()
    if not sections:
        _report_skipped_rankings(
            "similar_case_queries",
            query,
            cases,
            sections,
            "The similar-case corpus has no usable records.",
            debug=debug,
            ranking_callback=ranking_callback,
        )
        return _empty_retrieval_result()

    corpus_embeddings = _load_or_build_corpus_embeddings(cases, sections)

    eligible_indices = [
        index for index, section in enumerate(sections) if section.content
    ]
    bm25_section_ranking = _bm25_ranking(query)
    bm25_ranking, bm25_hits = _aggregate_section_ranking(
        bm25_section_ranking,
        sections,
    )
    _report_ranking(
        "similar_case_queries",
        "BM25",
        query,
        bm25_ranking,
        cases,
        sections,
        debug=debug,
        ranking_callback=ranking_callback,
        bm25_hits=bm25_hits,
        skipped_reason=(
            None
            if bm25_ranking
            else "BM25 produced no positive-scoring candidates."
        ),
    )
    dense_section_ranking = _dense_ranking(
        query,
        corpus_embeddings,
        eligible_indices,
    )
    dense_ranking, dense_hits = _aggregate_section_ranking(
        dense_section_ranking,
        sections,
    )
    _report_ranking(
        "similar_case_queries",
        "Dense",
        query,
        dense_ranking,
        cases,
        sections,
        debug=debug,
        ranking_callback=ranking_callback,
        dense_hits=dense_hits,
    )
    rrf_ranking, bm25_ranks, dense_ranks = _rrf_rank(
        bm25_ranking,
        dense_ranking,
        min(SIMILAR_CASE_RRF_CANDIDATE_K, len(cases)),
    )
    _report_ranking(
        "similar_case_queries",
        "RRF",
        query,
        rrf_ranking,
        cases,
        sections,
        debug=debug,
        ranking_callback=ranking_callback,
        bm25_ranks=bm25_ranks,
        dense_ranks=dense_ranks,
        bm25_hits=bm25_hits,
        dense_hits=dense_hits,
        skipped_reason=(
            None
            if rrf_ranking
            else "Neither retrieval branch produced candidates."
        ),
    )
    if not rrf_ranking:
        _report_ranking(
            "similar_case_queries",
            "Reranker",
            query,
            [],
            cases,
            sections,
            debug=debug,
            ranking_callback=ranking_callback,
            skipped_reason="RRF produced no candidate cases.",
        )
        return _empty_retrieval_result()

    reranker_hits = _select_reranker_sections(
        rrf_ranking,
        bm25_section_ranking,
        dense_section_ranking,
        sections,
    )
    try:
        final_ranking = _rerank_candidates(
            query,
            rrf_ranking,
            reranker_hits,
            sections,
        )
    except Exception as exc:
        logger.exception(
            "MedCPT reranker is unavailable; falling back to the RRF ranking: %s",
            exc,
        )
        _report_ranking(
            "similar_case_queries",
            "Reranker",
            query,
            [],
            cases,
            sections,
            debug=debug,
            ranking_callback=ranking_callback,
            reranker_hits=reranker_hits,
            skipped_reason=(
                "MedCPT reranker is unavailable; using the RRF ranking: "
                f"{exc}"
            ),
        )
        final_ranking = rrf_ranking[:SIMILAR_CASE_TOP_K]
    else:
        _report_ranking(
            "similar_case_queries",
            "Reranker",
            query,
            final_ranking,
            cases,
            sections,
            debug=debug,
            ranking_callback=ranking_callback,
            reranker_hits=reranker_hits,
        )

    top_indices = [record_index for record_index, _ in final_ranking]
    result_sections: list[list[dict[str, str]]] = []
    for case_index in top_indices:
        seen_section_indices: set[int] = set()
        case_sections: list[dict[str, str]] = []
        for hits in (bm25_hits, dense_hits):
            for section_index, _ in hits.get(case_index, []):
                if section_index in seen_section_indices:
                    continue
                seen_section_indices.add(section_index)
                case_sections.append(
                    {
                        "section": sections[section_index].name,
                        "content": sections[section_index].content,
                    }
                )
        result_sections.append(case_sections)
    return SimilarCaseRetrievalResult(
        discharge_disease=[
            cases[index].discharge_disease for index in top_indices
        ],
        Sections=result_sections,
    )
