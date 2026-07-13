from __future__ import annotations

from typing import Any

from agents import Agent, function_tool
from langchain_community.retrievers import ArxivRetriever, PubMedRetriever

from config import OPENAI_MODEL


KNOWLEDGE_SEARCHER_INSTRUCTIONS = """
You are a clinical Knowledge search specialist in gastroenterology.

Task:
1. Read the provided patient case information.
2. Use arxiv_search and pubmed_search when external literature search is useful.
3. Search only for literature-relevant evidence, mechanisms, differential diagnosis clues, diagnostic methods, and treatment references.
4. Do not invent papers, titles, authors, URLs, or conclusions.
5. Clearly distinguish Arxiv results from PubMed results.
6. Summarize what the retrieved literature may support, and state when the retrieved results are insufficient.
7. The output is for research assistance only and cannot replace clinical diagnosis or treatment decisions.
8. Write all search summaries, extracted evidence, and conclusions in English.
""".strip()


def _normalize_max_docs(max_docs: int) -> int:
    return max(1, min(max_docs, 10))


def _document_to_result(document: Any) -> dict[str, Any]:
    metadata = getattr(document, "metadata", {}) or {}
    title = metadata.get("Title") or metadata.get("title") or metadata.get("entry_id") or ""
    source = metadata.get("source") or metadata.get("entry_id") or metadata.get("uid") or ""
    url = metadata.get("url") or metadata.get("entry_id") or metadata.get("link") or ""
    published = metadata.get("Published") or metadata.get("published") or metadata.get("pub_date") or ""
    summary = metadata.get("Summary") or metadata.get("summary") or metadata.get("abstract") or ""

    return {
        "title": str(title),
        "source": str(source),
        "url": str(url),
        "published": str(published),
        "summary": str(summary),
        "content": getattr(document, "page_content", ""),
        "metadata": metadata,
    }


def _retrieve_documents(retriever: Any, query: str) -> list[Any]:
    if hasattr(retriever, "invoke"):
        return list(retriever.invoke(query))
    return list(retriever.get_relevant_documents(query))


@function_tool
def arxiv_search(case_info: str, max_docs: int = 5) -> dict[str, Any]:
    """
    Search Arxiv literature using patient case information as the query.

    Args:
        case_info: Patient case information or a focused literature query derived from it.
        max_docs: Maximum number of documents to retrieve. The value is limited to 1-10.
    """
    normalized_max_docs = _normalize_max_docs(max_docs)
    retriever = ArxivRetriever(load_max_docs=normalized_max_docs)
    documents = _retrieve_documents(retriever, case_info)
    return {
        "source": "arxiv",
        "query": case_info,
        "results": [_document_to_result(document) for document in documents],
    }


@function_tool
def pubmed_search(case_info: str, max_docs: int = 5) -> dict[str, Any]:
    """
    Search PubMed literature using patient case information as the query.

    Args:
        case_info: Patient case information or a focused literature query derived from it.
        max_docs: Maximum number of documents to retrieve. The value is limited to 1-10.
    """
    normalized_max_docs = _normalize_max_docs(max_docs)
    retriever = PubMedRetriever(load_max_docs=normalized_max_docs)
    documents = _retrieve_documents(retriever, case_info)
    return {
        "source": "pubmed",
        "query": case_info,
        "results": [_document_to_result(document) for document in documents],
    }


def build_knowledge_searcher_agent() -> Agent:
    return Agent(
        name="Knowledge Searcher Agent",
        model=OPENAI_MODEL,
        instructions=KNOWLEDGE_SEARCHER_INSTRUCTIONS,
        tools=[arxiv_search, pubmed_search],
    )
