from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
DEEPSEEK_THINKING = os.getenv("DEEPSEEK_THINKING", "true").strip().lower() == "true"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")

DIAGNOSIS_TOPK = int(os.getenv("DIAGNOSIS_TOPK", "5"))
DIAGNOSIS_PROVIDER = os.getenv("DIAGNOSIS_PROVIDER", "deepseek").strip().lower()
CHATKIT_TRANSLATION_MODEL = os.getenv("CHATKIT_TRANSLATION_MODEL", DEEPSEEK_MODEL)

NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_TOOL = os.getenv("NCBI_TOOL", "medical_skill_hub")
NCBI_REQUESTS_PER_SECOND = float(
    os.getenv("NCBI_REQUESTS_PER_SECOND", "10" if NCBI_API_KEY else "3")
)
NCBI_MAX_RETRIES = int(os.getenv("NCBI_MAX_RETRIES", "5"))
NCBI_RETRY_BASE_SECONDS = float(os.getenv("NCBI_RETRY_BASE_SECONDS", "0.5"))
NCBI_TIMEOUT_SECONDS = float(os.getenv("NCBI_TIMEOUT_SECONDS", "30"))

MIMIC_IV_CASE_PATH = Path(
    os.getenv(
        "MIMIC_IV_CASE_PATH",
        str(PROJECT_ROOT / "database" / "mimic_similar_case.csv"),
    )
).expanduser()
SIMILAR_CASE_TOP_K = int(os.getenv("SIMILAR_CASE_TOP_K", "5"))
SIMILAR_CASE_BM25_CANDIDATE_K = int(
    os.getenv("SIMILAR_CASE_BM25_CANDIDATE_K", "50")
)
SIMILAR_CASE_DENSE_CANDIDATE_K = int(
    os.getenv("SIMILAR_CASE_DENSE_CANDIDATE_K", "50")
)
SIMILAR_CASE_RRF_CANDIDATE_K = int(
    os.getenv("SIMILAR_CASE_RRF_CANDIDATE_K", "20")
)
SIMILAR_CASE_EMBEDDING_MODEL = os.getenv(
    "SIMILAR_CASE_EMBEDDING_MODEL",
    "BAAI/bge-m3",
)
SIMILAR_CASE_EMBEDDING_DEVICE = os.getenv(
    "SIMILAR_CASE_EMBEDDING_DEVICE",
    "cuda",
).strip().lower()
SIMILAR_CASE_EMBEDDING_CACHE_PATH = Path(
    os.getenv(
        "SIMILAR_CASE_EMBEDDING_CACHE_PATH",
        str(PROJECT_ROOT / "database" / "mimic_similar_case_embeddings.pt"),
    )
).expanduser()
SIMILAR_CASE_EMBEDDING_BATCH_SIZE = int(
    os.getenv("SIMILAR_CASE_EMBEDDING_BATCH_SIZE", "16")
)
SIMILAR_CASE_RERANKER_MODEL = os.getenv(
    "SIMILAR_CASE_RERANKER_MODEL",
    "ncbi/MedCPT-Cross-Encoder",
)
SIMILAR_CASE_RERANKER_BATCH_SIZE = int(
    os.getenv("SIMILAR_CASE_RERANKER_BATCH_SIZE", "16")
)
SIMILAR_CASE_RERANKER_DEVICE = os.getenv(
    "SIMILAR_CASE_RERANKER_DEVICE",
    "cuda",
).strip().lower()

SKILL_COMPILER_PROVIDER = os.getenv("SKILL_COMPILER_PROVIDER", "openai").lower()
SKILL_COMPILER_MODEL = os.getenv("SKILL_COMPILER_MODEL", OPENAI_MODEL)
