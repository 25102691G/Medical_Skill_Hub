from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_THINKING = True
DEEPSEEK_REASONING_EFFORT = "high"

OPENAI_MODEL = "gpt-5.5"

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8000/v1")
QWEN_THINKING = False

DIAGNOSIS_MODELS = {
    "openai-gpt-5.5": ("openai", OPENAI_MODEL),
    "deepseek-v4-pro": ("deepseek", DEEPSEEK_MODEL),
    "qwen3.5-27b": ("qwen", "Qwen3.5-27B"),
    "qwen3.5-122b-a10b": ("qwen", "Qwen3.5-122B-A10B"),
}
DIAGNOSIS_PROVIDER = os.getenv(
    "DIAGNOSIS_PROVIDER", "deepseek-v4-pro"
).strip().lower()
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
        str(PROJECT_ROOT / "database" / "mimic_similar.csv"),
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
        str(PROJECT_ROOT / "database" / "mimic_similar_embeddings.pt"),
    )
).expanduser()
SIMILAR_CASE_EMBEDDING_BATCH_SIZE = int(
    os.getenv("SIMILAR_CASE_EMBEDDING_BATCH_SIZE", "16")
)
SKILL_COMPILER_PROVIDER = os.getenv("SKILL_COMPILER_PROVIDER", "openai").lower()
SKILL_COMPILER_MODEL = os.getenv("SKILL_COMPILER_MODEL", OPENAI_MODEL)
