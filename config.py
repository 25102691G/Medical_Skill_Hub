from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
DIAGNOSIS_TOPK = int(os.getenv("DIAGNOSIS_TOPK", "5"))

SKILL_COMPILER_PROVIDER = os.getenv("SKILL_COMPILER_PROVIDER", "openai").lower()
SKILL_COMPILER_MODEL = os.getenv("SKILL_COMPILER_MODEL", OPENAI_MODEL)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

