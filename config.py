from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
DIAGNOSIS_TOPK = int(os.getenv("DIAGNOSIS_TOPK", "5"))

