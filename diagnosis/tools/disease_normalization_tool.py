from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agents import function_tool


MODEL_NAME = "FremyCompany/BioLORD-2023-C"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ICD10_MAPPING_PATH = PROJECT_ROOT / "database" / "icd10_id2diagnose.json"
ICD10_EMBEDDINGS_PATH = PROJECT_ROOT / "database" / "icd10_diagnose_embeddings.pt"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 128

_MODEL = None
_TOKENIZER = None
_ICD10_CODES: list[str] | None = None
_ICD10_NAMES: list[str] | None = None
_ICD10_EMBEDDINGS = None


def _require_torch_and_transformers() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Disease normalization requires torch and transformers. "
            "Install the project dependencies with: pip install -r requirements.txt"
        ) from exc

    return torch, F, AutoModel, AutoTokenizer


def _get_device(torch: Any) -> str:
    configured_device = os.getenv("DISEASE_NORMALIZATION_DEVICE", "").strip()
    if configured_device:
        return configured_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_icd10_mapping() -> tuple[list[str], list[str]]:
    if not ICD10_MAPPING_PATH.exists():
        raise FileNotFoundError(f"ICD10 mapping file not found: {ICD10_MAPPING_PATH}")

    with ICD10_MAPPING_PATH.open(encoding="utf-8") as f:
        mapping = json.load(f)

    codes: list[str] = []
    names: list[str] = []
    for code, name in mapping.items():
        code_text = str(code).strip()
        name_text = str(name).strip()
        if code_text and name_text:
            codes.append(code_text)
            names.append(name_text)
    return codes, names


def _load_model_and_tokenizer() -> tuple[Any, Any]:
    global _MODEL, _TOKENIZER

    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    torch, _, AutoModel, AutoTokenizer = _require_torch_and_transformers()
    device = _get_device(torch)
    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    _MODEL = AutoModel.from_pretrained(MODEL_NAME)
    _MODEL.to(device)
    _MODEL.eval()
    return _MODEL, _TOKENIZER


def _encode_texts(texts: list[str], *, batch_size: int | None = None) -> Any:
    torch, _, _, _ = _require_torch_and_transformers()
    model, tokenizer = _load_model_and_tokenizer()
    device = _get_device(torch)
    normalized_batch_size = batch_size or int(
        os.getenv("DISEASE_NORMALIZATION_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))
    )

    embeddings = []
    for start in range(0, len(texts), normalized_batch_size):
        batch = texts[start : start + normalized_batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())

    return torch.cat(embeddings, dim=0)


def _load_or_build_icd10_embeddings() -> tuple[list[str], list[str], Any]:
    global _ICD10_CODES, _ICD10_NAMES, _ICD10_EMBEDDINGS

    if _ICD10_CODES is not None and _ICD10_NAMES is not None and _ICD10_EMBEDDINGS is not None:
        return _ICD10_CODES, _ICD10_NAMES, _ICD10_EMBEDDINGS

    torch, _, _, _ = _require_torch_and_transformers()
    codes, names = _load_icd10_mapping()

    if ICD10_EMBEDDINGS_PATH.exists():
        checkpoint = torch.load(ICD10_EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
        if (
            isinstance(checkpoint, dict)
            and checkpoint.get("model_name") == MODEL_NAME
            and checkpoint.get("codes") == codes
            and checkpoint.get("names") == names
        ):
            embeddings = checkpoint["embeddings"]
        else:
            embeddings = _encode_texts(names)
            torch.save(
                {
                    "model_name": MODEL_NAME,
                    "codes": codes,
                    "names": names,
                    "embeddings": embeddings,
                },
                ICD10_EMBEDDINGS_PATH,
            )
    else:
        embeddings = _encode_texts(names)
        torch.save(
            {
                "model_name": MODEL_NAME,
                "codes": codes,
                "names": names,
                "embeddings": embeddings,
            },
            ICD10_EMBEDDINGS_PATH,
        )

    _ICD10_CODES = codes
    _ICD10_NAMES = names
    _ICD10_EMBEDDINGS = embeddings
    return codes, names, embeddings


def _normalize_top_k(top_k: int, total: int) -> int:
    return max(1, min(int(top_k), total))


@function_tool
def normalize_disease_name(disease_name: str, top_k: int = 10) -> dict[str, Any]:
    """
    Normalize a disease name to the closest ICD10 diagnosis names.

    Args:
        disease_name: Disease name or diagnosis text to normalize.
        top_k: Number of ICD10 candidates to return. The value is limited to the
            available ICD10 diagnosis count.
    """
    query = disease_name.strip()
    if not query:
        return {"query": disease_name, "results": []}

    torch, F, _, _ = _require_torch_and_transformers()
    codes, names, disease_embeddings = _load_or_build_icd10_embeddings()
    normalized_top_k = _normalize_top_k(top_k, len(names))

    query_embedding = _encode_texts([query])
    normalized_query_embedding = F.normalize(query_embedding, p=2, dim=1)
    normalized_disease_embeddings = F.normalize(disease_embeddings, p=2, dim=1)
    similarities = torch.matmul(normalized_disease_embeddings, normalized_query_embedding[0])
    values, indices = torch.topk(similarities, normalized_top_k, largest=True)

    results = []
    for value, index in zip(values.tolist(), indices.tolist(), strict=True):
        results.append(
            {
                "icd10_code": codes[index],
                "diagnose_name": names[index],
                "similarity": float(value),
            }
        )

    return {
        "query": query,
        "model": MODEL_NAME,
        "mapping_path": str(ICD10_MAPPING_PATH),
        "embeddings_path": str(ICD10_EMBEDDINGS_PATH),
        "results": results,
    }
