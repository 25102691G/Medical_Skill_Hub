from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from agents import function_tool


MODEL_NAME = "FremyCompany/BioLORD-2023-C"
MODEL_SOURCE = os.getenv(
    "DISEASE_NORMALIZATION_MODEL",
    MODEL_NAME,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ICD11_MAPPING_PATH = PROJECT_ROOT / "database" / "icd11_id2diagnose.json"
ICD11_EMBEDDINGS_PATH = PROJECT_ROOT / "database" / "icd11_diagnose_embeddings.pt"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 36

_MODEL = None
_TOKENIZER = None
_ICD11_CODES: list[str] | None = None
_ICD11_NAMES: list[str] | None = None
_ICD11_EMBEDDINGS = None
_DEPENDENCIES = None
_DEPENDENCIES_LOCK = threading.Lock()
_MODEL_LOCK = threading.Lock()
_EMBEDDINGS_LOCK = threading.Lock()


def _require_torch_and_transformers() -> tuple[Any, Any, Any, Any]:
    global _DEPENDENCIES

    if _DEPENDENCIES is not None:
        return _DEPENDENCIES

    with _DEPENDENCIES_LOCK:
        if _DEPENDENCIES is not None:
            return _DEPENDENCIES

        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Disease normalization requires torch and transformers. "
                "Install the project dependencies with: pip install -r requirements.txt"
            ) from exc

        _DEPENDENCIES = (torch, F, AutoModel, AutoTokenizer)
        return _DEPENDENCIES


def _get_device(torch: Any) -> str:
    configured_device = os.getenv("DISEASE_NORMALIZATION_DEVICE", "").strip()
    if configured_device:
        return configured_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_icd11_mapping() -> tuple[list[str], list[str]]:
    if not ICD11_MAPPING_PATH.exists():
        raise FileNotFoundError(f"ICD11 mapping file not found: {ICD11_MAPPING_PATH}")

    with ICD11_MAPPING_PATH.open(encoding="utf-8") as f:
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

    with _MODEL_LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            return _MODEL, _TOKENIZER

        torch, _, AutoModel, AutoTokenizer = _require_torch_and_transformers()
        device = _get_device(torch)
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_SOURCE)
        _MODEL = AutoModel.from_pretrained(MODEL_SOURCE)
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


def _load_or_build_icd11_embeddings() -> tuple[list[str], Any]:
    global _ICD11_CODES, _ICD11_NAMES, _ICD11_EMBEDDINGS

    if _ICD11_CODES is not None and _ICD11_NAMES is not None and _ICD11_EMBEDDINGS is not None:
        return _ICD11_NAMES, _ICD11_EMBEDDINGS

    with _EMBEDDINGS_LOCK:
        if _ICD11_CODES is not None and _ICD11_NAMES is not None and _ICD11_EMBEDDINGS is not None:
            return _ICD11_NAMES, _ICD11_EMBEDDINGS

        torch, _, _, _ = _require_torch_and_transformers()
        codes, names = _load_icd11_mapping()

        if ICD11_EMBEDDINGS_PATH.exists():
            checkpoint = torch.load(ICD11_EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("model_name") == MODEL_NAME
                and checkpoint.get("max_length") == DEFAULT_MAX_LENGTH
                and checkpoint.get("codes") == codes
                and checkpoint.get("names") == names
            ):
                embeddings = checkpoint["embeddings"]
            else:
                embeddings = _encode_texts(names)
                torch.save(
                    {
                        "model_name": MODEL_NAME,
                        "max_length": DEFAULT_MAX_LENGTH,
                        "codes": codes,
                        "names": names,
                        "embeddings": embeddings,
                    },
                    ICD11_EMBEDDINGS_PATH,
                )
        else:
            embeddings = _encode_texts(names)
            torch.save(
                {
                    "model_name": MODEL_NAME,
                    "max_length": DEFAULT_MAX_LENGTH,
                    "codes": codes,
                    "names": names,
                    "embeddings": embeddings,
                },
                ICD11_EMBEDDINGS_PATH,
            )

        _ICD11_CODES = codes
        _ICD11_NAMES = names
        _ICD11_EMBEDDINGS = embeddings
        return names, embeddings


def normalize_disease_name_text(disease_name: str, *, debug: bool = False) -> str:
    """Normalize a disease name to the closest ICD11 diagnosis name."""
    query = disease_name.strip()
    if not query:
        if debug:
            print(
                "[Disease Normalization] Original Disease Name: "
                f"{disease_name} -> Normalized Disease Name: ",
                file=sys.stderr,
            )
        return ""

    torch, F, _, _ = _require_torch_and_transformers()
    names, disease_embeddings = _load_or_build_icd11_embeddings()

    query_embedding = _encode_texts([query])
    normalized_query_embedding = F.normalize(query_embedding, p=2, dim=1)
    normalized_disease_embeddings = F.normalize(disease_embeddings, p=2, dim=1)
    similarities = torch.matmul(normalized_disease_embeddings, normalized_query_embedding[0])
    index = int(torch.topk(similarities, 1, largest=True).indices[0].item())
    normalized_name = names[index]
    if debug:
        print(
            "[Disease Normalization] Original Disease Name: "
            f"{query} -> Normalized Disease Name: {normalized_name}",
            file=sys.stderr,
        )
    return normalized_name


@function_tool
def normalize_disease_name(disease_name: str) -> str:
    """
    Normalize a disease name to the closest ICD11 diagnosis name.

    Args:
        disease_name: Disease name or diagnosis text to normalize.
    """
    return normalize_disease_name_text(disease_name)
