import re
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    """Create the directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove extra line breaks."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25.
    This simple tokenizer lowercases text and extracts alphanumeric tokens.
    """
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def save_json(data: Any, file_path: Path) -> None:
    """Save Python data as a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)