from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Callable
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

LOCAL_KB_PATH = ROOT_DIR / "data" / "test_dataset.csv"
UNAVAILABLE_MESSAGE = "This model is not enabled in the current demo environment"
EMPTY_OUTPUT_MESSAGE = "No explanation could be generated for this query."


def _format_error(model_name: str, error: Exception) -> str:
    del model_name, error
    return UNAVAILABLE_MESSAGE


def _normalize_output(value) -> str:
    import re

    if value is None:
        return EMPTY_OUTPUT_MESSAGE

    text = str(value).strip()
    if not text or text.lower() == "none":
        return EMPTY_OUTPUT_MESSAGE

    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?im)^\s*(answer|response|summary|key points)\s*:\s*", "", text)
    text = re.sub(r"(?im)^(based on general medical knowledge:)(\s*\1)+", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = text.strip()

    return text


def _extract_model_text(value) -> str:
    if isinstance(value, dict):
        for key in ("answer", "response", "result"):
            extracted = value.get(key)
            if extracted is not None:
                return _normalize_output(extracted)
        message = value.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if content is not None:
                return _normalize_output(content)
    return _normalize_output(value)


def _get_neo4j_uri_diagnostics() -> tuple[str, str | None]:
    uri = (os.getenv("NEO4J_URI") or "").strip()
    if not uri:
        return "not set", None
    note = None
    if uri.startswith("neo4j://"):
        note = "Note: for a local single-instance Neo4j setup, bolt://localhost:7687 is often more reliable than neo4j://."
    return uri, note


def _split_sentences(text: str) -> list[str]:
    import re

    normalized = " ".join(str(text).replace("\n", " ").split()).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _extract_bullets(lines: list[str]) -> list[str]:
    bullets: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("-", "*")):
            bullet = stripped[1:].strip()
            if bullet:
                bullets.append(bullet)
    return bullets


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item.strip())
    return result


def format_output(raw_text: str) -> str:
    text = _normalize_output(raw_text)
    if text == EMPTY_OUTPUT_MESSAGE:
        return text

    if text == UNAVAILABLE_MESSAGE:
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullets = _extract_bullets(lines)
    sentences = _split_sentences(text)

    summary_sentences = sentences[:3]
    if not summary_sentences and text:
        summary_sentences = [text]
    summary = " ".join(summary_sentences).strip()
    summary = _normalize_output(summary)

    if not bullets:
        remaining_sentences = sentences[3:] if len(sentences) > 3 else []
        if not remaining_sentences and len(sentences) > 1:
            remaining_sentences = sentences[1:]

        bullets = []
        for sentence in remaining_sentences:
            cleaned = sentence.strip(" -")
            if cleaned:
                bullets.append(cleaned)
            if len(bullets) >= 3:
                break

        if len(bullets) < 3:
            for sentence in sentences:
                cleaned = sentence.strip(" -")
                if cleaned:
                    bullets.append(cleaned)
                if len(bullets) >= 3:
                    break

    bullets = _dedupe_preserve_order(bullets)[:3]

    additional_info = ""
    lower_text = text.lower()
    if "based on general medical knowledge:" in lower_text:
        additional_info = "Based on general medical knowledge."
    elif "insufficient" in lower_text or "not enough" in lower_text:
        additional_info = "The available information may be limited."

    sections = [f"Summary:\n{summary or EMPTY_OUTPUT_MESSAGE}"]

    if bullets:
        bullet_block = "\n".join(f"- {bullet}" for bullet in bullets)
        sections.append(f"Key Points:\n{bullet_block}")
    else:
        sections.append("Key Points:\n- No key points available.")

    if additional_info:
        sections.append(f"Additional Info:\n{additional_info}")

    return "\n\n".join(sections)


def _clean_hybrid_rag_output(context: str, answer: str) -> str:
    import re

    normalized_context = str(context or "").strip()
    normalized_answer = _normalize_output(answer)
    weak_context = len(normalized_context) < 100
    disclaimer_removed = False

    disclaimer_pattern = re.compile(
        r"^\s*The retrieved context does not (?:provide|include) information about .*?[.!?]\s*",
        flags=re.IGNORECASE,
    )
    if disclaimer_pattern.search(normalized_answer):
        normalized_answer = disclaimer_pattern.sub("", normalized_answer, count=1).strip()
        disclaimer_removed = True

    provided_context_pattern = re.compile(
        r"^\s*Based on the provided context,?\s*I cannot (?:give|provide) .*?[.!?]\s*",
        flags=re.IGNORECASE,
    )
    if provided_context_pattern.search(normalized_answer):
        normalized_answer = provided_context_pattern.sub("", normalized_answer, count=1).strip()
        disclaimer_removed = True

    if weak_context or disclaimer_removed:
        normalized_answer = normalized_answer.lstrip(" ,:;.-")

        if normalized_answer and not normalized_answer.lower().startswith("while the retrieved context is limited"):
            normalized_answer = f"While the retrieved context is limited, here is a general explanation:\n\n{normalized_answer}"
        elif not normalized_answer:
            normalized_answer = "While the retrieved context is limited, here is a general explanation:\n\nNo explanation could be generated for this query."

    normalized_answer = normalized_answer.replace(
        "While the retrieved context is limited, here is a general explanation:\n\nWhile the retrieved context is limited, here is a general explanation:\n\n",
        "While the retrieved context is limited, here is a general explanation:\n\n",
    )
    return _normalize_output(normalized_answer)


def _safe_run(model_name: str, runner: Callable[[str], str], query: str) -> str:
    try:
        return _normalize_output(runner(query))
    except Exception as error:
        return _format_error(model_name, error)


@lru_cache(maxsize=1)
def _get_t5_model():
    module = import_module("models.T5.T5")
    model_class = getattr(module, "T5")
    return model_class()


def _run_t5(query: str) -> str:
    raw_output = _get_t5_model().predict(query)
    return format_output(raw_output)


@lru_cache(maxsize=1)
def _get_llm_client():
    from ollama import Client

    api_key = (os.getenv("OLLAMA_API_KEY") or "").strip()
    host = (os.getenv("OLLAMA_HOST") or "https://ollama.com").strip()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    if headers:
        return Client(host=host, headers=headers)
    return Client(host=host)


@lru_cache(maxsize=1)
def _get_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={
            "prompt": "Represent this sentence for searching relevant passages: "
        },
    )


def _load_rag_training_chunks() -> list[str]:
    import csv

    if not LOCAL_KB_PATH.exists():
        raise FileNotFoundError(f"Missing RAG knowledge base: {LOCAL_KB_PATH}")

    with LOCAL_KB_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    return [
        (
            f"Topic: {row.get('Topic', '')}\n"
            f"Question: {row.get('Question', '')}\n"
            f"Answer: {row.get('Answer', '')}"
        )
        for row in rows
    ]


def _load_hybrid_documents() -> list[dict]:
    import csv

    if not LOCAL_KB_PATH.exists():
        raise FileNotFoundError(f"Missing hybridRAG knowledge base: {LOCAL_KB_PATH}")

    with LOCAL_KB_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    documents = []
    for index, row in enumerate(rows):
        topic = row.get("Topic", "")
        question = row.get("Question", "")
        answer = row.get("Answer", "")
        source = row.get("Source", "local_csv")
        documents.append(
            {
                "doc_id": f"doc_{index}",
                "title": topic or question or f"Document {index}",
                "source": source,
                "text": f"Topic: {topic}\nQuestion: {question}\nAnswer: {answer}".strip(),
            }
        )

    return documents


@lru_cache(maxsize=1)
def _get_rag_model():
    from models.RAG.RAG import VectorRAG

    model = VectorRAG(
        embedding_model=_get_embedding_model(),
        llm_model=_get_llm_client(),
    )

    loaded = False
    if hasattr(model, "load"):
        loaded = model.load()
    if loaded is False:
        model.train(train=_load_rag_training_chunks())
    return model


def _run_rag(query: str) -> str:
    rag_model = _get_rag_model()
    _, answer = rag_model.predict(query)
    return format_output(answer)


def _run_graph_rag(query: str) -> str:
    from models.graphRAG.graphRAG import GraphRAG

    model = None
    try:
        model = GraphRAG(
            embedding_model=_get_embedding_model(),
            llm_model=_get_llm_client(),
        )
        _, answer = model.predict(query)
        return format_output(_extract_model_text(answer))
    except Exception as e:
        uri, note = _get_neo4j_uri_diagnostics()
        message = f"graphRAG error: {str(e)} Effective NEO4J_URI: {uri}."
        if note:
            message += f" {note}"
        return format_output(message)
    finally:
        if model is not None:
            try:
                model.close()
            except Exception:
                pass


def _run_no_rag(query: str) -> str:
    from models.noRag.noRag import NoRAG

    model = NoRAG(llm_model=_get_llm_client())
    _, answer = model.predict(query)
    return format_output(answer)


@lru_cache(maxsize=1)
def _get_hybrid_rag_pipeline():
    from models.hybrid_rag.chunker import build_chunks
    from models.hybrid_rag.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS
    from models.hybrid_rag.pipeline import HybridRetrievalPipeline

    documents = _load_hybrid_documents()
    chunks = build_chunks(
        documents=documents,
        chunk_size=CHUNK_SIZE_WORDS,
        overlap=CHUNK_OVERLAP_WORDS,
    )
    return HybridRetrievalPipeline(chunks=chunks, llm_model=_get_llm_client())


def _run_hybrid_rag(query: str) -> str:
    pipeline = _get_hybrid_rag_pipeline()
    context, answer = pipeline.predict(query)
    return format_output(_clean_hybrid_rag_output(context, answer))


def get_fact_check_result(claim: str, source_filter: str = "All sources") -> dict[str, str]:
    del source_filter
    query = claim.strip()

    return {
        "T5": _safe_run("T5", _run_t5, query),
        "noRAG": _safe_run("noRAG", _run_no_rag, query),
        "RAG": _safe_run("RAG", _run_rag, query),
        "graphRAG": _safe_run("graphRAG", _run_graph_rag, query),
        "hybridRAG": _safe_run("hybridRAG", _run_hybrid_rag, query),
    }
