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
    if value is None:
        return EMPTY_OUTPUT_MESSAGE

    text = str(value).strip()
    if not text or text.lower() == "none":
        return EMPTY_OUTPUT_MESSAGE

    return text


def _safe_run(model_name: str, runner: Callable[[str], str], query: str) -> str:
    try:
        return _normalize_output(runner(query))
    except Exception as error:
        return _format_error(model_name, error)


@lru_cache(maxsize=1)
def _get_t5_model():
    module = import_module("models.seq2seq.T5")
    model_class = getattr(module, "T5")
    return model_class()


def _run_t5(query: str) -> str:
    raw_output = _get_t5_model().predict(query)
    return _normalize_output(raw_output)


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
    return _normalize_output(answer)


def _run_graph_rag(query: str) -> str:
    from models.graphRAG.graphRAG import GraphRAG

    model = GraphRAG(
        embedding_model=_get_embedding_model(),
        llm_model=_get_llm_client(),
    )
    try:
        _, answer = model.predict(query)
        return _normalize_output(answer)
    finally:
        model.close()


def _run_no_rag(query: str) -> str:
    from models.noRag.noRag import NoRAG

    model = NoRAG(llm_model=_get_llm_client())
    _, answer = model.predict(query)
    return _normalize_output(answer)


def _run_hybrid_rag(query: str) -> str:
    del query
    return UNAVAILABLE_MESSAGE


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
