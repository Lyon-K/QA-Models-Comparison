from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _status(ok: bool, detail: str) -> dict[str, str | bool]:
    return {"ok": ok, "detail": detail}


def check_t5() -> dict[str, str | bool]:
    try:
        from models.seq2seq.T5 import T5

        model = T5()
        if model.load():
            return _status(True, "Imported and found a saved model at t5_group_aligned_runs/final_model.")
        return _status(
            False,
            "Imported successfully. Demo inference can still fall back to pretrained t5-small if needed.",
        )
    except Exception as error:
        return _status(False, f"Import/load failed: {error}")


def check_rag() -> dict[str, str | bool]:
    try:
        from backend.service import _get_rag_model

        _get_rag_model()
        return _status(True, "Imported successfully and built the in-memory FAISS index.")
    except Exception as error:
        return _status(False, f"Import/init failed: {error}")


def check_graph_rag() -> dict[str, str | bool]:
    try:
        from models.graphRAG.graphRAG import GraphRAG

        model = GraphRAG()
        model.close()
        return _status(True, "Imported successfully and connected to Neo4j.")
    except Exception as error:
        return _status(False, f"Import/init failed: {error}")


def check_no_rag() -> dict[str, str | bool]:
    try:
        from backend.service import _get_llm_client
        from models.noRag.noRag import NoRAG

        NoRAG(llm_model=_get_llm_client())
        return _status(True, "Imported successfully and created an Ollama-backed client.")
    except Exception as error:
        return _status(False, f"Import/init failed: {error}")


def check_hybrid_rag() -> dict[str, str | bool]:
    return _status(False, "hybridRAG implementation is not present in the current repository snapshot.")


def main() -> None:
    results = {
        "T5": check_t5(),
        "noRAG": check_no_rag(),
        "RAG": check_rag(),
        "graphRAG": check_graph_rag(),
        "hybridRAG": check_hybrid_rag(),
    }

    ok_count = sum(1 for result in results.values() if result["ok"])
    print(json.dumps({"summary": f"{ok_count}/5 checks passed", "models": results}, indent=2))


if __name__ == "__main__":
    main()
