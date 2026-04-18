from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def main() -> None:
    from backend.service import _get_embedding_model, _get_llm_client, _run_graph_rag
    from models.graphRAG.graphRAG import GraphRAG
    import os
    print("NEO4J_URI:", os.getenv("NEO4J_URI"))
    print("NEO4J_USER:", os.getenv("NEO4J_USER"))
    print("NEO4J_PASSWORD:", os.getenv("NEO4J_PASSWORD"))
    print("NEO4J_DATABASE:", os.getenv("NEO4J_DATABASE"))

    query = "Vaccines cause infertility"
    neo4j_uri = (os.getenv("NEO4J_URI") or "").strip() or "not set"
    diagnostics = {
        "query": query,
        "neo4j_uri": neo4j_uri,
        "retrieved_context_count": 0,
        "raw_answer": "",
        "answer": "",
    }
    if neo4j_uri.startswith("neo4j://"):
        diagnostics["neo4j_note"] = (
            "For a local single-instance Neo4j setup, bolt://localhost:7687 is often more reliable than neo4j://."
        )
    model = None
    try:
        model = GraphRAG(
            embedding_model=_get_embedding_model(),
            llm_model=_get_llm_client(),
        )
        context, raw_answer = model.predict(query)
        diagnostics["retrieved_context_count"] = len(context)
        diagnostics["raw_answer"] = raw_answer
    except Exception as error:
        diagnostics["error"] = str(error)
    finally:
        if model is not None:
            try:
                model.close()
            except Exception as close_error:
                diagnostics["close_error"] = str(close_error)

    try:
        diagnostics["answer"] = _run_graph_rag(query)
    except Exception as wrapper_error:
        diagnostics["wrapper_error"] = str(wrapper_error)

    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
