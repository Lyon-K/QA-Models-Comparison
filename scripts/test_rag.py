from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def main() -> None:
    from backend.service import _get_llm_client, _get_rag_model, _run_rag

    query = "Vaccines cause infertility"
    client = _get_llm_client()
    model = _get_rag_model()
    context, _ = model.predict(query)
    answer = _run_rag(query)

    print(
        json.dumps(
            {
                "query": query,
                "ollama_model": getattr(model, "chat_model_name", "unknown"),
                "ollama_host": getattr(client, "_host", "unknown"),
                "retrieved_context_count": len(context),
                "answer": answer,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
