from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def main() -> None:
    from backend.service import _get_llm_client, _run_no_rag
    from models.noRag.noRag import NoRAG

    query = "Vaccines cause infertility"
    client = _get_llm_client()
    model = NoRAG(llm_model=client)
    _, raw_answer = model.predict(query)
    answer = _run_no_rag(query)

    print(
        json.dumps(
            {
                "query": query,
                "ollama_host": getattr(client, "_host", "unknown"),
                "raw_answer": raw_answer,
                "answer": answer,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
