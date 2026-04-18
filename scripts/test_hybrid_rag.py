from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def main() -> None:
    from backend.service import _get_hybrid_rag_pipeline, _run_hybrid_rag

    query = "Vaccines cause infertility"
    pipeline = _get_hybrid_rag_pipeline()
    context, raw_answer = pipeline.predict(query)
    answer = _run_hybrid_rag(query)

    print(
        json.dumps(
            {
                "query": query,
                "retrieved_context_length": len(context),
                "raw_answer": raw_answer,
                "answer": answer,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
