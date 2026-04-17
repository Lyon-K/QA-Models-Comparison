import numpy as np
from rank_bm25 import BM25Okapi
from .utils import simple_tokenize


class SparseRetriever:
    """
    BM25-based sparse retriever.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.corpus_tokens = [simple_tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Return top_k chunk results ranked by BM25.
        """
        query_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[idx]),
                    "retrieval_type": "sparse",
                    "chunk": self.chunks[idx],
                }
            )

        return results