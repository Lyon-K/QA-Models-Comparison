import numpy as np
from rank_bm25 import BM25Okapi
from .utils import simple_tokenize


def get_chunk_text(chunk: dict) -> str:
    """
    Extract usable text from a chunk.
    Adjust this based on the actual dataset fields.
    """
    if "text" in chunk and chunk["text"] is not None:
        return str(chunk["text"])
    elif "context" in chunk and chunk["context"] is not None:
        return str(chunk["context"])
    elif "question" in chunk and "answer" in chunk:
        return f"{chunk.get('question', '')} {chunk.get('answer', '')}".strip()
    elif "query" in chunk and "answer" in chunk:
        return f"{chunk.get('query', '')} {chunk.get('answer', '')}".strip()
    else:
        raise KeyError(f"Cannot find a usable text field in chunk. Keys found: {list(chunk.keys())}")


class SparseRetriever:
    """
    BM25-based sparse retriever.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        self.corpus_texts = [get_chunk_text(chunk) for chunk in chunks]
        self.corpus_tokens = [simple_tokenize(text) for text in self.corpus_texts]
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