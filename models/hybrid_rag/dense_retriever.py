import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def get_chunk_text(chunk: dict) -> str:
    """
    Extract usable text from a chunk.
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
        raise KeyError(
            f"Cannot find a usable text field in chunk. Keys found: {list(chunk.keys())}"
        )


class DenseRetriever:
    """
    Local offline retriever using TF-IDF vectors
    and cosine similarity.

    This class keeps the same interface as the original dense retriever
    so the rest of the pipeline does not need to change.
    """

    def __init__(self, chunks: list[dict], model_name: str):
        """
        Initialize the retriever.

        Parameters:
        - chunks: a list of chunk dictionaries
        - model_name: kept only for interface compatibility, not used here
        """
        self.chunks = chunks
        self.model_name = model_name

        # Extract chunk texts safely
        self.chunk_texts = [get_chunk_text(chunk) for chunk in chunks]

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        # Fit and transform all chunk texts
        self.chunk_matrix = self.vectorizer.fit_transform(self.chunk_texts)

        # Convert to dense numpy array for compatibility with postprocess.py
        self.chunk_embeddings = self.chunk_matrix.toarray().astype(np.float32)

        # Normalize vectors so that dot product behaves like cosine similarity
        norms = np.linalg.norm(self.chunk_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.chunk_embeddings = self.chunk_embeddings / norms

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Return top_k chunk results ranked by cosine similarity.
        """
        query_matrix = self.vectorizer.transform([query])
        query_vector = query_matrix.toarray().astype(np.float32)[0]

        # Normalize query vector
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            norm = 1.0
        query_vector = query_vector / norm

        # Cosine similarity via dot product on normalized vectors
        scores = np.dot(self.chunk_embeddings, query_vector)

        # Get top indices sorted from high to low
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[idx]),
                    "retrieval_type": "dense",
                    "chunk": self.chunks[idx],
                }
            )

        return results