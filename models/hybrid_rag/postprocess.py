import numpy as np
from collections import defaultdict


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two normalized vectors."""
    return float(np.dot(vec1, vec2))


def deduplicate_results(
    fused_results: list[dict],
    dense_retriever,
    similarity_threshold: float = 0.90,
    max_chunks_per_document: int = 2,
    final_top_k: int = 5,
) -> list[dict]:
    """
    Remove near-duplicate chunks and control document-level redundancy.

    Rules:
    1. Skip chunks that are too similar to already selected chunks.
    2. Limit how many chunks can come from the same source document.
    """
    selected = []
    selected_embeddings = []
    doc_counter = defaultdict(int)

    embedding_lookup = {
        chunk["chunk_id"]: dense_retriever.chunk_embeddings[idx]
        for idx, chunk in enumerate(dense_retriever.chunks)
    }

    for item in fused_results:
        chunk = item["chunk"]
        chunk_id = chunk["chunk_id"]
        doc_id = chunk["doc_id"]

        if doc_counter[doc_id] >= max_chunks_per_document:
            continue

        current_embedding = embedding_lookup[chunk_id]

        is_duplicate = False
        for existing_embedding in selected_embeddings:
            sim = cosine_similarity(current_embedding, existing_embedding)
            if sim >= similarity_threshold:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        selected.append(item)
        selected_embeddings.append(current_embedding)
        doc_counter[doc_id] += 1

        if len(selected) >= final_top_k:
            break

    return selected