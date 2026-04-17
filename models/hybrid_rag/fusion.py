from collections import defaultdict


def reciprocal_rank_fusion(
    sparse_results: list[dict],
    dense_results: list[dict],
    rrf_k: int = 60,
) -> list[dict]:
    """
    Combine sparse and dense results using Reciprocal Rank Fusion (RRF).
    """
    fused_scores = defaultdict(float)
    chunk_lookup = {}

    for result in sparse_results:
        chunk_id = result["chunk"]["chunk_id"]
        fused_scores[chunk_id] += 1.0 / (rrf_k + result["rank"])
        chunk_lookup[chunk_id] = result["chunk"]

    for result in dense_results:
        chunk_id = result["chunk"]["chunk_id"]
        fused_scores[chunk_id] += 1.0 / (rrf_k + result["rank"])
        chunk_lookup[chunk_id] = result["chunk"]

    fused_results = []
    for chunk_id, score in fused_scores.items():
        fused_results.append(
            {
                "chunk_id": chunk_id,
                "fused_score": score,
                "chunk": chunk_lookup[chunk_id],
            }
        )

    fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused_results