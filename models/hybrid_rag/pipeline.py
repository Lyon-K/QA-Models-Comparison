from .config import (
    EMBEDDING_MODEL_NAME,
    SPARSE_TOP_K,
    DENSE_TOP_K,
    FINAL_TOP_K,
    RRF_K,
    DEDUP_SIMILARITY_THRESHOLD,
    MAX_CHUNKS_PER_DOCUMENT,
)
from .sparse_retriever import SparseRetriever
from .dense_retriever import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .postprocess import deduplicate_results


class HybridRetrievalPipeline:
    """
    End-to-end hybrid retrieval pipeline:
    sparse -> dense -> fusion -> dedup -> final evidence
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.sparse_retriever = SparseRetriever(chunks)
        self.dense_retriever = DenseRetriever(chunks, EMBEDDING_MODEL_NAME)

    def retrieve(self, query: str) -> dict:
        """
        Run hybrid retrieval and return intermediate and final outputs.
        """
        sparse_results = self.sparse_retriever.search(query, top_k=SPARSE_TOP_K)
        dense_results = self.dense_retriever.search(query, top_k=DENSE_TOP_K)

        fused_results = reciprocal_rank_fusion(
            sparse_results=sparse_results,
            dense_results=dense_results,
            rrf_k=RRF_K,
        )

        final_results = deduplicate_results(
            fused_results=fused_results,
            dense_retriever=self.dense_retriever,
            similarity_threshold=DEDUP_SIMILARITY_THRESHOLD,
            max_chunks_per_document=MAX_CHUNKS_PER_DOCUMENT,
            final_top_k=FINAL_TOP_K,
        )

        return {
            "query": query,
            "sparse_results": sparse_results,
            "dense_results": dense_results,
            "fused_results": fused_results,
            "final_results": final_results,
        }