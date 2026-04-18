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


def get_chunk_text(chunk: dict) -> str:
    """
    Extract usable text from a chunk.
    """
    if "text" in chunk and chunk["text"] is not None:
        return str(chunk["text"])
    elif "context" in chunk and chunk["context"] is not None:
        return str(chunk["context"])
    elif "question" in chunk and "answer" in chunk:
        return f"Q: {chunk.get('question', '')}\nA: {chunk.get('answer', '')}".strip()
    elif "query" in chunk and "answer" in chunk:
        return f"Q: {chunk.get('query', '')}\nA: {chunk.get('answer', '')}".strip()
    else:
        return str(chunk)


class HybridRetrievalPipeline:
    """
    End-to-end hybrid retrieval pipeline:
    sparse -> dense -> fusion -> dedup -> final evidence
    """

    def __init__(self, chunks: list[dict], llm_model=None):
        self.chunks = chunks
        self.llm_model = llm_model
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
    
    def predict(self, query: str):
        """
        Retrieve evidence first, then use the LLM to generate the final answer.
        Returns:
            context: concatenated retrieved evidence
            answer: LLM-generated answer
        """
        retrieval_output = self.retrieve(query)
        final_results = retrieval_output["final_results"]

        context_parts = []
        for item in final_results:
            chunk = item.get("chunk", {})
            context_parts.append(get_chunk_text(chunk))

        context = "\n\n".join(context_parts)

        # If no LLM was passed in, return a fallback answer
        if self.llm_model is None:
            if context_parts:
                answer = context_parts[0]
            else:
                answer = "No relevant information found."
            return context, answer

        # Build a prompt for RAG
        prompt = f"""You are a helpful assistant for public health question answering.

Use the retrieved context below to answer the user's question.
Answer as clearly and accurately as possible.
If the context is insufficient, say so instead of making up facts.

Retrieved Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm_model.chat(
            # model="ministral-3:3b-cloud",
            model="ministral-3:8b-cloud",
            # model="ministral-3:14b-cloud",
            # model="gpt-oss:20b-cloud",
            # model="gpt-oss:120b-cloud",
            # model="mistral-large-3:675b-cloud",
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.message.content

        return context, answer

