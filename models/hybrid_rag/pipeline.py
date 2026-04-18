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

    def __init__(self, chunks: list[dict], llm_model=None):
        self.llm_model = llm_model

        # 👇 新增代码：如果外部没传 chunk_id，流水线自己生成一个
        for i, chunk in enumerate(chunks):
            if "chunk_id" not in chunk:
                # 用它在列表中的索引位置作为唯一的 ID，必须转为字符串 str()
                chunk["chunk_id"] = str(i)
            
            # 2. 👇 新增：注入 Doc ID (Document ID)
            # 让文档 ID 和块 ID 保持一致，这样去重逻辑就不会误删数据
            if "doc_id" not in chunk:
                chunk["doc_id"] = chunk["chunk_id"]

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
    
    def predict(self, query: str):
        """
        补充的端到端生成接口 (End-to-End Generation Interface)
        """
        if not self.llm_model:
            raise ValueError("LLM model is not initialized. Please pass llm_model to __init__.")

        # 1. 执行混合检索 (Hybrid Retrieval)
        retrieval_output = self.retrieve(query)
        final_docs = retrieval_output["final_results"]
        
        # 2. 拼接上下文 (Context Concatenation)
        contexts = [doc["chunk"]["context"] for doc in final_docs]
        context_str = "\n---\n".join(contexts)
        
        # 3. 构建提示词 (Prompt Construction) - 保持和原作者一模一样的 Prompt 风格
        prompt = (
            f"**context(only use when it is relevant to the prompt given)**:\n{context_str}\n\n**prompt**:\n{query}"
            if context_str
            else query
        )
        
        # 4. 调用大模型生成 (LLM Generation) - 完全照抄原作者的 API 格式
        try:
            response = self.llm_model.chat(
                model="ministral-3:8b-cloud",  # 这里你可以换成你要测试的模型名
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.message.content # 重点：直接取 message.content
        except Exception as e:
            answer = f"Error generating answer: {e}"
            
        return context_str, answer