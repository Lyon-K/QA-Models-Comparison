import os
from typing import Optional, List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pandas as pd

class VectorRAG:
    embedding_model: HuggingFaceEmbeddings
    vector_store: FAISS = None
    model = None

    def __init__(
        self, embedding_model: Optional[HuggingFaceEmbeddings] = None, llm_model=None
    ):
        # 初始化词嵌入模型 (Embedding Model)
        if embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
                query_encode_kwargs={
                    "prompt": "Represent this sentence for searching relevant passages: "
                },
            )
        else:
            self.embedding_model = embedding_model
        
        # 初始化大语言模型 (Large Language Model, LLM)
        self.model = llm_model

    def build_index(self, chunks: List[str]):
        """
        将文本块 (Chunks) 向量化并存入向量数据库中。
        """
        # 使用 FAISS 从文本列表构建本地向量索引 (Vector Index)
        self.vector_store = FAISS.from_texts(chunks, self.embedding_model)

    def predict(self, query: str) -> Dict[str, Any]:
        """
        执行 RAG 流程：检索上下文并生成回答。
        返回包含 'answer', 'retrieved_chunks' 的字典。
        """
        # 1. 获取检索到的上下文、以及对应的相似度
        context_records = self._rag_retrieval(query)

        # 2. 将检索到的 Chunk 拼接进 Prompt 中
        context_str = "\n\n".join(
            [f"【Chunk】: {rec['chunk']}\n【Similarity】: {rec['similarity']:.4f}" 
             for rec in context_records]
        )

        prompt = f"""**context**:
{context_str}

**prompt**:
{query}
"""
        
        # 3. 调用生成模型
        if self.model:
            # 假设 model 具有 generate_chat 方法 (与你提供的格式一致)
            answer = self.model.generate_chat(
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            # 如果未加载 LLM 模型，直接将 Prompt 作为 mock 结果返回，方便调试检索效果
            answer = "LLM not loaded. Prompt preview:\n" + prompt

        # 4. 结构化输出：同时返回生成的答案以及检索的 Chunk 和 Similarity
        return {
            "answer": answer,
            "retrieved_context": context_records
        }

    def load(self, llm_model, **kwargs):
        if llm_model:
            self.model = llm_model
            return True
        else:
            return False

    def _rag_retrieval(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        执行向量检索，返回包含 chunk 和 similarity 的字典列表。
        """
        if not self.vector_store:
            raise ValueError("Vector DB is empty. Please call `build_index` first.")

        # 使用带有相关性分数的相似度搜索 (Similarity Search with Relevance Scores)
        # BGE 模型使用了 normalize_embeddings=True，这里的分数通常反映余弦相似度 (Cosine Similarity)
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)

        context = []
        for doc, score in results:
            context.append({
                "chunk": doc.page_content,
                "similarity": score
            })
            
        return context


if __name__ == "__main__":
    # --- 测试流程 ---
    
    # 1. 实例化 RAG (如果之前没加载 LLM，这里依旧默认不用 LLM)
    rag = VectorRAG()

    # 2. 读取上传的 CSV 数据集 (Dataset)
    print("正在加载数据集 test_dataset.csv ...")
    df = pd.read_csv('C:/Users/liudo/Desktop/NTU_learning/EE6405/group work/test_dataset.csv')

    # 3. 将表格数据转换为文本块 (Chunks)
    # 我们将 Topic, Question, Answer 和 Source 拼接在一起，这样向量检索时能捕捉到所有维度的语义
    chunks = []
    for index, row in df.iterrows():
        chunk_text = (
            f"Topic (主题): {row['Topic']}\n"
            f"Question (问题): {row['Question']}\n"
            f"Answer (答案): {row['Answer']}\n"
            f"Source (来源): {row['Source']}"
        )
        chunks.append(chunk_text)
    
    # 4. 构建索引 (Build Index)
    print(f"正在为 {len(chunks)} 条数据构建向量索引...")
    rag.build_index(chunks)
    print("索引构建完成！\n" + "-"*50)

    # 5. 测试查询 (根据你数据集的内容，我们可以问一个关于压力管理的问题)
    query = "How can I manage daily stress?"
    print(f"Query: {query}\n" + "-"*50)

    # 6. 调用 predict 方法获取结果
    response = rag.predict(query)

    # 7. 打印输出结构：Chunk, Similarity 以及 LLM 回答
    print("=== 检索到的 Chunks & 相似度 (Similarities) ===")
    for i, record in enumerate(response["retrieved_context"]):
        print(f"Rank {i+1}:")
        print(f"  Similarity : {record['similarity']:.4f}")
        print(f"  Chunk      :\n{record['chunk']}")
        print("-" * 30)

    print("\n=== LLM Answer ===")
    print(response["answer"])