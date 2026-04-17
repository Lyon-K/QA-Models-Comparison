import os
from typing import Optional, List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pandas as pd
from ollama import Client


class VectorRAG:
    embedding_model: HuggingFaceEmbeddings
    vector_store: FAISS = None
    model: Optional[Client] = None

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

    def train(self, train_x, train_y=None, val_x=None, val_y=None):
        """
        在 RAG 的语境下，'train' 本质上就是构建知识库索引 (Index Building)。
        为了兼容统一的评测流水线，我们将原本的建库逻辑封装在 train 方法中。
        """
        print("⏳ 正在提取文本并构建 FAISS 向量数据库索引 (Building Vector Index)...")
        
        chunks = []
        # 兼容性处理：判断传进来的 train_x 是字典列表还是纯字符串列表
        for item in train_x:
            if isinstance(item, dict):
                # 如果是字典（比如来自 dataset.py），提取上下文内容
                # 你可以根据实际情况提取 "Input_Context" 或拼接多个字段
                chunk_text = str(item.get("Input_Context", ""))
                if chunk_text and chunk_text.lower() != "nan":
                    chunks.append(chunk_text)
            elif isinstance(item, str):
                # 如果直接是纯文本
                chunks.append(item)
                
        if not chunks:
            raise ValueError("提取到的文本块为空，无法建库！请检查输入数据格式。")

        # 使用 FAISS 从文本列表构建本地向量索引 (Vector Index)
        self.vector_store = FAISS.from_texts(chunks, self.embedding_model)
        print(f"✅ 成功为 {len(chunks)} 条数据建立向量索引！")

    def predict(self, test_x):
        predictions = []
        
        # 遍历测试集
        for item in test_x:
            # 👇 新增这一步：从字典中提取 "Input_Query"
            query = item["Input_Query"] if isinstance(item, dict) else item
            
            # 后面的逻辑保持不变
            context_records = self._rag_retrieval(query)
            context_str = "\n".join([record["chunk"] for record in context_records])
            # ... 拼接 prompt 并调用 LLM

        prompt = f"""**context**:
{context_str}

**prompt**:
{query}
"""
        
        # 3. 调用生成模型
        if self.model:
            # 假设 model 具有 generate_chat 方法 (与你提供的格式一致)
            answer = self.model.chat(
                model="ministral-3:8b-cloud",
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
    rag.train(chunks)
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