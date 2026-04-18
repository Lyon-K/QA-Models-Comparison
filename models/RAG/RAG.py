import os
from typing import Optional, List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
from ollama import Client

class VectorRAG:
    embedding_model: HuggingFaceEmbeddings
    vector_store: FAISS = None
    model: Optional[Client] = None
    chat_model_name: str

    def __init__(
        self, embedding_model: Optional[HuggingFaceEmbeddings] = None, llm_model: Optional[Client] = None
    ):
        # 初始化词嵌入模型 (Embedding Model)
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={
                "prompt": "Represent this sentence for searching relevant passages: "
            },
        )
        # 初始化大语言模型 (Large Language Model, LLM)
        self.model = llm_model
        self.chat_model_name = os.environ.get("OLLAMA_MODEL", "ministral-3:8b-cloud")

    def load(self, **kwargs) -> bool:
        """
        统一接口规范 (Unified Interface Specification):
        由于 FAISS 是基于内存的 (In-Memory)，每次运行都需要重新建库。
        因此强制返回 False，通知外部流水线调用 self.train()。
        """
        return False

    def train(self, train, **kwargs):
        """
        构建知识库索引 (Index Building)。
        """
        print("⏳ 正在提取文本并构建 FAISS 向量数据库索引 (Building Vector Index)...")
        
        chunks = []
        for item in train:
            if isinstance(item, dict):
                # 兼容字典格式输入
                chunk_text = str(item.get("Input_Context", ""))
                if chunk_text and chunk_text.lower() != "nan":
                    chunks.append(chunk_text)
            elif isinstance(item, str):
                # 兼容纯文本输入
                chunks.append(item)
                
        if not chunks:
            raise ValueError("提取到的文本块为空，无法建库！请检查输入数据格式。")

        # 从文本列表构建本地向量索引 (Vector Index)
        self.vector_store = FAISS.from_texts(chunks, self.embedding_model)
        print(f"✅ 成功为 {len(chunks)} 条数据建立向量索引！")

    def predict(self, query: str, **kwargs):
        """
        接口完全对齐 GraphRAG (Interface Aligned with GraphRAG)
        返回 (context, answer) 元组 (Tuple)
        """
        # 1. 检索上下文 (Retrieve Context)
        context = self._rag_retrieval(query)
        
        # 2. 拼接提示词 (Prompt Construction) - 采用与 GraphRAG 完全一致的逻辑
        prompt = (
            f"**context**:\n{context}\n\n**prompt**:\n{query}" if context else query
        )
        
        # 3. 调用生成模型 (Generation)
        if self.model:
            try:
                response = self.model.chat(
                    model=self.chat_model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                # 兼容字典和对象两种返回形式
                if isinstance(response, dict):
                    answer = response.get('message', {}).get('content', '')
                else:
                    answer = response.message.content
            except Exception as e:
                answer = f"Error during LLM generation: {e}"
        else:
            answer = "LLM not loaded. Prompt preview:\n" + prompt

        # 4. 严格按照 main.py 的要求，返回 (context, answer) 元组
        return context, answer

    def _rag_retrieval(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        执行向量检索，返回字段名对齐 GraphRAG ('text' 和 'score')。
        """
        if not self.vector_store:
            raise ValueError("Vector DB is empty. Please call `train` first.")

        # 使用带有相关性分数的相似度搜索 (Similarity Search with Relevance Scores)
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)

        context = []
        for doc, score in results:
            context.append({
                "text": doc.page_content,
                "score": score
            })
            
        return context

# =====================================================================
# 独立测试块 (Standalone Test Block)
# 仅在直接运行此文件 (python RAG.py) 时执行，被 main.py import 时不会执行
# =====================================================================
if __name__ == "__main__":
    from dotenv import load_dotenv
    import logging

    load_dotenv()
    logging.basicConfig(level=logging.DEBUG)

    print("=== 开始独立测试 VectorRAG (Starting standalone test for VectorRAG) ===")
    
    # 1. 实例化 (Instantiation)
    # 这里我们只测检索，所以暂时不传入 LLM client，如果你想测生成，可以在这里传入
    vector_rag = VectorRAG()

    # 2. 准备数据并建库 (Data preparation and Index Building)
    try:
        print("正在尝试读取数据集...")
        # 🚨 重点修复：加入 encoding='utf-8' 解决 gbk 解码报错 (Decoding Error)
        # 请将下面的路径替换为你实际存放 test_dataset.csv 的绝对或相对路径
        csv_path = 'C:/Users/liudo/Desktop/NTU_learning/EE6405/group work/test_dataset.csv'
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        train_chunks = []
        for index, row in df.iterrows():
            chunk_text = f"Topic: {row.get('Topic', '')}\nQuestion: {row.get('Question', '')}\nAnswer: {row.get('Answer', '')}"
            train_chunks.append(chunk_text)
            
        vector_rag.train(train=train_chunks)
        
    except FileNotFoundError:
        print("⚠️ 未找到 CSV 文件，使用模拟数据进行建库 (Using mock data to build index)...")
        mock_data = [
            "The recommended daily sugar intake for adults is less than 25g.", 
            "Regular cardiovascular exercise improves heart health."
        ]
        vector_rag.train(train=mock_data)
    except Exception as e:
        print(f"❌ 数据加载出错: {e}")

    # 3. 执行查询 (Execution of Query)
    query = "What is the recommended daily sugar intake?"
    print(f"\n执行查询 (Executing query): {query}")
    
    context, message = vector_rag.predict(query=query)
    
    print("\n=== 检索到的上下文 (Retrieved Context) ===")
    for item in context:
        print(f"Score: {item['score']:.3f} | Text: {item['text']}")

    print("\n=== 模型返回结果 (Model Output) ===")
    print(message)
