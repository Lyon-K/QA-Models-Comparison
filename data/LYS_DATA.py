import os
from datasets import load_dataset
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. 自动拉取并预处理现成数据集 (Fetch & Preprocess Datasets)
# ==========================================
def fetch_and_preprocess_datasets(sample_size=50):
    print("🌐 正在连接 Hugging Face 拉取真实数据集 (Fetching Datasets from Hugging Face)...")
    documents = []

    # --- 数据集 1: COVID-QA (公共卫生传染病数据) ---
    print("⬇️ 拉取 COVID-QA...")
    covid_dataset = load_dataset("covid_qa_deepset", split=f"train[:{sample_size}]")
    for row in covid_dataset:
        # 去重处理，因为有的 context 是重复的
        doc = Document(
            page_content=row['context'],
            metadata={"source": "COVID-QA", "dataset_type": "public_health", "document_id": row['document_id']}
        )
        documents.append(doc)

    # --- 数据集 2: PubMedQA (学术文献摘要) ---
    print("⬇️ 拉取 PubMedQA...")
    pubmed_dataset = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split=f"train[:{sample_size}]")
    for row in pubmed_dataset:
        # PubMedQA 的 context 是列表，需要合并成字符串
        context_text = " ".join(row['contexts'])
        doc = Document(
            page_content=context_text,
            metadata={"source": "PubMedQA", "dataset_type": "academic_paper", "title": row['title']}
        )
        documents.append(doc)

    # --- 数据集 3: MedQuad (临床百科数据) ---
    print("⬇️ 拉取 MedQuad...")
    medquad_dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split=f"train[:{sample_size}]")
    for row in medquad_dataset:
        doc = Document(
            page_content=row['Answer'],
            metadata={"source": "MedQuad", "dataset_type": "clinical_encyclopedia", "qtype": row['qtype']}
        )
        documents.append(doc)

    print(f"✅ 成功加载 {len(documents)} 篇原始长文档 (Raw Documents)。")
    return documents

# ==========================================
# 2. 统一文本分块 (Unified Text Chunking)
# ==========================================
def chunk_documents(documents):
    print("✂️ 正在进行文本语义分块 (Semantic Text Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,       # 保证文本块粒度足够小，减少大模型幻觉
        chunk_overlap=40,     # 保留上下文重叠 (Context Overlap)
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 预处理完成，共生成 {len(chunks)} 个检索块 (Text Chunks)。")
    return chunks

# ==========================================
# 3. 向量化与建立本地数据库 (Embedding & Vector Indexing)
# ==========================================
def build_vector_db(chunks):
    print("🧠 正在加载嵌入模型 (Loading Embedding Model)...")
    # 由于拉取的医疗数据集多为英文，使用多语言兼容性好的 BGE 或 all-MiniLM
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    persist_dir = "./hf_medical_rag_db"
    print(f"🗄️ 正在建立向量索引 (Building Vector Index: {persist_dir})...")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print("🎉 真实医疗多源数据库构建完毕！")
    return vector_store

# ==========================================
# 4. 执行流水线
# ==========================================
if __name__ == "__main__":
    # 1. 自动拉取数据 (Fetch)
    raw_docs = fetch_and_preprocess_datasets(sample_size=5) # 建议跑通后再把 50 改成 500 或更大
    
    # 2. 清洗分块 (Chunking)
    doc_chunks = chunk_documents(raw_docs)
    
    # 3. 构建数据库 (Indexing)
    db = build_vector_db(doc_chunks)

    # 4. 简单测试：验证跨数据集的元数据过滤是否生效
    print("\n🔍 测试检索 (Test Retrieval): 从学术文献库中查找关于孕妇 (pregnant) 的记录")
    results = db.similarity_search(
        "What are the risks for pregnant women?", 
        k=2,
        filter={"dataset_type": "academic_paper"} # 精准过滤
    )
    for i, res in enumerate(results):
        print(f"\n[结果 {i+1} | 来源: {res.metadata['source']}]")
        print(f"片段: {res.page_content[:150]}...")