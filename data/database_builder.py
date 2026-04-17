import os
import time
import feedparser
from html.parser import HTMLParser
from datasets import load_dataset

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ---------------------------------------------------------
# 工具类：HTML 标签清理器 (用于处理 RSS 订阅源内容)
# ---------------------------------------------------------
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []

    def handle_data(self, d):
        self.text.append(d)

    def get_data(self):
        return ''.join(self.text)

def strip_html(html):
    """清理文本中的 HTML 标签"""
    stripper = MLStripper()
    try:
        stripper.feed(html)
        return stripper.get_data()
    except:
        return html

# ---------------------------------------------------------
# 核心功能 1：构建本地 PDF 数据库 (Local Database)
# ---------------------------------------------------------
def build_local_database(
    pdf_dir="WHO", 
    persist_dir="./db_local_who", 
    embedding_model_name="pritamdeka/S-PubMedBert-MS-MARCO"
):
    """读取指定文件夹的 PDF，切分并建立本地 Chroma 向量数据库"""
    print(f"\n>>> [1/3] 正在构建本地 PDF 数据库 (扫描 '{pdf_dir}' 文件夹)...")
    
    if not os.path.exists(pdf_dir):
        print(f"⚠️ 找不到文件夹 '{pdf_dir}'，请确保路径正确。本地建库跳过。")
        return False

    try:
        loader = PyPDFDirectoryLoader(pdf_dir)
        documents = loader.load()
        
        if not documents:
            print(f"⚠️ '{pdf_dir}' 文件夹中没有可读取的 PDF。本地建库跳过。")
            return False

        print("   - 正在进行文本切分...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        local_chunks = text_splitter.split_documents(documents)
        
        print("   - 正在加载 Embedding 模型并构建数据库...")
        rag_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # 建立并持久化数据库
        Chroma.from_documents(local_chunks, rag_embeddings, persist_directory=persist_dir)
        
        print(f"✅ 本地医学数据库建立完成！共提取 {len(local_chunks)} 个片段，保存至 '{persist_dir}'。")
        return True
        
    except Exception as e:
        print(f"⚠️ 本地知识库构建失败: {e}")
        return False

# ---------------------------------------------------------
# 核心功能 2：构建网络 RSS 数据库 (Web Database)
# ---------------------------------------------------------
def build_web_database(
    persist_dir="./db_web_news", 
    embedding_model_name="pritamdeka/S-PubMedBert-MS-MARCO"
):
    """通过 Google News RSS 抓取最新公共卫生要闻，并建立网络 Chroma 向量数据库"""
    print("\n>>> [2/3] 正在构建网络实时数据库 (抓取 Google News RSS)...")
    
    GOOGLE_NEWS_URLS = [
        "https://news.google.com/rss/search?q=public+health+policy&hl=en&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=disease+outbreak&hl=en&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=WHO+health&hl=en&gl=US&ceid=US:en",
    ]
    
    web_chunks = []
    
    for feed_url in GOOGLE_NEWS_URLS:
        try:
            topic = feed_url.split('q=')[1].split('&')[0].replace('+', ' ')
            print(f"   🔍 正在获取主题: {topic}...")
            feed = feedparser.parse(feed_url)
            
            if feed.entries:
                for entry in feed.entries[:5]:  # 每个订阅源取前 5 条
                    title = entry.get('title', '')
                    content = entry.get('summary', '')
                    link = entry.get('link', '')
                    
                    clean_content = strip_html(content)
                    
                    if title and clean_content and link:
                        doc = Document(
                            page_content=f"Title: {title}. Content: {clean_content}",
                            metadata={'source': link, 'published': entry.get('published', '')}
                        )
                        web_chunks.append(doc)
            else:
                print(f"   ⚠️ 该订阅源暂无新闻")
            
            time.sleep(1) # 礼貌抓取，防止被封
            
        except Exception as feed_error:
            print(f"   ⚠️ 订阅源获取失败: {feed_error}")

    if web_chunks:
        print(f"   - 总计获取 {len(web_chunks)} 篇有效新闻，正在构建数据库...")
        try:
            rag_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            Chroma.from_documents(web_chunks, rag_embeddings, persist_directory=persist_dir)
            print(f"✅ 网络实时数据库建立完成！已索引 {len(web_chunks)} 篇报道，保存至 '{persist_dir}'。")
            return True
        except Exception as e:
            print(f"⚠️ 建立向量数据库时出错: {e}")
            return False
    else:
        print("⚠️ 未成功获取任何新闻，网络建库跳过。")
        return False

# ---------------------------------------------------------
# 核心功能 3：构建 Hugging Face 历史经验数据库 (Historical QA DB)
# ---------------------------------------------------------
def build_historical_database(
    dataset_name="BryanTegomoh/public-health-intelligence-datasetpublic-health-intelligence-dataset",
    persist_dir="./db_historical_qa",
    train_sample_size=2000,
    embedding_model_name="pritamdeka/S-PubMedBert-MS-MARCO"
):
    """将 Hugging Face 数据集中的训练集问答对，转化为历史经验向量数据库"""
    print(f"\n>>> [3/3] 正在构建历史经验问答库 (下载 HF 数据集: {dataset_name})...")
    
    try:
        raw_datasets = load_dataset(dataset_name)
        # 抽取部分训练集作为历史库，防止数据库过于庞大
        train_sample = raw_datasets["train"].shuffle(seed=42).select(range(train_sample_size))
        
        hf_chunks = []
        for example in train_sample:
            user_q, target_a = "", ""
            for msg in example["messages"]:
                if msg["role"] == "user": user_q = msg["content"]
                elif msg["role"] == "assistant": target_a = msg["content"]
            
            if user_q and target_a:
                # 拼接问答作为 Document，供 VectorRAG 检索
                doc = Document(
                    page_content=f"Historical Question: {user_q}\nHistorical Answer: {target_a}",
                    metadata={"source": "HF_Historical_QA_Dataset"}
                )
                hf_chunks.append(doc)
                
        if hf_chunks:
            print(f"   - 成功提取 {len(hf_chunks)} 条历史问答，正在构建数据库...")
            rag_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            Chroma.from_documents(hf_chunks, rag_embeddings, persist_directory=persist_dir)
            print(f"✅ 历史经验问答库建立完成！保存至 '{persist_dir}'。")
            return True
        else:
            print("⚠️ 未能提取到有效的历史问答，历史建库跳过。")
            return False
            
    except Exception as e:
        print(f"⚠️ 构建历史问答库失败: {e}")
        return False

# ---------------------------------------------------------
# 辅助功能：获取性能评估测试集 (Ground Truth Dataset)
# ---------------------------------------------------------
def get_evaluation_dataset(
    dataset_name="BryanTegomoh/public-health-intelligence-datasetpublic-health-intelligence-dataset",
    val_sample_size=20
):
    """从验证集中提取标准问题和答案，供 benchmark_arena.py 进行跑分测试使用"""
    print(f"正在准备测试集 (从 {dataset_name} 的验证集中提取 {val_sample_size} 条)...")
    raw_datasets = load_dataset(dataset_name)
    val_sample = raw_datasets["validation"].shuffle(seed=42).select(range(val_sample_size))
    
    test_queries = []
    ground_truths = []
    
    for example in val_sample:
        user_q, target_a = "", ""
        for msg in example["messages"]:
            if msg["role"] == "user": user_q = msg["content"]
            elif msg["role"] == "assistant": target_a = msg["content"]
            
        test_queries.append(user_q)
        ground_truths.append(target_a)
        
    return test_queries, ground_truths


# ---------------------------------------------------------
# 独立运行入口
# ---------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print(" 🏭 启动中央数据工厂 (Central Data Builder)")
    print("="*70)
    
    start_time = time.time()
    
    # 依次执行三大知识库的构建
    build_local_database()
    build_web_database()
    build_historical_database()
    
    end_time = time.time()
    print("\n" + "="*70)
    print(f" 🎉 数据工厂流水线执行完毕！总耗时: {end_time - start_time:.2f} 秒")
    print(" 提示: 现在所有的本地库、网络库和历史库都已持久化保存，你可以安全地运行评测或前端界面了。")
    print("="*70)