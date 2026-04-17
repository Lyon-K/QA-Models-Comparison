import os
# 强制让 datasets 库使用加速镜像点拉取数据
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from datasets import load_dataset
import pandas as pd


def build_evaluation_dataset(sample_per_source=100, dataset_path="data/rag_test_dataset.json"):
    print("🚀 开始构建测试集 (Building Evaluation Dataset)...")
    
    final_dataset = []

    # --- 数据源 1: COVID-QA (代表公共卫生与传染病知识) ---
    print("⬇️ 正在处理 COVID-QA (Public Health)...")
    try:
        ds1 = load_dataset("covid_qa_deepset", split=f"train[:{sample_per_source}]")
        for row in ds1:
            final_dataset.append({
                "source_dataset": "COVID-QA",
                "category": "Public Health / Outbreak",
                "context": row['context'],
                "question": row['question'],
                "answer": row['answers']['text'][0] # 提取第一个标准答案
            })
    except Exception as e:
        print(f"❌ 加载 COVID-QA 失败: {e}")

    # --- 数据源 2: PubMedQA (代表学术文献与医学研究) ---
    # 注意：使用 'pqa_labeled' 确保有明确的真值答案
    print("⬇️ 正在处理 PubMedQA (Academic Research)...")
    try:
        ds2 = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=f"train[:{sample_per_source}]")
        print(ds2.column_names)
        for row in ds2:
            final_dataset.append({
                "source_dataset": "PubMedQA",
                "category": "Academic / Scientific",
                "context": " ".join(row['context']), # 合并多个背景段落
                "question": row['question'],
                "answer": row['final_decision'] # 答案通常为 yes/no/maybe
            })
    except Exception as e:
        print(f"❌ 加载 PubMedQA 失败: {e}")

    # --- 数据源 3: MedQuad (代表临床百科与问答) ---
    print("⬇️ 正在处理 MedQuad (Clinical Encyclopedia)...")
    try:
        ds3 = load_dataset("keivalya/MedQuad-MedicalQnADataset", split=f"train[:{sample_per_source}]")
        for row in ds3:
            final_dataset.append({
                "source_dataset": "MedQuad",
                "category": "Clinical / General Medical",
                "context": row['Answer'], # 在 MedQuad 中，答案本身就是高质量的知识块
                "question": row['Question'],
                "answer": row['Answer']
            })
    except Exception as e:
        print(f"❌ 加载 MedQuad 失败: {e}")

    # --- 保存为 JSON 文件 ---
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)
    
    print(f"\n🎉 测试集构建完成！")
    print(f"📁 总计条数: {len(final_dataset)}")
    print(f"📄 文件路径: {dataset_path}")
    return final_dataset

def get_dataset(sample_per_source=100, dataset_path="data/rag_test_dataset.json", use_cached=True):
    if use_cached and os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            print(f"Using cached dataset from {dataset_path}")
            dataset = json.load(f)
    else:
        dataset = build_evaluation_dataset(sample_per_source=sample_per_source, dataset_path=dataset_path)

    train_set = pd.json_normalize(dataset)
    test_set = None
    return train_set, test_set

if __name__ == "__main__":
    # 你可以调整 sample_per_source 来控制每个数据源抽取的样本量
    get_dataset(sample_per_source=10)