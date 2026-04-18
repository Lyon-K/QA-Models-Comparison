import os
import logging
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from ollama import Client

from data.LYS_dataset import get_dataset
from models.hybrid_rag.pipeline import HybridRetrievalPipeline
from models.graphRAG.graphRAG import GraphRAG
from models.RAG.RAG import VectorRAG
from models.noRag.noRag import NoRAG
from models.T5.T5 import TP5
from evaluation.metrics import QAEvaluator

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.WARNING)

# Initialize global models and clients
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    query_encode_kwargs={
        "prompt": "Represent this sentence for searching relevant passages: "
    },
)

llm_client: Client = Client(
    host="https://ollama.com",
    headers={"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", "")},
)


def prepare_hybrid_dataframe(df):
    """
    Ensure the dataframe contains the fields required by the retrieval pipelines.
    """
    df = df.copy()

    if "chunk_id" not in df.columns:
        df["chunk_id"] = [f"chunk_{i}" for i in range(len(df))]

    if "doc_id" not in df.columns:
        df["doc_id"] = [f"doc_{i}" for i in range(len(df))]

    if "text" not in df.columns:
        if "context" in df.columns:
            df["text"] = df["context"].fillna("").astype(str)
        elif "Question" in df.columns and "Answer" in df.columns:
            df["text"] = (
                df["Question"].fillna("").astype(str)
                + " "
                + df["Answer"].fillna("").astype(str)
            ).str.strip()
        elif "question" in df.columns and "answer" in df.columns:
            df["text"] = (
                df["question"].fillna("").astype(str)
                + " "
                + df["answer"].fillna("").astype(str)
            ).str.strip()
        elif "query" in df.columns and "answer" in df.columns:
            df["text"] = (
                df["query"].fillna("").astype(str)
                + " "
                + df["answer"].fillna("").astype(str)
            ).str.strip()
        else:
            raise ValueError(
                f"Cannot build 'text' column from dataset columns: {list(df.columns)}"
            )

    return df


def main():
    # Load dataset natively
    train, test = get_dataset()
    train_df = prepare_hybrid_dataframe(train)
    
    # Create a lowercased version of the dataframe strictly for T5 compatibility
    t5_train_df = train_df.copy()
    t5_train_df.columns = [col.lower() for col in t5_train_df.columns]

    print("Train columns:", list(train_df.columns))

    models = {
        "noRag": NoRAG(llm_model=llm_client),
        "rag": VectorRAG(embedding_model=embedding_model, llm_model=llm_client),
        "hybrid": HybridRetrievalPipeline(chunks=train_df.to_dict(orient='records'), llm_model=llm_client),
        "graphrag": GraphRAG(embedding_model=embedding_model, llm_model=llm_client),
        "t5": TP5(embedding_model=embedding_model, llm_model=llm_client)
    }

    evaluator = QAEvaluator(embedding_model=embedding_model, llm_client=llm_client)
    evaluation_results = []

    # Ensure test is iterable as a list of dictionaries
    test_records = test.to_dict(orient="records") if hasattr(test, "to_dict") else test

    for name, model in models.items():
        print(f"\nProcessing Model: {name.upper()}")
        loaded = False

        if hasattr(model, "load"):
            print(f"Loading weights for {name}...")
            loaded = model.load(llm_model=llm_client)

        if not loaded and hasattr(model, "train"):
            print(f"Training or indexing {name}...")
            if name == "hybrid":
                model.train(train=train_df.to_dict(orient='records'), test=test)
            elif name == "t5":
                model.train(train=t5_train_df, test=test)
            else:
                model.train(train=train_df, test=test)

        # Batch Evaluation
        model_scores = {"Model": name, "ROUGE-L": 0.0, "Semantic_Sim": 0.0, "LLM_Score": 0.0}
        
        for idx, row in enumerate(test_records):
            query = row.get("Question", row.get("question", ""))
            ground_truth = row.get("Answer", row.get("answer", ""))
            
            if not query or not ground_truth:
                continue

            if name == "graphrag":
                _, answer = model.predict(query, n_hop=1)
            elif name == "t5":
                _, answer = model.predict(query)
            else:
                _, answer = model.predict(query)
                
            metrics = evaluator.evaluate_single(
                prediction=answer, 
                reference=ground_truth, 
                question=query
            )
            
            model_scores["ROUGE-L"] += metrics["ROUGE-L"]
            model_scores["Semantic_Sim"] += metrics["Semantic_Sim"]
            model_scores["LLM_Score"] += metrics["LLM_Score"]
            
        num_tests = len(test_records)
        if num_tests > 0:
            model_scores["ROUGE-L"] = round(model_scores["ROUGE-L"] / num_tests, 4)
            model_scores["Semantic_Sim"] = round(model_scores["Semantic_Sim"] / num_tests, 4)
            model_scores["LLM_Score"] = round(model_scores["LLM_Score"] / num_tests, 2)
            
        evaluation_results.append(model_scores)

    print("\n\n=============== FINAL EVALUATION REPORT ===============")
    report_df = pd.DataFrame(evaluation_results)
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()