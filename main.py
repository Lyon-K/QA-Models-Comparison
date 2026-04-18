import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from ollama import Client
import logging

from data.LYS_dataset import get_dataset

from models.hybrid_rag.pipeline import HybridRetrievalPipeline
from models.graphRAG.graphRAG import GraphRAG
from models.RAG.RAG import VectorRAG
from models.noRag.noRag import NoRAG
from evaluation.metrics import evaluate
import os

# load environment variables
load_dotenv()
logging.basicConfig(level=logging.WARNING)

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
    headers={"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY")},
)


def prepare_hybrid_dataframe(df):
    """
    Make sure the dataframe contains the fields required by hybrid retrieval:
    - chunk_id
    - doc_id
    - text
    """
    df = df.copy()

    # Add unique chunk_id if missing
    if "chunk_id" not in df.columns:
        df["chunk_id"] = [f"chunk_{i}" for i in range(len(df))]

    # Add doc_id if missing
    if "doc_id" not in df.columns:
        df["doc_id"] = [f"doc_{i}" for i in range(len(df))]

    # Add text field if missing
    if "text" not in df.columns:
        if "context" in df.columns:
            df["text"] = df["context"].fillna("").astype(str)

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
    train, test = get_dataset()
    # 在 main.py 第 50 行左右插入

    # Prepare train set for hybrid pipeline
    train = prepare_hybrid_dataframe(train)

    # Optional: print columns for debugging
    print("Train columns:", list(train.columns))
    print("Sample train record:", train.head(1).to_dict(orient="records")[0])

    if not os.environ.get("OLLAMA_API_KEY"):
        raise ValueError("OLLAMA_API_KEY is not set. Please add it to your .env file.")

    models = {
        #"w2v": W2V(),
         "rag": VectorRAG(embedding_model=embedding_model, llm_model=llm_client),
         "hybrid": HybridRetrievalPipeline(chunks=train.to_dict(orient='records'), llm_model=llm_client),
         #"graphrag": GraphRAG(embedding_model=embedding_model, llm_model=llm_client),
         "noRag": NoRAG(llm_model=llm_client),
    }

    for name, model in models.items():
        loaded = False

        # ---- Train / Load ----
        if hasattr(model, "load"):
            print(f"Loading {name}")
            loaded = model.load(llm_model=llm_client)

        if loaded is False and hasattr(model, "train"):
            print(f"Training {name}...")
            model.train(train=train, test=test)

        # ---- Evaluate ----
        query = "What is the main cause of HIV-1 infection in children?"
        print(f"Model: {name} - Query: {query}")

        context, answer = model.predict(query)

        print(f"Context:\n{context}")
        print(f"response:\n{answer}")

        evaluate(answer, test)


if __name__ == "__main__":
    main()