from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from data.dataset import load_dataset
from models.template_model import TemplateModel as W2V
from models.graphRAG.graphRAG import GraphRAG
from models.template_model import TemplateModel as RAG
from evaluation.metrics import evaluate

# load environment variables
load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    query_encode_kwargs={
        "prompt": "Represent this sentence for searching relevant passages: "
    },
)

llm = None


def main():
    train_x, test_x, train_y, test_y = load_dataset()

    models = {
        "w2v": W2V(),
        # "rag": RAG(),
        "graphrag": GraphRAG(embedding_model=embedding_model, llm_model=llm),
    }

    for name, model in models.items():
        loaded = False
        # ---- Train / Load ----
        if hasattr(model, "load"):
            print(f"Loading {name}")
            loaded = model.load(llm_model=llm)
        if loaded == False and hasattr(model, "train"):
            print(f"Training {name}...")
            model.train(train_x, train_y, val_x=test_x, val_y=test_y)

        # ---- Evaluate ----
        print(f"Evaluating {name}...")

        y_pred = model.predict(test_x)
        evaluate(y_pred, test_y)


if __name__ == "__main__":
    main()
