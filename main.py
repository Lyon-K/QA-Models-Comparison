from data.dataset import load_dataset
from models.template_model import TemplateModel as W2V
from models.template_model import TemplateModel as RAG
from models.template_model import TemplateModel as GraphRAG
from evaluation.metrics import evaluate


def main():
    train_x, test_x, train_y, test_y = load_dataset()

    models = {
        "w2v": W2V(),
        # "rag": RAG(),
        # "graphrag": GraphRAG(),
    }

    for name, model in models.items():
        # ---- Train / Load ----
        if hasattr(model, "load") and model.load() is True:
            print(f"Loaded {name}")
        else:
            print(f"Training {name}...")
            model.train(train_x, train_y, val_x=test_x, val_y=test_y)

        # ---- Evaluate ----
        print(f"Evaluating {name}...")

        y_pred = model.predict(test_x)
        evaluate(y_pred, test_y)


if __name__ == "__main__":
    main()
