# ---------------------------------------------------------
# CELL 1: Setup And Shared Configuration
# ---------------------------------------------------------
import csv
import math
import os
from pathlib import Path
import re
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DATASET_NAME = "BryanTegomoh/public-health-intelligence-datasetpublic-health-intelligence-dataset"
TEST_DATA_PATH = Path("/Users/tanglan/Desktop/test_dataset.csv")
MODEL_CHECKPOINT = "t5-small"
OUTPUT_DIR = Path("./t5_group_aligned_runs")

# Keep the notebook light enough to run on CPU during integration.
MAX_TRAIN_SAMPLES = 300 if DEVICE.type == "cpu" else 3000
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
SEED = 42

TEST_METADATA = None


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


# ---------------------------------------------------------
# CELL 2: load_dataset()
# ---------------------------------------------------------
def _format_question(text: str) -> str:
    text = str(text).strip()
    return text if text.lower().startswith("question:") else f"question: {text}"


def _build_inference_prompt(text: str) -> str:
    normalized = str(text).strip()
    return f"explain: {normalized} context: public health explanation"


def _clean_generated_text(text: str, query: str) -> str:
    cleaned = " ".join(str(text).replace("\n", " ").split()).strip()
    cleaned = re.sub(r"https?://\S+|www\.\S+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"^(answer|response|explanation)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b([A-Za-z0-9\- ]{2,40}explanations?)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b([A-Za-z0-9\- ]{2,40}explanations?)\b(\s+\1\b)+",
        r"\1",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -:;,.")

    if not cleaned:
        cleaned = "This claim needs more context to be explained clearly."

    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    sentences = [part.strip() for part in sentence_pattern.split(cleaned) if part.strip()]
    word_count = len(cleaned.split())
    query_text = str(query).strip()
    is_question = query_text.endswith("?") or query_text.lower().startswith(
        ("how ", "what ", "why ", "when ", "where ", "which ", "who ")
    )

    if len(sentences) == 0:
        sentences = [cleaned]

    if word_count < 20 or len(sentences) < 2:
        topic = cleaned.rstrip(".!?").strip() or query_text or "This topic"
        if is_question:
            sentences = [
                f"The question about {topic} relates to a public health topic.",
                "It connects to important health issues affecting individuals and communities.",
                "Understanding this helps improve awareness and decision-making.",
            ]
        else:
            sentences = [
                f"{topic} is a public health topic.",
                "It relates to important health issues affecting individuals and communities.",
                "Understanding this helps improve awareness and decision-making.",
            ]
    elif len(sentences) == 2:
        sentences.append("Understanding this helps improve awareness and decision-making.")

    final_text = " ".join(sentences[:3])
    final_text = re.sub(r"\s{2,}", " ", final_text)
    final_text = re.sub(r"\s+([,.!?;:])", r"\1", final_text).strip()
    return final_text


def _extract_train_pair(example: Dict[str, Any]):
    user_turn = ""
    assistant_turn = ""
    for message in example["messages"]:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "user" and not user_turn:
            user_turn = content.strip()
        elif role == "assistant" and not assistant_turn:
            assistant_turn = content.strip()
        if user_turn and assistant_turn:
            break
    return user_turn, assistant_turn


def _build_test_input(row: dict[str, str]) -> str:
    return _format_question(row["Question"])


def load_dataset(max_train_samples: int = MAX_TRAIN_SAMPLES):
    global TEST_METADATA

    from datasets import load_dataset as hf_load_dataset

    print(f"Loading training dataset: {TRAIN_DATASET_NAME}")
    full_dataset = hf_load_dataset(TRAIN_DATASET_NAME)
    raw_train = full_dataset["train"].shuffle(seed=SEED).select(range(max_train_samples))

    train_x, train_y = [], []
    for example in raw_train:
        question, answer = _extract_train_pair(example)
        if question and answer:
            train_x.append(_format_question(question))
            train_y.append(str(answer).strip())

    with TEST_DATA_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    TEST_METADATA = rows
    test_x = [_build_test_input(row) for row in rows]
    test_y = [str(row.get("Answer", "")).strip() for row in rows]

    print(f"Train samples: {len(train_x)}")
    print(f"Test samples: {len(test_x)}")

    return train_x, test_x, train_y, test_y


def load_default_datasets():
    """Explicit helper for loading the module's default train/test data."""
    return load_dataset()


# ---------------------------------------------------------
# CELL 3: T5 Class Implementation
# ---------------------------------------------------------
class T5:
    tokenizer = None
    model = None

    def __init__(
        self,
        model_name: str = MODEL_CHECKPOINT,
        output_dir: Path = OUTPUT_DIR,
        max_input_length: int = MAX_INPUT_LENGTH,
        max_target_length: int = MAX_TARGET_LENGTH,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.model_dir = self.output_dir / "final_model"
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def _load_from_identifier(self, model_identifier: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, use_fast=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_identifier)
        self.model.to(DEVICE)

    def _ensure_model(self):
        if self.tokenizer is None or self.model is None:
            self._load_from_identifier(self.model_name)

    def _extract_query_text(self, item: Any) -> str:
        if isinstance(item, dict):
            for key in ["Input_Query", "Question", "input_text", "query"]:
                if key in item and not _is_missing(item[key]):
                    return _build_inference_prompt(item[key])
            raise ValueError(f"Unsupported query dictionary format: {item.keys()}")
        return _build_inference_prompt(item)

    def _extract_answer_text(self, item: Any) -> str:
        if isinstance(item, dict):
            for key in ["Answer", "True_Answer", "target_text", "output_text"]:
                if key in item and not _is_missing(item[key]):
                    return str(item[key]).strip()
            raise ValueError(f"Unsupported answer dictionary format: {item.keys()}")
        return str(item).strip()

    def _prepare_train_pairs(self, train_x, train_y=None):
        inputs, targets = [], []

        if train_y is None:
            for item in train_x:
                if not isinstance(item, dict):
                    raise ValueError("When train_y is None, train_x must contain dictionaries with both question and answer.")
                inputs.append(self._extract_query_text(item))
                targets.append(self._extract_answer_text(item))
        else:
            for x_item, y_item in zip(train_x, train_y):
                inputs.append(self._extract_query_text(x_item))
                targets.append(self._extract_answer_text(y_item))

        return inputs, targets

    def _build_dataset(self, x_values, y_values):
        from datasets import Dataset

        return Dataset.from_dict({"input_text": list(x_values), "target_text": list(y_values)})

    def _tokenize_dataset(self, dataset):
        def preprocess_function(examples):
            model_inputs = self.tokenizer(
                examples["input_text"],
                max_length=self.max_input_length,
                truncation=True,
            )
            labels = self.tokenizer(
                text_target=examples["target_text"],
                max_length=self.max_target_length,
                truncation=True,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

    def load(self):
        if not self.model_dir.exists():
            return False

        self._load_from_identifier(str(self.model_dir))
        return True

    def train(self, train_x, train_y=None, val_x=None, val_y=None):
        from transformers import (
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )

        self._ensure_model()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        processed_train_x, processed_train_y = self._prepare_train_pairs(train_x, train_y)
        train_dataset = self._build_dataset(processed_train_x, processed_train_y)
        tokenized_train = self._tokenize_dataset(train_dataset)

        print(f"T5 training samples: {len(processed_train_x)}")
        if val_x is not None:
            print(f"T5 validation samples (not used for EM now): {len(val_x)}")

        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        per_device_batch_size = 2 if DEVICE.type == "cpu" else 8
        num_train_epochs = 1 if DEVICE.type == "cpu" else 2

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_batch_size,
            learning_rate=5e-4,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="no",
            evaluation_strategy="no",
            predict_with_generate=False,
            report_to="none",
            fp16=torch.cuda.is_available(),
            use_cpu=(DEVICE.type == "cpu"),
            use_mps_device=False,
            dataloader_num_workers=0,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        self.model = trainer.model
        self.model.to(DEVICE)
        trainer.save_model(str(self.model_dir))
        self.tokenizer.save_pretrained(str(self.model_dir))
        return trainer

    def predict(self, test_x):
        if self.model is None or self.tokenizer is None:
            loaded = self.load()
            if loaded is False:
                self._ensure_model()

        if isinstance(test_x, (str, dict)):
            single_input = True
            test_items = [test_x]
        else:
            single_input = False
            test_items = list(test_x)

        predictions = []
        self.model.eval()
        model_device = next(self.model.parameters()).device

        for item in test_items:
            query = self._extract_query_text(item)
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length,
            ).to(model_device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_target_length,
                    min_new_tokens=56,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    length_penalty=1.35,
                    early_stopping=True,
                )

            prediction = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            prediction = _clean_generated_text(prediction, item)
            predictions.append(prediction)

        return predictions[0] if single_input else predictions
