from pathlib import Path
from typing import Optional, Tuple

import os

import pandas as pd
from transformers import trainer

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        T5ForConditionalGeneration,
    )
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "TP5 requires torch, datasets, transformers, and accelerate to be installed."
    ) from exc


DEVICE = torch.device("cpu")
MODEL_CHECKPOINT = "t5-small"
OUTPUT_DIR = Path("./t5_tp5_runs")
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128


def _normalize_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _build_prompt(question: str, context: str = "") -> str:
    question_text = _normalize_text(question)
    context_text = _normalize_text(context)

    if not question_text.lower().startswith("question:"):
        question_text = f"question: {question_text}"

    if context_text:
        return f"{question_text} context: {context_text}"
    return question_text


class TP5:
    def __init__(
        self,
        embedding_model=None,
        llm_model=None,
        model_name: str = MODEL_CHECKPOINT,
        output_dir: Path | str = OUTPUT_DIR,
        max_input_length: int = MAX_INPUT_LENGTH,
        max_target_length: int = MAX_TARGET_LENGTH,
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.model_dir = self.output_dir / "final_model"
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[T5ForConditionalGeneration] = None

    def _ensure_backbone(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        if self.model is None:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(DEVICE)

    def _build_training_frame(self, train: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"question", "answer"}
        missing_columns = required_columns - set(train.columns)
        if missing_columns:
            raise ValueError(
                f"[TP5] train DataFrame is missing required columns: {sorted(missing_columns)}"
            )

        frame = train.copy()
        if "context" not in frame.columns:
            frame["context"] = ""

        frame["question"] = frame["question"].map(_normalize_text)
        frame["answer"] = frame["answer"].map(_normalize_text)
        frame["context"] = frame["context"].map(_normalize_text)

        frame = frame[(frame["question"] != "") & (frame["answer"] != "")]
        if frame.empty:
            raise ValueError("[TP5] No valid question/answer rows found in train DataFrame.")

        frame["input_text"] = frame.apply(
            lambda row: _build_prompt(row["question"], row["context"]),
            axis=1,
        )
        frame["target_text"] = frame["answer"]
        return frame[["input_text", "target_text"]].reset_index(drop=True)

    def _tokenize_dataset(self, frame: pd.DataFrame) -> Dataset:
        dataset = Dataset.from_pandas(frame, preserve_index=False)

        def preprocess(batch):
            model_inputs = self.tokenizer(
                batch["input_text"],
                max_length=self.max_input_length,
                truncation=True,
            )
            labels = self.tokenizer(
                text_target=batch["target_text"],
                max_length=self.max_target_length,
                truncation=True,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    def load(self, llm_model=None) -> bool:
        if llm_model is not None:
            self.llm_model = llm_model

        if not self.model_dir.exists():
            print(f"[TP5] No saved weights at {self.model_dir}, will train.")
            return False

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=False)
        self.model = T5ForConditionalGeneration.from_pretrained(str(self.model_dir))
        self.model.to(DEVICE)
        print(f"[TP5] Loaded weights from {self.model_dir}")
        return True

    def train(self, train: pd.DataFrame, test=None) -> None:
        self._ensure_backbone()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if train is None or len(train) == 0:
            raise ValueError("[TP5] train is empty.")

        training_frame = self._build_training_frame(train)
        tokenized_train = self._tokenize_dataset(training_frame)

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=5e-4,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            use_cpu=True,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
            ),
        )

        print(f"[TP5] Training on {len(training_frame)} samples...")
        print(" stop before)")
        trainer.train()
        print("stop after")

        self.model = trainer.model
        self.model.to(DEVICE)
        trainer.save_model(str(self.model_dir))
        self.tokenizer.save_pretrained(str(self.model_dir))
        print(f"[TP5] Weights saved to {self.model_dir}")

    def predict(self, query: str, context: str = "") -> Tuple[str, str]:
        if self.model is None or self.tokenizer is None:
            if not self.load():
                raise RuntimeError(
                    f"[TP5] No trained weights found in {self.model_dir}. Please train first."
                )

        input_text = _build_prompt(query, context)
        encoded = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(DEVICE)

        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **encoded,
                max_new_tokens=self.max_target_length,
                num_beams=4,
                early_stopping=True,
            )

        answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        return input_text, answer
