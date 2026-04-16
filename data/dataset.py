# ---------------------------------------------------------
# CELL 1: Imports and Environment Setup
# ---------------------------------------------------------
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import KFold

# Set up the computation device (GPU is highly recommended for LLM fine-tuning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

# ---------------------------------------------------------
# CELL 2: Data Sourcing & Downsampling
# ---------------------------------------------------------
dataset_name = "BryanTegomoh/public-health-intelligence-datasetpublic-health-intelligence-dataset"
print(f"Fetching the dataset from Hugging Face: {dataset_name}...")
full_dataset = load_dataset(dataset_name)

# Downsampling: The original dataset contains ~588k samples. 
# We shuffle the dataset with a fixed seed and select a subset of 10,000 samples.
SUBSET_SIZE = 10000
print(f"Downsampling the dataset to {SUBSET_SIZE} samples...")
dataset_subset = full_dataset['train'].shuffle(seed=42).select(range(SUBSET_SIZE))

# Dynamic Format Detection: Locate the column containing the chat dictionaries.
sample_row = dataset_subset[0]
chat_column = None
for col_name, value in sample_row.items():
    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
        if "role" in value[0] and "content" in value[0]:
            chat_column = col_name
            break

if not chat_column:
    raise ValueError("Could not automatically find the chat formatting column.")
print(f"Detected chat data in column: '{chat_column}'")

# ---------------------------------------------------------
# CELL 3: Tokenization via apply_chat_template
# ---------------------------------------------------------
# Using Mistral as a placeholder. Replace with your preferred base model.
model_checkpoint = "mistralai/Mistral-7B-Instruct-v0.1" 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    # Extract the batch of message lists
    batch_messages = examples[chat_column]
    
    # apply_chat_template formats the list of dicts into a single prompt string
    # complete with specific model control tokens and the EOS token.
    formatted_texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        for messages in batch_messages
    ]
    
    # Tokenize the formatted textual conversations
    model_inputs = tokenizer(
        formatted_texts,
        padding="max_length",
        truncation=True,
        max_length=512  # Adjust based on your GPU VRAM capabilities
    )
    
    # For Causal Language Modeling, labels are identical to input_ids.
    # The model shifts them internally to predict the next token.
    model_inputs["labels"] = [input_id.copy() for input_id in model_inputs["input_ids"]]
    
    return model_inputs

print("\nApplying chat templates and tokenizing the 10k dataset...")
# It is highly efficient to tokenize the entire subset BEFORE splitting into K-Folds.
encoded_dataset = dataset_subset.map(
    preprocess_function, 
    batched=True,
    remove_columns=dataset_subset.column_names
)

# Convert to PyTorch tensors
encoded_dataset.set_format("torch")
print("Tokenization complete!")

# ---------------------------------------------------------
# CELL 4: K-Fold Cross-Validation Setup
# ---------------------------------------------------------
NUM_FOLDS = 10
print(f"\nSetting up {NUM_FOLDS}-Fold Cross-Validation...")

# Initialize the KFold generator
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Create a list to store the (train_dataset, validation_dataset) tuples for each fold
folds = []

# Generate the indices for each fold and create HF dataset subsets
for fold_idx, (train_indices, val_indices) in enumerate(kf.split(range(len(encoded_dataset)))):
    train_fold = encoded_dataset.select(train_indices)
    val_fold = encoded_dataset.select(val_indices)
    
    folds.append((train_fold, val_fold))
    print(f"Fold {fold_idx + 1}: {len(train_fold)} Train samples | {len(val_fold)} Validation samples")

print("\nData processing complete! You can now iterate through the `folds` list for training.")
# Example of how to access a specific fold during your training loop:
# for fold, (train_data, val_data) in enumerate(folds):
#     trainer = Trainer(model=model, train_dataset=train_data, eval_dataset=val_data, ...)
#     trainer.train()
