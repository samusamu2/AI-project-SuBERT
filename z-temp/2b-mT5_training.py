import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import GenerationConfig, EarlyStoppingCallback
from torch.utils.data import DataLoader


from datasets import load_dataset, Dataset as HFDataset
from load_dataset import preprocess_dataset
from compute_metrics import compute_metrics


MODEL_NAME = "google/mt5-small"
# Directory to save the fine-tuned model
OUTPUT_DIR = "./mt5_model"
# Directory for TensorBoard logs
LOGGING_DIR = "./mt5_logs"

# Some hyperparameters
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 100

# check if dirs exist, if not create them
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


preprocessed_train = preprocess_dataset('../datasets/SumTablets_English_train.csv')
preprocessed_val = preprocess_dataset('../datasets/SumTablets_English_validation.csv')
preprocessed_test = preprocess_dataset('../datasets/SumTablets_English_test.csv')

train_data = [{
    'source': row['sumerian'],
    'target': row['english']
} for _, row in preprocessed_train.iterrows()]

val_data = [{
    'source': row['sumerian'],
    'target': row['english']
} for _, row in preprocessed_val.iterrows()]

test_data = [{
    'source': row['sumerian'],
    'target': row['english']
} for _, row in preprocessed_test.iterrows()]

def preprocess_function(examples):
    """
    Tokenizes the source (Sumerian) and target (English) texts.
    """
    inputs = examples['source']
    targets = examples['target']

    # DEBUG: Print flag if any of inputs or targets are none
    if any(x is None for x in inputs) or any(x is None for x in targets):
        print("Warning: Found None values in inputs or targets. This may affect training.")


    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="longest")

    # Tokenize targets (English) using the newer approach
    labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="longest")
    label_ids = labels["input_ids"]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Convert lists to Hugging Face Dataset objects
train_dataset = HFDataset.from_list(train_data)
val_dataset = HFDataset.from_list(val_data)

# Apply preprocessing to the datasets
print("Tokenizing datasets...")
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

print("Example of tokenized input:")
print(tokenized_train_dataset[0])


# Set the training arguments for the Seq2SeqTrainer
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,                  # Directory to save the model

    num_train_epochs=NUM_TRAIN_EPOCHS,      # Number of training epochs
    per_device_train_batch_size=BATCH_SIZE, # Batch size for training
    per_device_eval_batch_size=BATCH_SIZE,  # Batch size for evaluation

    learning_rate=LEARNING_RATE,            # Learning rate for the optimizer
    weight_decay=0.01,                      # Weight decay for regularization
    warmup_ratio=0.05,                      # Warmup ratio for learning rate scheduler
    gradient_accumulation_steps=4,          # Gradient accumulation steps to simulate larger batch sizes
    lr_scheduler_type="cosine",             # Use cosine learning rate scheduler
    label_smoothing_factor=0.1,             # Label smoothing factor for better generalization
    max_grad_norm=5.0,                      # gradient clipping

    save_total_limit=1,                     # Only keep the last checkpoint
    predict_with_generate=True,             # Enable generation during evaluation
    report_to="tensorboard",                # Report metrics to TensorBoard
    logging_dir=LOGGING_DIR,                # Directory for TensorBoard logs
    logging_steps=50,                       # Log every 50 steps

    eval_strategy="epoch",                  # Evaluate at the end of each epoch
    save_strategy="epoch",                  # Save model at the end of each epoch
    load_best_model_at_end=True,            # Load the best model at the end of training
    metric_for_best_model="meteor",         # Metric to determine the best model
    fp16=False,         # Use mixed precision training if GPU is available
)

# Set up generation configuration for the model
generation_config = GenerationConfig(
    max_length=MAX_TARGET_LENGTH,           # Maximum length of the generated sequences
    early_stopping=True,                    # Stop generation when all beams reach the EOS token
    num_beams=4,                            # Number of beams for beam search
    no_repeat_ngram_size=3,                 # Prevent repetition of n-grams in the generated text
    pad_token_id=tokenizer.pad_token_id,    # Padding token ID for the tokenizer
    eos_token_id=tokenizer.eos_token_id,    # End of sequence token ID for the tokenizer
    decoder_start_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id   # Decoder start token ID for the model
)
model.generation_config = generation_config


# Data collator for padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,  # This ensures padding tokens in labels are ignored in loss
)

# Initialize the Seq2SeqTrainer with the model, training arguments, datasets, tokenizer, data collator, and metrics computation
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda p: compute_metrics(p, tokenizer),        # Function to compute metrics during evaluation
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]    # Early stopping callback to prevent overfitting
    )


# Start the training process
print("Starting model training...")
try:
    trainer.train()
    print("Training finished successfully!")

    # Save the final model and tokenizer
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model_tokenizer")
    print(f"Final model saved to {OUTPUT_DIR}/final_model")

except Exception as e:
    print(f"An error occurred during training: {e}")