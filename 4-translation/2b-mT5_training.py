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
from evaluate import load


MODEL_NAME = "google/mt5-small"
OUTPUT_DIR = "./sumerian_mt5_model"
LOG_DIR = "./sumerian_mt5_logs"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print(f"Loading model: {MODEL_NAME}")
# Fix the tokenizer initialization by using AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.config.decoder_start_token_id = tokenizer.pad_token_id

print(f"Device being used: {device}")
print(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")


print("Loading data...")
train_data = pd.read_csv('datasets/SumTablets_English_train.csv')

# For evaluation, use a separate test set if available, otherwise split from train
try:
    test_data = pd.read_csv('datasets/SumTablets_English_test.csv')
    print(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples")
except:
    print("No separate test file found. Will split from training data.")
    test_data = train_data

train_data

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256

class SumerianEnglishDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_len, max_target_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        
        # Filter out rows with missing data
        self.filtered_data = []

        for idx, row in data.iterrows():
            if isinstance(row['transliteration'], str) and isinstance(row['translation'], str):
                self.filtered_data.append({
                    'sumerian': row['transliteration'].replace('\n', ' '),
                    'english': row['translation'].replace('\n', ' ')
                })

    def __len__(self):
        return len(self.filtered_data)
    
    def __getitem__(self, idx):
        example = self.filtered_data[idx]
        source_text = f"translate Sumerian to English: {example['sumerian']}"
        target_text = example['english']

        # Tokenize input and target
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Replace padding tokens with -100 for loss calculation
        labels = target_encoding["input_ids"].squeeze(0)
        labels = labels.clone()  # Ensure a separate copy
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Return PyTorch tensors only
        return {
            "input_ids": source_encoding["input_ids"].squeeze(0).long(),
            "attention_mask": source_encoding["attention_mask"].squeeze(0).long(),
            "labels": labels.long()
        }

# Create training dataset
train_dataset = SumerianEnglishDataset(
    train_data, 
    tokenizer, 
    max_source_len=MAX_SOURCE_LENGTH, 
    max_target_len=MAX_TARGET_LENGTH
)

# Create test dataset
test_dataset = SumerianEnglishDataset(
    test_data, 
    tokenizer, 
    max_source_len=MAX_SOURCE_LENGTH, 
    max_target_len=MAX_TARGET_LENGTH
)


TRAIN_VALID_SPLIT = 0.1

# Split into training and validation sets
if TRAIN_VALID_SPLIT > 0:
    train_size = int((1 - TRAIN_VALID_SPLIT) * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, eval_dataset = random_split(train_dataset, [train_size, valid_size])
    print(f"Split into {train_size} training and {valid_size} validation samples")
else:
    train_dataset = train_dataset
    eval_dataset = None

# Training hyperparameters
NUM_EPOCHS = 15
LEARNING_RATE = 1e-5
BATCH_SIZE = 8

# --- Define evaluation metrics ---
bleu_metric = load("bleu")
meteor_metric = load("meteor")
rouge_metric = load("rouge")

def compute_metrics(eval_preds):    
    preds, labels = eval_preds
    
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up predictions and labels
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Print some examples for debugging
    print("\nSample predictions (first 2):")
    for i in range(min(2, len(decoded_preds))):
        print(f"Pred: '{decoded_preds[i]}'")
        print(f"Label: '{decoded_labels[i]}'")
        print("---")
    
    # Check if we have any valid predictions/labels to work with
    if not decoded_preds or not decoded_labels:
        print("Warning: Empty predictions or labels")
        return {
            "bleu": 0.0,
            "meteor": 0.0, 
            "rougeL": 0.0,
            "gen_len": 0.0
        }
    
    # Ensure all predictions and labels have content (not empty strings)
    valid_pairs = [(p, l) for p, l in zip(decoded_preds, decoded_labels) if p.strip() and l.strip()]
    if not valid_pairs:
        print("Warning: No valid (non-empty) prediction-label pairs found")
        return {
            "bleu": 0.0,
            "meteor": 0.0, 
            "rougeL": 0.0,
            "gen_len": 0.0
        }
    
    # Unzip the valid pairs
    valid_preds, valid_labels = zip(*valid_pairs)
    
    # Format references for BLEU
    references_for_bleu = [[label] for label in valid_labels]
    
    # Calculate metrics
    results = {}
    
    try:
        # BLEU
        bleu_results = bleu_metric.compute(predictions=valid_preds, references=references_for_bleu)
        results["bleu"] = bleu_results["bleu"] if bleu_results else 0.0
        
        # METEOR
        meteor_results = meteor_metric.compute(predictions=valid_preds, references=valid_labels)
        results["meteor"] = meteor_results["meteor"] if meteor_results else 0.0
        
        # ROUGE
        rouge_results = rouge_metric.compute(predictions=valid_preds, references=valid_labels)
        results["rougeL"] = rouge_results["rougeL"] if rouge_results else 0.0
        
        # Add prediction length
        pred_lens = [len(pred.split()) for pred in valid_preds]
        results["gen_len"] = np.mean(pred_lens) if pred_lens else 0.0
    
    except Exception as e:
        print(f"Error computing metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return zeros for all metrics if computation fails
        return {
            "bleu": 0.0,
            "meteor": 0.0, 
            "rougeL": 0.0,
            "gen_len": 0.0
        }
    
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in results.items()}
    
# --- Data collator ---
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

# --- 7. Training arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    logging_dir=LOG_DIR,
    logging_steps=100,
    save_strategy="epoch",
    fp16=False,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,                      # gradient clipping
    generation_max_length=MAX_TARGET_LENGTH,
    report_to="tensorboard",
    warmup_steps=500,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# --- 8. Initialize trainer ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# --- 9. Train the model ---
print("Starting training...")
trainer.train()

# --- 10. Save the model ---
print(f"Saving model to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)