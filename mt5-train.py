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
from evaluate import load

MODEL_NAME = "google/mt5-small"
OUTPUT_DIR = "./sumerian_mt5_model"
LOG_DIR = "./sumerian_mt5_logs"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Training hyperparameters
NUM_EPOCHS = 20
LEARNING_RATE = 5e-4
BATCH_SIZE = 8
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256
TRAIN_VALID_SPLIT = 0.1

# --- 2. Load and prepare data ---
print("Loading data...")
train_data = pd.read_csv('datasets/SumTablets_English_train.csv')

# For evaluation, use a separate test set if available, otherwise split from train
try:
    test_data = pd.read_csv('datasets/SumTablets_English_test.csv')
    print(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples")
except:
    print("No separate test file found. Will split from training data.")
    test_data = train_data

# Load tokenizer and model
print(f"Loading {MODEL_NAME}...")
tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using {device} for training")

# --- 3. Create dataset class ---
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
        
        print(f"Kept {len(self.filtered_data)} examples after filtering")
        
    def __len__(self):
        return len(self.filtered_data)
    
    def __getitem__(self, idx):
        example = self.filtered_data[idx]
        
        # For MT5, we prepend a task prefix to clarify the task
        source_text = f"translate Sumerian to English: {example['sumerian']}"
        target_text = example['english']
        
        # Tokenize inputs
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Replace padding token id's with -100 for loss calculation
        target_ids = target_encoding["input_ids"]
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": target_ids.squeeze()
        }

# --- 4. Prepare datasets ---
print("Creating datasets...")
full_dataset = SumerianEnglishDataset(
    train_data, 
    tokenizer, 
    max_source_len=MAX_SOURCE_LENGTH, 
    max_target_len=MAX_TARGET_LENGTH
)

# Split into training and validation sets
if TRAIN_VALID_SPLIT > 0:
    train_size = int((1 - TRAIN_VALID_SPLIT) * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, valid_size])
    print(f"Split into {train_size} training and {valid_size} validation samples")
else:
    train_dataset = full_dataset
    eval_dataset = None

# Create test dataset
test_dataset = SumerianEnglishDataset(
    test_data, 
    tokenizer, 
    max_source_len=MAX_SOURCE_LENGTH, 
    max_target_len=MAX_TARGET_LENGTH
)

# --- 5. Define evaluation metrics ---
def compute_metrics(eval_preds):
    bleu_metric = load("bleu")
    meteor_metric = load("meteor")
    rouge_metric = load("rouge")
    
    preds, labels = eval_preds
    
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up predictions and labels
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Format references for BLEU
    references_for_bleu = [[label] for label in decoded_labels]
    
    # Calculate metrics
    results = {}
    
    # BLEU
    bleu_results = bleu_metric.compute(predictions=decoded_preds, references=references_for_bleu)
    results["bleu"] = bleu_results["bleu"]
    
    # METEOR
    meteor_results = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    results["meteor"] = meteor_results["meteor"]
    
    # ROUGE
    rouge_results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    results["rougeL"] = rouge_results["rougeL"]
    
    # Add prediction length
    pred_lens = [len(pred.split()) for pred in decoded_preds]
    results["gen_len"] = np.mean(pred_lens)
    
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in results.items()}

# --- 6. Data collator ---
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
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,
    generation_max_length=MAX_TARGET_LENGTH,
    report_to="tensorboard"
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

# --- 11. Testing on examples ---
print("\nTesting on example data...")

def generate_translation(sumerian_text):
    input_text = f"translate Sumerian to English: {sumerian_text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    outputs = model.generate(
        input_ids=input_ids,
        max_length=MAX_TARGET_LENGTH,
        num_beams=4,
        length_penalty=0.6,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Test on 5 examples
for i, row in test_data.head(5).iterrows():
    if isinstance(row['transliteration'], str):
        sumerian_text = row['transliteration'].replace('\n', ' ')
        actual_translation = row['translation'].replace('\n', ' ') if isinstance(row['translation'], str) else "N/A"
        
        print(f"\nExample {i+1}:")
        print(f"Sumerian: {sumerian_text}")
        print(f"Actual Translation: {actual_translation}")
        
        generated_translation = generate_translation(sumerian_text)
        print(f"MT5 Translation: {generated_translation}")
        print("-" * 50)