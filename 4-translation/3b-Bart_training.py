import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import GenerationConfig
from transformers import EarlyStoppingCallback
from datasets import load_dataset, Dataset as HFDataset # Using Hugging Face's Dataset object
import pandas as pd
from load_dataset import preprocess_dataset
from compute_metrics import compute_metrics

MODEL_NAME = "facebook/bart-large"
MAX_INPUT_LENGTH = 512  # Max length for source sentences
MAX_TARGET_LENGTH = 512 # Max length for target sentences
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
NUM_TRAIN_EPOCHS = 50    # Number of training epochs
OUTPUT_DIR = "4-translation/bart_large_model" # Directory to save the fine-tuned model
LOGGING_DIR = "4-translation/bart_large_logs" # Directory for TensorBoard logs

# check if dirs exist, if not create them
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(device)

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

if not hasattr(model.config, "decoder_start_token_id") or model.config.decoder_start_token_id is None:
    model.config.decoder_start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

preprocessed_train = preprocess_dataset('datasets/SumTablets_English_train.csv')
preprocessed_val = preprocess_dataset('datasets/SumTablets_English_validation.csv')
preprocessed_test = preprocess_dataset('datasets/SumTablets_English_test.csv')

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

    # print flag if any of inputs or targets are none
    if any(x is None for x in inputs) or any(x is None for x in targets):
        print("Warning: Found None values in inputs or targets. This may affect training.")


    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")

    # Tokenize targets (English) using the newer approach
    labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")

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

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    gradient_accumulation_steps=1, 
    lr_scheduler_type="cosine", # Use cosine learning rate scheduler
    label_smoothing_factor=0.1,

    save_total_limit=1,         # Only keep the last checkpoint
    predict_with_generate=True, # Important for generation tasks like translation
    logging_dir=LOGGING_DIR,
    logging_steps=50,          # Log training loss every N steps (e.g. 100)
    eval_strategy="epoch", # Evaluate at the end of each epoch
    save_strategy="epoch",       # Save model at the end of each epoch
    load_best_model_at_end=True, # Optionally load the best model at the end of training
    metric_for_best_model="meteor", # Metric to determine the best model (e.g., 'bleu' if you add custom metrics)
    fp16=torch.cuda.is_available(), # Use mixed precision training if GPU is available
    report_to="tensorboard" # To visualize logs with TensorBoard
)

generation_config = GenerationConfig(
    max_length=MAX_TARGET_LENGTH,
    early_stopping=True,
    num_beams=4,
    no_repeat_ngram_size=3,
    forced_bos_token_id=0,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
)

# Assign it to the model
model.generation_config = generation_config

# --- 6. Data Collator ---
# The DataCollatorForSeq2Seq handles padding dynamically batch-wise for inputs and labels.
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --- 7. Initialize Trainer ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda p: compute_metrics(p, tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
# --- 8. Train the Model ---
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