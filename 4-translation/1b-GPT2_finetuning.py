import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)


model_name = 'gpt2-medium'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model = GPT2LMHeadModel.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


train_data = pd.read_csv('datasets/SumTablets_English_train.csv')
test_data = pd.read_csv('datasets/SumTablets_English_train.csv')

# Format the data for GPT-2:
# We'll combine Sumerian and English with a separator.
# GPT-2 will learn to generate the English part after seeing "English: ".
# The <|endoftext|> token is GPT-2's standard end-of-sequence token.
formatted_texts = []
for index, row in train_data.iterrows():
    sumerian_texts = row['transliteration']
    english_translations = row['translation']
    if isinstance(sumerian_texts, str) and isinstance(english_translations, str):
        sumerian_texts = sumerian_texts.replace('\n', ' ')
        english_translations = english_translations.replace('\n', ' ')
        formatted_texts.append(f"Sumerian: {sumerian_texts}\nEnglish: {english_translations}<|endoftext|>")
print(f"Loaded {len(formatted_texts)} formatted examples.")

lengths = [len(text.split()) for text in formatted_texts]
print(lengths)
mean_length = np.mean(lengths)
print(f"Mean length of the texts: {mean_length} words")
print(f"Percentage of texts longer than 528 words: {sum(length > 528 for length in lengths) / len(lengths) * 100:.2f}%")

# remove texts longer than 528 words
formatted_texts = [text for text in formatted_texts if len(text.split()) <= 528]
print(len(formatted_texts), "texts after filtering by length.")

print(f"\nExample formatted text:\n{formatted_texts[0]}")


# --- 1. Configuration & Parameters ---
MODEL_NAME = 'gpt2-medium'
OUTPUT_DIR = './sumerian_gpt2_finetuned' # Directory to save the fine-tuned model
LOG_DIR = './logs'                     # Directory for training logs

# Training hyperparameters (adjust these based on your dataset size and resources)
NUM_EPOCHS = 20                        # Number of training epochs
LEARNING_RATE = 3e-5                   # Learning rate
WARMUP_RATIO = 0.1                     # Number of warmup steps for learning rate scheduler
WEIGHT_DECAY = 0.01                    # Weight decay
MAX_LENGTH = 528                       # Maximum sequence length for tokenizer
TRAIN_VALID_SPLIT = 0.1                # Proportion of data to use for validation


class SumerianEnglishDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        self.encodings = []
        for text in texts:
            
            # Tokenize the combined text
            # truncation=True ensures that sequences longer than max_length are cut.
            # padding='max_length' pads shorter sequences to max_length.
            # return_tensors='pt' returns PyTorch tensors.
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,     # Truncate to max_length
                padding="max_length",           # Ensure all sequences have the same length for batching
                return_attention_mask=True,     # Return attention masks
                return_tensors='pt'             # Explicitly specify to return PyTorch tensors
            )
            
            # For language modeling, the 'labels' are typically the same as 'input_ids'.
            self.encodings.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze()
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        # The labels are the input_ids and the model is trained to predict the next token in the sequence.
        # The DataCollatorForLanguageModeling will shift them appropriately.
        return {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"], "labels": item["input_ids"].clone()}

# Create the full dataset
full_dataset = SumerianEnglishDataset(formatted_texts, tokenizer, MAX_LENGTH)

# Split into training and validation sets
if TRAIN_VALID_SPLIT > 0:
    num_train = int((1 - TRAIN_VALID_SPLIT) * len(full_dataset))
    num_valid = len(full_dataset) - num_train
    train_dataset, eval_dataset = random_split(full_dataset, [num_train, num_valid])
    print(f"Split dataset into {len(train_dataset)} training samples and {len(eval_dataset)} validation samples.")
else:
    train_dataset = full_dataset
    eval_dataset = None     # No validation
    print(f"Using all {len(train_dataset)} samples for training. No validation set.")


from evaluate import load
import numpy as np

# load the evaluation metrics
bleu_metric = load("bleu")
meteor_metric = load("meteor")
rouge_metric = load("rouge")

# maybe also chrf but we need to install evaluate e sacrebleu
# chrf_metric = load("chrf")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    print(f"DEBUG: Tipo iniziale di preds: {type(preds)}")
    if hasattr(preds, 'shape'):
        print(f"DEBUG: Shape iniziale di preds: {preds.shape}")
    elif isinstance(preds, (list, tuple)):
        print(f"DEBUG: Lunghezza iniziale di preds: {len(preds)}")


    # --- INIZIO PARTE CRUCIALE PER GESTIRE PREDS ---
    actual_token_ids = preds # Rinominiamo per chiarezza

    # Caso 1: preds potrebbe essere una tupla (comune output di model.generate())
    # Le sequenze di token generate sono di solito il primo elemento.
    if isinstance(preds, tuple):
        print("DEBUG: preds è una tupla, prendo il primo elemento.")
        actual_token_ids = preds[0]

    print(f"DEBUG: Tipo di actual_token_ids dopo il check della tupla: {type(actual_token_ids)}")
    if hasattr(actual_token_ids, 'shape'):
        print(f"DEBUG: Shape di actual_token_ids dopo il check della tupla: {actual_token_ids.shape}")


    # Caso 2: actual_token_ids potrebbero essere logits (es. shape: batch_size, seq_len, vocab_size)
    # Convertiamo i logits in ID token usando argmax.
    # Nota: Questo è greedy decoding. Per beam search, etc., il Trainer DEVE usare model.generate().
    # Il check ndim == 3 è un buon indicatore di logits per modelli di linguaggio.
    if isinstance(actual_token_ids, (np.ndarray, torch.Tensor)) and actual_token_ids.ndim == 3:
        print("DEBUG: actual_token_ids sembrano logits, applico argmax.")
        if isinstance(actual_token_ids, torch.Tensor): # Se è un tensore PyTorch (magari su GPU)
            actual_token_ids = actual_token_ids.cpu().numpy() # Sposta su CPU e converti in NumPy
        actual_token_ids = np.argmax(actual_token_ids, axis=-1) # Prendi gli ID dei token con probabilità massima
        print(f"DEBUG: Shape di actual_token_ids dopo argmax: {actual_token_ids.shape}")

    # A questo punto, actual_token_ids DOVREBBE essere un array 2D (batch_size, seq_len) di ID token
    # o una lista di liste di ID token.
    # --- FINE PARTE CRUCIALE PER GESTIRE PREDS ---

    # Sostituisci -100 nelle etichette (questo è corretto)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decodifica i token in testo
    try:
        # 'actual_token_ids' ora dovrebbe avere il formato corretto
        decoded_preds = tokenizer.batch_decode(actual_token_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        print(f"ERRORE durante tokenizer.batch_decode:")
        print(f"  Tipo di actual_token_ids: {type(actual_token_ids)}")
        if hasattr(actual_token_ids, 'shape'): print(f"  Shape di actual_token_ids: {actual_token_ids.shape}")
        if hasattr(actual_token_ids, 'dtype'): print(f"  Dtype di actual_token_ids: {actual_token_ids.dtype}")
        print(f"  Esempio di un elemento in actual_token_ids (se lista/array): {actual_token_ids[0] if len(actual_token_ids)>0 else 'N/A'}")
        raise e


    # Pulizia del testo (la tua logica qui sembra un buon punto di partenza, da adattare)
    cleaned_preds = [pred.split("English:")[-1].replace("<|endoftext|>", "").strip() if "English:" in pred else pred.replace("<|endoftext|>", "").strip() for pred in decoded_preds]
    cleaned_labels = [label.split("English:")[-1].replace("<|endoftext|>", "").strip() if "English:" in label else label.replace("<|endoftext|>", "").strip() for label in decoded_labels]

    list_of_lists_labels = [[label] for label in cleaned_labels]
    results = {}

    try:
        bleu_score_dict = bleu_metric.compute(predictions=cleaned_preds, references=list_of_lists_labels)
        # 'sacrebleu' (usato da evaluate.load("sacrebleu")) di solito restituisce il punteggio in 'score'
        # versioni più vecchie o altre implementazioni potrebbero usare 'bleu'
        results["bleu"] = bleu_score_dict.get("score", bleu_score_dict.get("bleu", 0.0))

        meteor_score_dict = meteor_metric.compute(predictions=cleaned_preds, references=cleaned_labels)
        results["meteor"] = meteor_score_dict["meteor"]

        rouge_score_dict = rouge_metric.compute(predictions=cleaned_preds, references=cleaned_labels)
        # ROUGE di 'evaluate' restituisce diversi score, rougeLsum è spesso usato per riassunti/traduzioni
        results["rougeL"] = rouge_score_dict.get("rougeLsum", rouge_score_dict.get("rougeL", 0.0)) 
    except Exception as e:
        print(f"AVVISO: Errore nel calcolo di una metrica: {e}")
        print(f"  cleaned_preds (primi 2): {cleaned_preds[:2]}")
        print(f"  list_of_lists_labels (primi 2): {list_of_lists_labels[:2]}")
        # Imposta valori di default se il calcolo fallisce per non bloccare tutto
        results["bleu"] = results.get("bleu", 0.0)
        results["meteor"] = results.get("meteor", 0.0)
        results["rougeL"] = results.get("rougeL", 0.0)


    # Calcolo lunghezza (attenzione: tokenizer.encode su testo già pulito potrebbe non essere l'ideale
    # se vuoi la lunghezza originale dei token generati, meglio contare gli ID in actual_token_ids prima di skip_special_tokens)
    # Ma per una stima della lunghezza del testo generato va bene.
    try:
        prediction_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in cleaned_preds]
        results["gen_len"] = np.mean(prediction_lens) if prediction_lens else 0.0
    except Exception as e:
        print(f"AVVISO: Errore nel calcolo di gen_len: {e}")
        results["gen_len"] = 0.0

    return {k: round(v, 4) if isinstance(v, float) else v for k, v in results.items()}



# Set the pad_token_id in the model configuration (important for generation and padding)
model.config.pad_token_id = tokenizer.pad_token_id
print(f"Set model.config.pad_token_id to {tokenizer.pad_token_id}")

# The DataCollatorForLanguageModeling will automatically create batches and
# shift the input_ids to create labels for causal language modeling (predicting the next token).
# It also handles padding. `mlm=False` means we are doing Causal Language Modeling (CLM), not Masked Language Modeling (MLM).
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal Language Modeling for GPT-2
)


training_args = TrainingArguments(
    num_train_epochs=NUM_EPOCHS,                        # Total number of training epochs
    per_device_train_batch_size=2,                      # Batch size per device during training
    per_device_eval_batch_size=2,                       # Batch size for evaluation
    eval_accumulation_steps=4,                          # Number of steps to accumulate for evaluation (to save memory)
    warmup_ratio=WARMUP_RATIO,                          # Warmup ratio for learning rate scheduler
    weight_decay=WEIGHT_DECAY,                          # Strength of weight decay
    learning_rate=LEARNING_RATE,
    gradient_checkpointing=True,                        # Enable gradient checkpointing to save memory

    output_dir=OUTPUT_DIR,                              # Directory to save model checkpoints and outputs
    logging_dir=LOG_DIR,                                # Directory for storing logs
    
    eval_strategy="epoch" if eval_dataset else "no",    # Evaluate at the end of each epoch if eval_dataset exists
    save_strategy="epoch",                              # Save a checkpoint at the end of each epoch
    
    load_best_model_at_end=True if eval_dataset else False, # Load the best model found during training (based on eval loss)
    metric_for_best_model="bleu" if eval_dataset else None, # Metric to use for determining the best model
    greater_is_better=True if eval_dataset else None,   # Whether a higher metric is better (for BLEU, it is)
    fp16=torch.cuda.is_available(),                     # Use 16-bit (mixed) precision training if a GPU is available
    report_to="tensorboard",                            # Report metrics to TensorBoard
    save_total_limit=2,                                 # Limit the total amount of checkpoints. Deletes the older checkpoints.
    
    gradient_accumulation_steps=2,                      # Gradient accumulation steps (if you want to simulate larger batch sizes)
    lr_scheduler_type="linear",                         # Learning rate scheduler type
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # Function to compute metrics during evaluation
)

print("Starting fine-tuning...")
try:
    trainer.train(resume_from_checkpoint=True)
    print("Fine-tuning completed.")

except Exception as e:
    print(f"An error occurred during training: {e}")
    raise e

print(f"Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model and tokenizer saved to {OUTPUT_DIR}")
