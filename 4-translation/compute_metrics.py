from evaluate import load
import numpy as np
import torch

# Load the evaluation metrics
bleu_metric = load("bleu")
meteor_metric = load("meteor")
rouge_metric = load("rouge")

def compute_metrics(eval_preds, tokenizer):
    preds, label_ids = eval_preds

    actual_token_ids = preds
    if isinstance(preds, tuple):
        actual_token_ids = preds[0]

    if isinstance(actual_token_ids, (np.ndarray, torch.Tensor)) and actual_token_ids.ndim == 3:
        if isinstance(actual_token_ids, torch.Tensor):
            actual_token_ids = actual_token_ids.cpu().numpy()
        actual_token_ids = np.argmax(actual_token_ids, axis=-1)

    # If pad_token_id is None, use eos_token_id as a fallback
    if tokenizer.pad_token_id is None:
        print("WARNING: tokenizer.pad_token_id is None. Using eos_token_id as fallback for replacing -100 in preds.")
        preds_replacement_pad_id = tokenizer.eos_token_id 
    else:
        preds_replacement_pad_id = tokenizer.pad_token_id
    actual_token_ids = np.where(actual_token_ids == -100, preds_replacement_pad_id, actual_token_ids)
    
    # Replace -100 (ignore_index) in label_ids with pad_token_id for proper decoding
    if tokenizer.pad_token_id is None:
        print("WARNING: tokenizer.pad_token_id is None. Using eos_token_id as fallback for replacing -100 in labels.")
        labels_replacement_pad_id = tokenizer.eos_token_id
    else:
        labels_replacement_pad_id = tokenizer.pad_token_id 
    processed_label_ids = np.where(label_ids == -100, labels_replacement_pad_id, label_ids)

    try:
        if isinstance(actual_token_ids, np.ndarray):
            actual_token_ids = actual_token_ids.astype(np.int32)
        decoded_preds = tokenizer.batch_decode(actual_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        if isinstance(processed_label_ids, np.ndarray):
            processed_label_ids = processed_label_ids.astype(np.int32)
        decoded_labels = tokenizer.batch_decode(processed_label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    except Exception as e:
        print(f"ERROR during tokenizer.batch_decode:")
        raise e

    # Clean up the predictions and labels
    cleaned_preds = [pred.split("English:")[-1].replace("<|endoftext|>", "").strip() if "English:" in pred else pred.replace("<|endoftext|>", "").strip() for pred in decoded_preds]
    cleaned_labels = [label.split("English:")[-1].replace("<|endoftext|>", "").strip() if "English:" in label else label.replace("<|endoftext|>", "").strip() for label in decoded_labels]

    # Filter out empty label strings AFTER cleaning, as they can be problematic for metrics
    # And ensure corresponding predictions are also removed to maintain alignment.
    filtered_preds = []
    filtered_list_of_lists_labels = []
    for pred, label_text in zip(cleaned_preds, cleaned_labels):
        if label_text:
            filtered_preds.append(pred)
            filtered_list_of_lists_labels.append([label_text])
        else:
            print(f"WARNING: Found empty label after cleaning. Skipping prediction: '{pred}'")

    if not filtered_list_of_lists_labels: # If all labels were empty and filtered out
        print("WARNING: No valid references left after filtering empty labels. Metrics will be 0.")
        return {"bleu": 0.0, "meteor": 0.0, "rougeL": 0.0, "gen_len": 0.0}

    # Calculate the metrics using the filtered predictions and labels
    results = {}
    try:
        # Metrics are calculated on filtered lists
        bleu_score_dict = bleu_metric.compute(predictions=filtered_preds, references=filtered_list_of_lists_labels)
        results["bleu"] = bleu_score_dict.get("score", bleu_score_dict.get("bleu", 0.0))

        # For meteor and rouge, references should be a list of strings, not list of lists
        filtered_cleaned_labels_for_meteor_rouge = [item[0] for item in filtered_list_of_lists_labels]
        meteor_score_dict = meteor_metric.compute(predictions=filtered_preds, references=filtered_cleaned_labels_for_meteor_rouge)
        results["meteor"] = meteor_score_dict["meteor"]

        rouge_score_dict = rouge_metric.compute(predictions=filtered_preds, references=filtered_cleaned_labels_for_meteor_rouge)
        results["rougeL"] = rouge_score_dict.get("rougeLsum", rouge_score_dict.get("rougeL", 0.0))
    
    except Exception as e:
        print(f"WARNING: Error calculating metrics (pre-filtering): {e}")
        results["bleu"] = results.get("bleu", 0.0)
        results["meteor"] = results.get("meteor", 0.0)
        results["rougeL"] = results.get("rougeL", 0.0)


    # Calculate the average generation length
    try:
        prediction_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in filtered_preds] if filtered_preds else [0]
        results["gen_len"] = np.mean(prediction_lens)
    except Exception as e:
        print(f"WARNING: Error calculating average generation length: {e}")
        results["gen_len"] = 0.0

    # Round the results to 4 decimal places for floats
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in results.items()}