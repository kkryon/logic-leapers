import os
import json
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import glob # Import glob for finding checkpoint directories

# --- Configuration ---
BASE_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct" # Or your original base model
TRAINING_OUTPUT_DIR = "cai6307-henrykobs/model/Qwen-7B-GRPO" # Directory containing 'checkpoint-XXXX' folders
EVALUATION_ROOT_DIR = "./folio_checkpoint_evaluation" # Main folder to store all evaluation results
BATCH_SIZE = 8 # Adjust based on A100 memory and model size
MAX_LENGTH = 1536 # Context window size
MAX_NEW_TOKENS = 1024 # Max tokens for generation

# --- Hugging Face Login ---
# Securely authenticate with Hugging Face using environment variable
# Ensure HF_TOKEN environment variable is set before running
try:
    login(os.environ.get("HF_TOKEN", ""))
    print("Hugging Face login successful.")
except Exception as e:
    print(f"Hugging Face login failed: {e}. Ensure HF_TOKEN is set.")
    # Decide if you want to exit or continue (might work for public models)
    # exit(1) 

# --- System Prompt ---
SYSTEM_PROMPT = """You are a logical reasoning assistant. You must use the exact format below:
<think>
Think through the problem step by step, analyzing the premises and conclusion carefully. 
Keep your thinking concise and limited to 5-6 sentences maximum.
</think>
<answer>
Provide ONLY ONE of these three options: True, False, or Uncertain.
</answer>
"""

# --- Helper Functions (Unchanged from your original) ---

def load_folio_dataset():
    """Load the FOLIO dataset validation split ONCE for all evaluations"""
    try:
        dataset = load_dataset("yale-nlp/FOLIO", split="validation")
        print(f"Loaded {len(dataset)} validation examples from FOLIO dataset")
        return dataset
    except Exception as e:
        print(f"Error loading FOLIO dataset: {e}")
        return None

def extract_answer_with_status(response):
    """
    Parse model responses and return the label plus the extraction method status.
    Status codes: 'TAG_SUCCESS', 'TAG_FALLBACK', 'REGEX_FALLBACK', 'KEYWORD_FALLBACK', 'DEFAULT_UNCERTAIN'
    """
    response = response.strip()
    # Try to extract content between <answer> tags
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    match = answer_pattern.search(response)
    
    if match:
        answer = match.group(1).strip().lower()
        if answer == "true": return "True", "TAG_SUCCESS"
        if answer == "false": return "False", "TAG_SUCCESS"
        if answer == "uncertain": return "Uncertain", "TAG_SUCCESS"
        
        # Fallback within the tag if exact match failed
        words = answer.split()
        if "true" in words: return "True", "TAG_FALLBACK"
        if "false" in words: return "False", "TAG_FALLBACK"
        if "uncertain" in words: return "Uncertain", "TAG_FALLBACK"

    # If no answer tag or no match inside, check the whole response (fallbacks)
    response_lower = response.lower()
    
    if re.search(r'\b(is|are|definitely|clearly|must be)\s+true\b', response_lower):
        return "True", "REGEX_FALLBACK"
    if re.search(r'\b(is|are|definitely|clearly|must be)\s+false\b', response_lower):
        return "False", "REGEX_FALLBACK"
    if re.search(r'\b(is|are|seems|could be|might be|possibly)\s+uncertain\b', response_lower):
        return "Uncertain", "REGEX_FALLBACK"
        
    words = response_lower.split()
    if "true" in words: return "True", "KEYWORD_FALLBACK"
    if "false" in words: return "False", "KEYWORD_FALLBACK"
    if "uncertain" in words: return "Uncertain", "KEYWORD_FALLBACK"
    
    # Default to Uncertain if nothing found
    return "Uncertain", "DEFAULT_UNCERTAIN"


def prepare_prompt(example, tokenizer):
    """
    Format each dataset example into a standardized prompt structure.
    Returns the formatted string ready for tokenization.
    """
    premises = example["premises"]
    conclusion = example["conclusion"]
    
    user_message = f"""Given the following premises, determine if the conclusion is True, False, or Uncertain.

Premises:
{premises}

Conclusion: 
{conclusion}"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    
    # Apply chat template to get the final formatted string
    try:
        prompt_string = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt_string
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return "Error: Prompt formatting failed"


# --- Generation Function (Modified for clarity and resource handling) ---

def generate_predictions_for_checkpoint(model_path, dataset, batch_size, max_length, max_new_tokens, checkpoint_output_dir):
    """
    Loads a specific model/checkpoint and generates predictions.

    Returns:
    - List of model responses for this checkpoint.
    """
    model = None
    tokenizer = None
    all_completions = []
    
    try:
        print(f"\n--- Evaluating: {model_path} ---")
        print(f"Loading model and tokenizer from: {model_path}")

        # GPU detection (redundant if checked globally, but informative per checkpoint)
        if torch.cuda.is_available():
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("Warning: No GPU detected. Running on CPU will be slow.")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("Set pad_token_id to eos_token_id")
        print(f"Tokenizer loaded. Padding side: {tokenizer.padding_side}")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model.eval()
        print(f"Model loaded onto device: {model.device}")
        
        # --- Prepare Prompts (using the already loaded dataset) ---
        print("Preparing prompts...")
        prompts = []
        for i, example in enumerate(dataset):
            prompt_str = prepare_prompt(example, tokenizer)
            prompts.append({"prompt": prompt_str})
            if prompt_str == "Error: Prompt formatting failed":
                 print(f"Warning: Failed to format prompt for example {i}")

        # --- Batch Generation ---
        print(f"Generating completions in batches of {batch_size}...")
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating for {os.path.basename(model_path)}"):
            batch_items = prompts[i:min(i + batch_size, len(prompts))]
            
            # Filter out errored prompts from this batch
            valid_prompts_batch = [item["prompt"] for item in batch_items if item["prompt"] != "Error: Prompt formatting failed"]
            original_indices = [idx for idx, item in enumerate(batch_items) if item["prompt"] != "Error: Prompt formatting failed"]

            batch_completions = ["Error: Invalid prompt"] * len(batch_items) # Initialize with error placeholders

            if not valid_prompts_batch:
                 all_completions.extend(batch_completions)
                 continue # Skip generation if batch has no valid prompts

            try:
                inputs = tokenizer(
                    valid_prompts_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(model.device)

                with torch.no_grad():
                   outputs = model.generate(
                       **inputs,
                       max_new_tokens=max_new_tokens,
                       do_sample=False,
                       temperature=1.0,
                       top_p=None,
                       top_k=None,
                       repetition_penalty=1.2,
                       pad_token_id=tokenizer.eos_token_id
                   )

                # Decode and store valid completions
                input_length = inputs.input_ids.shape[1]
                decoded_outputs = tokenizer.batch_decode(
                    outputs[:, input_length:], # Decode only generated tokens
                    skip_special_tokens=True
                )

                for original_idx, completion in zip(original_indices, decoded_outputs):
                    batch_completions[original_idx] = completion.strip()

            except Exception as e:
                print(f"\nError during generation for batch starting at {i}: {e}")
                # Mark all originally valid prompts in this batch as failed generation
                for original_idx in original_indices:
                    batch_completions[original_idx] = "Error: Generation failed"
            
            all_completions.extend(batch_completions)

            # Optional: Intermediate saving within a checkpoint's evaluation
            if (i // batch_size) % 10 == 0 and i > 0: # Save less frequently maybe
                 intermediate_path = os.path.join(checkpoint_output_dir, f"intermediate_responses_{i}.json")
                 print(f"Saving intermediate results to {intermediate_path}...")
                 try:
                     with open(intermediate_path, "w") as f:
                         json.dump(all_completions, f, indent=2)
                 except Exception as e_save:
                     print(f"Warning: Failed to save intermediate results: {e_save}")
        
        print(f"Finished generation for {model_path}.")

    except Exception as e:
        print(f"FATAL ERROR during evaluation for {model_path}: {e}")
        # Fill remaining completions with error if generation was interrupted
        num_missing = len(dataset) - len(all_completions)
        if num_missing > 0:
            print(f"Appending {num_missing} error messages due to failure.")
            all_completions.extend(["Error: Checkpoint evaluation failed"] * num_missing)
        # Ensure list length matches dataset length
        all_completions = all_completions[:len(dataset)]


    finally:
        # --- Resource Cleanup ---
        print(f"Cleaning up resources for {model_path}...")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")
        
    return all_completions


# --- Analysis and Reporting Functions (Mostly Unchanged) ---

def analyze_extraction_methods(statuses):
    """Analyzes distribution of how answers were extracted."""
    status_counts = {}
    for status in statuses:
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\nAnswer Extraction Method Distribution:")
    total = len(statuses)
    if total == 0:
        print("  No responses to analyze.")
        return
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count} ({count/total*100:.2f}%)")

def analyze_response_distribution(predictions):
    """Analyzes distribution of predicted labels (True/False/Uncertain)."""
    prediction_counts = {}
    for pred in predictions:
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
    
    print("\nPrediction distribution:")
    total = len(predictions)
    if total == 0:
        print("  No predictions to analyze.")
        return
    for pred, count in sorted(prediction_counts.items()):
        print(f"  {pred}: {count} ({count/total*100:.2f}%)")


def plot_confusion_matrix(cm, classes, output_path):
    """Generate visualization of confusion matrix."""
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Confusion matrix saved to {output_path}")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

# --- Main Evaluation Orchestration ---

def run_evaluation_for_checkpoint(model_path, dataset, checkpoint_output_dir):
    """
    Runs the full evaluation pipeline for a single model/checkpoint.
    """
    os.makedirs(checkpoint_output_dir, exist_ok=True)
    print(f"Output directory for this checkpoint: {checkpoint_output_dir}")

    # 1. Generate Predictions
    responses = generate_predictions_for_checkpoint(
        model_path, dataset, BATCH_SIZE, MAX_LENGTH, MAX_NEW_TOKENS, checkpoint_output_dir
    )

    # Save raw responses
    raw_responses_path = os.path.join(checkpoint_output_dir, "raw_responses.json")
    try:
        with open(raw_responses_path, "w") as f:
            json.dump(responses, f, indent=2)
        print(f"Raw responses saved to {raw_responses_path}")
    except Exception as e:
        print(f"Error saving raw responses: {e}")

    # 2. Parse Predictions and Analyze Extraction
    predictions = []
    extraction_statuses = []
    print("Parsing predictions...")
    for response in responses:
        # Handle potential errors from generation phase
        if isinstance(response, str) and response.startswith("Error:"):
            pred, status = "Uncertain", "GENERATION_ERROR" # Assign a specific status
        else:
             pred, status = extract_answer_with_status(response)
        predictions.append(pred)
        extraction_statuses.append(status)
    
    analyze_extraction_methods(extraction_statuses)
    analyze_response_distribution(predictions)

    # 3. Get Ground Truth Labels
    labels = [example["label"] for example in dataset]

    # Ensure predictions list length matches labels list length
    if len(predictions) != len(labels):
         print(f"ERROR: Mismatch between predictions ({len(predictions)}) and labels ({len(labels)}). Check generation errors.")
         # Optionally, truncate or pad predictions to match labels, or skip metrics calculation
         # For now, we'll stop metric calculation for this checkpoint if lengths mismatch
         results = {"error": "Prediction/Label length mismatch", "model_path": model_path}
         results_path = os.path.join(checkpoint_output_dir, "evaluation_results.json")
         with open(results_path, "w") as f:
             json.dump(results, f, indent=2)
         print(f"Skipping metrics calculation due to length mismatch. Results saved to {results_path}")
         return # Exit evaluation for this checkpoint
            
    # 4. Calculate Metrics
    print("Calculating metrics...")
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(
        labels, 
        predictions,
        labels=["True", "False", "Uncertain"],
        output_dict=True,
        zero_division=0 # Avoid warnings if a class has no predictions
    )
    cm = confusion_matrix(labels, predictions, labels=["True", "False", "Uncertain"])

    # 5. Save Results
    results = {
        "model_path": model_path,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "extraction_method_summary": {status: extraction_statuses.count(status) for status in set(extraction_statuses)}
    }
    results_path = os.path.join(checkpoint_output_dir, "evaluation_results.json")
    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation metrics saved to {results_path}")
    except Exception as e:
        print(f"Error saving evaluation metrics: {e}")

    # 6. Plot Confusion Matrix
    cm_path = os.path.join(checkpoint_output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes=["True", "False", "Uncertain"], output_path=cm_path)

    # 7. Save Detailed Predictions
    print("Saving detailed predictions...")
    prediction_details = []
    for i, (example, pred, label, response, status) in enumerate(zip(dataset, predictions, labels, responses, extraction_statuses)):
        prediction_details.append({
            "example_id": i,
            "premises": example["premises"],
            "conclusion": example["conclusion"],
            "true_label": label,
            "predicted_label": pred,
            "extraction_status": status,
            "is_correct": pred == label if status != "GENERATION_ERROR" else None, # Correctness is undefined if generation failed
            "model_response": response
        })
    
    details_path = os.path.join(checkpoint_output_dir, "prediction_details.json")
    try:
        with open(details_path, "w") as f:
            json.dump(prediction_details, f, indent=2)
        print(f"Detailed predictions saved to {details_path}")
    except Exception as e:
        print(f"Error saving detailed predictions: {e}")

    print(f"\n--- Finished evaluation for: {model_path} ---")


if __name__ == "__main__":
    # Create the main evaluation directory
    os.makedirs(EVALUATION_ROOT_DIR, exist_ok=True)

    # --- Load Dataset Once ---
    print("Loading FOLIO dataset...")
    folio_val_dataset = load_folio_dataset()

    if folio_val_dataset is None:
        print("Exiting due to dataset loading failure.")
        exit(1)

    # --- Identify Models/Checkpoints to Evaluate ---
    models_to_evaluate = []

    # 1. Add the base model
    if BASE_MODEL_PATH and os.path.exists(BASE_MODEL_PATH): # Check if path exists locally first
         models_to_evaluate.append(BASE_MODEL_PATH)
    elif BASE_MODEL_PATH: # Assume it's a Hugging Face Hub ID if not local
         print(f"Base model '{BASE_MODEL_PATH}' assumed to be on Hugging Face Hub.")
         models_to_evaluate.append(BASE_MODEL_PATH)
    else:
         print("No BASE_MODEL_PATH specified, skipping base model evaluation.")


    # 2. Find checkpoint directories
    if os.path.isdir(TRAINING_OUTPUT_DIR):
        checkpoint_pattern = os.path.join(TRAINING_OUTPUT_DIR, "checkpoint-*")
        checkpoint_dirs = sorted(glob.glob(checkpoint_pattern)) # Sort checkpoints naturally
        
        # Filter out any non-directory matches if necessary
        checkpoint_dirs = [d for d in checkpoint_dirs if os.path.isdir(d)] 
        
        if checkpoint_dirs:
            print(f"Found {len(checkpoint_dirs)} checkpoints in {TRAINING_OUTPUT_DIR}:")
            for chkpt in checkpoint_dirs:
                print(f"  - {os.path.basename(chkpt)}")
            models_to_evaluate.extend(checkpoint_dirs)
        else:
            print(f"No checkpoint directories matching 'checkpoint-*' found in {TRAINING_OUTPUT_DIR}.")
    else:
        print(f"Training output directory '{TRAINING_OUTPUT_DIR}' not found or not a directory. Skipping checkpoint evaluation.")

    # --- Run Evaluation Loop ---
    if not models_to_evaluate:
        print("No models or checkpoints found to evaluate. Exiting.")
        exit(1)
        
    print(f"\nStarting evaluation for {len(models_to_evaluate)} model(s)/checkpoint(s)...")

    for model_path in models_to_evaluate:
        # Determine a unique name for the output directory
        if model_path == BASE_MODEL_PATH:
             # Handle potential slashes in HF model IDs
             model_name = BASE_MODEL_PATH.replace("/", "_") + "_BASE" 
        else:
             model_name = os.path.basename(model_path) # e.g., "checkpoint-1000"

        checkpoint_output_dir = os.path.join(EVALUATION_ROOT_DIR, model_name)
        
        run_evaluation_for_checkpoint(model_path, folio_val_dataset, checkpoint_output_dir)
        print("-" * 50) # Separator between checkpoint evaluations

    print("\nAll checkpoint evaluations complete!")
    print(f"Results saved in subdirectories under: {EVALUATION_ROOT_DIR}")