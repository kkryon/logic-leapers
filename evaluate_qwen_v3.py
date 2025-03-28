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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re

# Securely authenticate with Hugging Face using environment variable
login(os.environ.get("HF_TOKEN", ""))

# System prompt defining task format and constraints
SYSTEM_PROMPT = """You are a logical reasoning assistant. You must use the exact format below:
<think>
Think through the problem step by step, analyzing the premises and conclusion carefully. 
Keep your thinking concise and limited to 5-6 sentences maximum.
</think>
<answer>
Provide ONLY ONE of these three options: True, False, or Uncertain.
</answer>
"""

def load_folio_dataset():
    """Load the FOLIO dataset validation split for logical reasoning evaluation"""
    try:
        dataset = load_dataset("yale-nlp/FOLIO", split="validation")
        print(f"Loaded {len(dataset)} validation examples from FOLIO dataset")
        return dataset
    except Exception as e:
        print(f"Error loading FOLIO dataset: {e}")
        return None

def extract_answer(response):
    """
    Parse model responses to extract standardized True/False/Uncertain answers
    Uses regex patterns to find the answer within tags or in the full response
    """
    response = response.strip()
    
    # Try to extract content between <answer> tags
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    match = answer_pattern.search(response)
    
    if match:
        # Found the answer tag, extract and clean the content
        answer = match.group(1).strip().lower()
        
        # Check for exact matches
        if answer == "true":
            return "True"
        elif answer == "false":
            return "False"
        elif answer == "uncertain":
            return "Uncertain"
        
        # If there are multiple words in the answer, check for any of our target words
        words = answer.split()
        if "true" in words:
            return "True"
        elif "false" in words:
            return "False"
        elif "uncertain" in words:
            return "Uncertain"
    
    # If no answer tag or no match inside the tag, check the whole response
    response_lower = response.lower()
    
    # Check for phrases indicating logical judgment
    if re.search(r'\b(is|are|definitely|clearly|must be)\s+true\b', response_lower):
        return "True"
    elif re.search(r'\b(is|are|definitely|clearly|must be)\s+false\b', response_lower):
        return "False"
    elif re.search(r'\b(is|are|seems|could be|might be|possibly)\s+uncertain\b', response_lower):
        return "Uncertain"
    
    # Simple word check as fallback
    words = response_lower.split()
    if "true" in words:
        return "True"
    elif "false" in words:
        return "False"
    elif "uncertain" in words:
        return "Uncertain"
    
    # Default to Uncertain if no clear answer
    return "Uncertain"

def prepare_prompt(example):
    """
    Format each dataset example into a standardized prompt structure
    for logical reasoning evaluation
    """
    premises = example["premises"]
    conclusion = example["conclusion"]
    
    # Create the prompt using the format from train_grpo.py
    user_message = f"""Given the following premises, determine if the conclusion is True, False, or Uncertain.

Premises:
{premises}

Conclusion: 
{conclusion}"""
    
    # Format as chat messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    
    return messages

def prepare_simplified_prompt(premises, conclusion):
    """
    Create shorter prompts for cases where the original exceeds token limits
    Truncates premises to first 500 chars to fit within context window
    """
    # Create a much simpler prompt
    simple_message = f"Is this conclusion: '{conclusion}' True, False, or Uncertain based on these premises: {premises[:500]}...?"
    
    # Format as chat message
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": simple_message}
    ]
    
    return messages

def generate_predictions(model_path, dataset, batch_size=8):
    """
    Generate predictions using the specified LLM
    
    Parameters:
    - model_path: HuggingFace model ID
    - dataset: FOLIO dataset to evaluate
    - batch_size: Number of examples to process at once (8 optimized for A100)
    
    Returns:
    - List of model responses
    """
    import torch
    
    print(f"Loading model: {model_path}")
    
    # GPU detection and reporting for performance monitoring
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Set CUDA device explicitly
        torch.cuda.set_device(0)
        print(f"Using CUDA device: {torch.cuda.current_device()}")
    else:
        print("No GPU detected. Running on CPU only.")
    
    # Load model with optimizations for A100 GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,     # FP16 for memory efficiency and speed
        device_map="auto",             # Automatic tensor placement across devices
        low_cpu_mem_usage=True         # Minimize CPU memory during loading
    )
    
    # Load tokenizer with appropriate settings for decoder-only models
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.padding_side = 'left'  # Left padding is optimal for causal language models
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    print(f"Tokenizer padding side set to: {tokenizer.padding_side}")
    
    # Set model to evaluation mode to disable dropout
    model.eval()
    
    # Process in batches for memory efficiency
    all_completions = []
    
    # Create a dataset class for efficient prompt processing
    class PromptDataset:
        def __init__(self, dataset, tokenizer):
            self.examples = []
            self.tokenizer = tokenizer
            
            # Pre-process all prompts at once
            for i, example in enumerate(dataset):
                try:
                    messages = prepare_prompt(example)
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    self.examples.append({"prompt": prompt})
                except Exception as e:
                    print(f"Error preparing example {i}: {e}")
                    self.examples.append({"prompt": "Error"})
                    
        def __len__(self):
            return len(self.examples)
            
        def __getitem__(self, idx):
            return self.examples[idx]
    
    # Create the dataset
    print("Preparing prompts...")
    prompt_dataset = PromptDataset(dataset, tokenizer)
    
    # Function to process a single batch
    def process_batch(batch_prompts):
        batch_completions = []
        
        # Group non-error prompts
        valid_prompts = []
        valid_indices = []
        
        for i, item in enumerate(batch_prompts):
            if item["prompt"] != "Error":
                valid_prompts.append(item["prompt"])
                valid_indices.append(i)
            else:
                batch_completions.append("Error: Invalid prompt")
        
        if not valid_prompts:
            return batch_completions
            
        # Tokenize all valid prompts
        try:
            # Configuration for A100 GPU with large memory capacity
            inputs = tokenizer(
                valid_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1536  # Increased context window for A100
            ).to(model.device)
            
            # Print device placement for debug purposes
            if not hasattr(process_batch, "device_printed"):
                device = next(model.parameters()).device
                print(f"Model is on device: {device}")
                process_batch.device_printed = True
            
            # Generate with optimized parameters for A100
            with torch.no_grad():
                # Use CUDA streams for parallel processing
                with torch.cuda.stream(torch.cuda.Stream()):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,        # Maximum tokens to generate
                        do_sample=False,            # Deterministic generation (greedy)
                        temperature=1.0,            # No temperature adjustment
                        top_p=None,                 # No nucleus sampling
                        top_k=None,                 # No top-k filtering
                        repetition_penalty=1.2,     # Slight penalty for repeating tokens
                        pad_token_id=tokenizer.eos_token_id  # Proper padding token
                    )
            
            # Process each output
            for i, output_idx in enumerate(range(len(valid_indices))):
                # Get input length
                input_length = inputs.input_ids.shape[1]
                
                # Decode only the new tokens (exclude prompt tokens)
                completion = tokenizer.decode(
                    outputs[i, input_length:],
                    skip_special_tokens=True
                ).strip()
                
                # Insert at the correct position
                while len(batch_completions) <= valid_indices[i]:
                    batch_completions.append("Error: Placeholder")
                batch_completions[valid_indices[i]] = completion
                
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # Fill remaining slots with error message
            for idx in valid_indices:
                if idx >= len(batch_completions):
                    batch_completions.extend(["Error: Generation failed"] * (idx - len(batch_completions) + 1))
                else:
                    batch_completions[idx] = "Error: Generation failed"
        
        # Ensure all positions are filled
        while len(batch_completions) < len(batch_prompts):
            batch_completions.append("Error: Missing completion")
            
        return batch_completions
    
    # Test with single example first to validate setup
    print("Testing with a single example...")
    single_example = [prompt_dataset[0]]
    single_result = process_batch(single_example)
    print(f"Test successful. Generated response: {single_result[0][:50]}...")
    
    # Process all examples in batches
    print("Generating completions...")
    all_completions = []
    
    # Process dataset with periodic cache clearing to prevent memory fragmentation
    for i in tqdm(range(0, len(prompt_dataset), batch_size)):
        batch_prompts = [prompt_dataset[j] for j in range(i, min(i + batch_size, len(prompt_dataset)))]
        batch_completions = process_batch(batch_prompts)
        all_completions.extend(batch_completions)
        
        # Save intermediate results every 5 batches for checkpoint recovery
        if (i // batch_size) % 5 == 0 and i > 0:
            print(f"Completed {i}/{len(prompt_dataset)} examples. Saving intermediate results...")
            with open(f"./folio_evaluation_results/intermediate_responses_{i}.json", "w") as f:
                json.dump(all_completions, f, indent=2)
            
            # Clear CUDA cache to prevent memory fragmentation on long runs
            torch.cuda.empty_cache()
    
    return all_completions

def analyze_response_distribution(responses, predictions):
    """
    Analyze format compliance and distribution of model outputs
    Helps identify systematic errors in response formatting
    """
    # For original responses
    answer_tag_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    answers_found = 0
    
    for response in responses:
        if answer_tag_pattern.search(response):
            answers_found += 1
    
    print(f"Responses with <answer> tags: {answers_found}/{len(responses)} ({answers_found/len(responses)*100:.2f}%)")
    
    # For parsed predictions
    prediction_counts = {}
    for pred in predictions:
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
    
    print("\nPrediction distribution:")
    for pred, count in sorted(prediction_counts.items()):
        print(f"  {pred}: {count} ({count/len(predictions)*100:.2f}%)")

def plot_confusion_matrix(cm, classes, output_path):
    """Generate visualization of confusion matrix for error analysis"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_evaluation():
    """
    Main evaluation pipeline for assessing model performance on FOLIO dataset
    Handles dataset loading, prediction generation, and results analysis
    """
    # Configuration
    model_path = "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B parameter model from Qwen series
    output_dir = "./folio_evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_folio_dataset()
    if dataset is None:
        return
    
    # Debug: Print dataset structure information
    print(f"Dataset type: {type(dataset)}")
    if hasattr(dataset, "__getitem__"):
        first_item = dataset[0]
        print(f"First item type: {type(first_item)}")
        print(f"First item content: {first_item}")
        if hasattr(first_item, "keys"):
            print(f"First item keys: {first_item.keys()}")
    
    # Generate predictions for all examples
    print(f"Generating predictions for {len(dataset)} examples using A100 GPU...")
    
    # Generate predictions using transformers
    responses = generate_predictions(model_path, dataset)
    
    # Save raw responses for debugging
    with open(os.path.join(output_dir, "raw_responses.json"), "w") as f:
        json.dump(responses, f, indent=2)
    
    # Parse predictions with improved answer extraction
    predictions = [extract_answer(response) for response in responses]
    
    # Analyze response distribution
    analyze_response_distribution(responses, predictions)
    
    # Get ground truth labels
    labels = [example["label"] for example in dataset]
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(
        labels, 
        predictions,
        labels=["True", "False", "Uncertain"],
        output_dict=True
    )
    cm = confusion_matrix(labels, predictions, labels=["True", "False", "Uncertain"])
    
    # Save results
    results = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm, 
        classes=["True", "False", "Uncertain"],
        output_path=os.path.join(output_dir, "confusion_matrix.png")
    )
    
    # Print summary
    print("\nEvaluation Complete!")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nPerformance by class:")
    for label in ["True", "False", "Uncertain"]:
        if label in report:
            precision = report[label]['precision']
            recall = report[label]['recall']
            f1 = report[label]['f1-score']
            support = report[label]['support']
            print(f"  {label}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Support={support}")
    
    # Save detailed predictions for analysis
    prediction_details = []
    for i, (example, pred, label, response) in enumerate(zip(dataset, predictions, labels, responses)):
        prediction_details.append({
            "example_id": i,
            "premises": example["premises"],
            "conclusion": example["conclusion"],
            "true_label": label,
            "predicted_label": pred,
            "is_correct": pred == label,
            "model_response": response
        })
    
    with open(os.path.join(output_dir, "prediction_details.json"), "w") as f:
        json.dump(prediction_details, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    run_evaluation()