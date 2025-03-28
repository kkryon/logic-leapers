# -*- coding: utf-8 -*-
import os
import torch
import logging
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from datasets import load_dataset, Dataset
from transformers import pipeline, AutoTokenizer
from tqdm.auto import tqdm

# --- Flash Attention Check ---
try:
    import flash_attn
    _flash_attn_available = True
    logging.info("Flash Attention 2 available. Will use if applicable.")
except ImportError:
    _flash_attn_available = False
    logging.warning("Flash Attention 2 not found. Install with `pip install flash-attn --no-build-isolation` for potential speedup.")

# --- Constants ---
SYSTEM_PROMPT = """
Respond in the following format, you have to adhere to the format, only output the final answer without **ANY** additional information in the "answer" box.

<think>
...
</think>
<answer>
...
</answer>
"""
DEFAULT_CACHE_DIR = "cai6307-henrykobs/cache"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
DEFAULT_OUTPUT_DIR = "cai6307-henrykobs/model/Qwen-7B-GRPO"
DEFAULT_EVAL_FRACTION = 0.2
DEFAULT_BATCH_SIZE = 32 # Reduced default for demonstration if needed, adjust per GPU
SEED = 42
TORCH_DTYPE = torch.bfloat16
ATTN_IMPLEMENTATION = "flash_attention_2" if _flash_attn_available else "eager"
DEFAULT_DEBUG_SAMPLE_OUTPUT = True # Enable debug logging by default
MAX_DEBUG_SAMPLES = 5 # Max *initial* raw samples to log
GENERATION_MAX_NEW_TOKENS = 1024
MAX_MISMATCH_LOG = 5

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Extraction Functions ---
def extract_xml_answer(text: str):
    """Extracts content within the first <answer>...</answer> tag, with fallbacks."""
    try:
        start_tag = "<answer>"
        end_tag = "</answer>"
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag, start_idx + len(start_tag)) # Search after start tag

        if start_idx != -1 and end_idx != -1:
             return text[start_idx + len(start_tag):end_idx].strip().lower()

        # Fallback 1: Robust XML parsing
        try:
            xml_content = f"<root>{text}</root>" # Dummy root
            root = ET.fromstring(xml_content)
            answer_element = root.find('.//answer')
            if answer_element is not None and answer_element.text is not None:
                return answer_element.text.strip().lower()
        except ET.ParseError:
             pass # Malformed XML

        # Fallback 2: Simple split
        if start_tag in text and end_tag in text:
             try:
                 potential_answer = text.split(start_tag, 1)[1].split(end_tag, 1)[0]
                 return potential_answer.strip().lower()
             except IndexError:
                 pass # Split failed

        return None

    except Exception as e:
         logger.error(f"Unexpected error during XML extraction: {e} for text: {text[:150]}...", exc_info=True)
         return None

def extract_hash_answer(text: str):
    """Extracts the answer from GSM8K format '... #### <answer>'."""
    if "####" not in text:
        return None
    try:
        answer = text.split("####")[1].strip().replace(",", "").replace("$", "").lower()
        return answer
    except IndexError:
         logger.warning(f"Could not split gold answer on '####': {text}")
         return None

# --- Dataset Preparation ---
def get_gsm8k_dataset(cache_dir: str, split="test"):
    """Loads GSM8K, extracts gold answers, and filters invalid samples."""
    logger.info(f"Loading GSM8K dataset ({split} split)...")
    try:
        dataset = load_dataset("openai/gsm8k", "main", cache_dir=cache_dir, split=split, trust_remote_code=True)
        dataset = dataset.map(
            lambda x: {"gold_answer": extract_hash_answer(x["answer"]), "question": x["question"]},
            remove_columns=["answer"],
            desc="Extracting gold answers"
        )
        initial_count = len(dataset)
        dataset = dataset.filter(lambda x: x["gold_answer"] is not None)
        filtered_count = len(dataset)
        if initial_count != filtered_count:
             logger.warning(f"Filtered {initial_count - filtered_count} samples with invalid gold answers.")
        logger.info(f"Dataset loaded with {filtered_count} samples.")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load or process dataset: {e}", exc_info=True)
        raise

def prepare_prompts(dataset: Dataset, tokenizer):
    """Applies the chat template to each question."""
    logger.info(f"Applying chat template using {tokenizer.name_or_path}...")
    prompts = []
    skipped_count = 0
    for sample in tqdm(dataset, desc="Applying template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["question"]},
        ]
        try:
            # Important: Ensure the tokenizer is compatible with apply_chat_template
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)
        except Exception as e:
            logger.warning(f"Failed template application for question (truncated): {sample['question'][:50]}... Error: {e}. Skipping.")
            prompts.append(None) # Placeholder for filtering
            skipped_count += 1

    # Add prompts and filter failures
    dataset = dataset.add_column("prompt_text", prompts)
    original_len = len(prompts) # Count before filtering Nones
    dataset = dataset.filter(lambda x: x["prompt_text"] is not None)
    if skipped_count > 0 or len(dataset) != original_len: # Check both ways
        actual_filtered = original_len - len(dataset)
        logger.warning(f"Filtered out {actual_filtered} samples due to chat template errors.")

    if not dataset:
         logger.error("Dataset is empty after applying chat template. Cannot proceed.")
         # Consider raising an error or handling this state appropriately upstream
         return dataset # Return empty dataset

    logger.info(f"Chat template applied. {len(dataset)} prompts ready.")
    return dataset


# --- Evaluation Function ---
@torch.no_grad()
def evaluate_model(model_id: str, dataset: Dataset, batch_size: int, tokenizer, debug_sample_output: bool):
    """Evaluates a model using a pipeline, calculates accuracy, logs samples per batch if debug enabled."""
    logger.info(f"Starting evaluation for: {model_id}")
    logger.info(f"Using dtype={TORCH_DTYPE}, attn='{ATTN_IMPLEMENTATION}', max_new_tokens={GENERATION_MAX_NEW_TOKENS}")
    start_load_time = time.time()
    pipe = None

    # Check if dataset is empty before proceeding
    if not dataset or len(dataset) == 0:
        logger.error(f"Dataset provided to evaluate_model for {model_id} is empty. Skipping evaluation.")
        return 0.0

    try:
        # Load pipeline
        pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer, # Use pre-loaded tokenizer
            device_map="auto",
            torch_dtype=TORCH_DTYPE,
            trust_remote_code=True,
            model_kwargs={"attn_implementation": ATTN_IMPLEMENTATION} if ATTN_IMPLEMENTATION == "flash_attention_2" else {} # Only pass if using flash
        )

        # Ensure padding token setup for batching
        if pipe.tokenizer.pad_token is None or pipe.tokenizer.pad_token_id is None:
            logger.warning(f"Tokenizer for {model_id} lacks pad token/ID. Setting pad_token = eos_token.")
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
            pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id # Make sure ID is set
        if pipe.model.config.pad_token_id is None:
             logger.warning(f"Model config for {model_id} lacks pad_token_id. Setting from tokenizer: {pipe.tokenizer.pad_token_id}")
             pipe.model.config.pad_token_id = pipe.tokenizer.pad_token_id

        # Use left-padding for decoder-only models during generation
        pipe.tokenizer.padding_side = "left"
        logger.info(f"Tokenizer padding side: '{pipe.tokenizer.padding_side}', Pad token: '{pipe.tokenizer.pad_token}', ID: {pipe.tokenizer.pad_token_id}")

    except Exception as e:
        logger.error(f"FATAL: Failed to load pipeline for {model_id}: {e}", exc_info=True)
        if pipe is not None: del pipe
        torch.cuda.empty_cache()
        return 0.0 # Indicate failure

    load_time = time.time() - start_load_time
    logger.info(f"Pipeline loaded in {load_time:.2f}s.")

    logger.info(f"Generating predictions for {len(dataset)} samples (batch size: {batch_size})...")
    predictions = []
    generation_start_time = time.time()
    debug_samples_logged = 0 # Counter for the *initial* debug samples

    # Process dataset in batches using the pipeline
    # The pipeline handles batching internally based on `batch_size`
    # The loop iterates through results sample by sample, `i` is the dataset index
    for i, output in enumerate(tqdm(pipe(dataset["prompt_text"],
                                           max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                                           do_sample=False, # Greedy decoding for consistency
                                           batch_size=batch_size,
                                           return_full_text=False, # Only generated part
                                           pad_token_id=pipe.tokenizer.pad_token_id), # Crucial for batching
                                      total=len(dataset), desc=f"Generating {os.path.basename(model_id)}")):
        pred = None # Default prediction if extraction fails
        generated_text = "Error: Could not get generated text." # Default message

        try:
            # Standard output structure: [{'generated_text': '...'}]
            generated_text = output[0]['generated_text']
            pred = extract_xml_answer(generated_text) # Extract final answer

            # --- Combined Debug Logging ---
            if debug_sample_output:
                # 1. Log first N raw outputs (original behavior)
                if debug_samples_logged < MAX_DEBUG_SAMPLES:
                    generated_length = len(tokenizer.encode(generated_text))
                    logger.info(f"\n[DEBUG RAW OUTPUT {debug_samples_logged+1}/{MAX_DEBUG_SAMPLES} - {model_id}]"
                                f"\nRaw Output (Tokens: ~{generated_length}/{GENERATION_MAX_NEW_TOKENS}):\n---\n{generated_text}\n---"
                                f"\nExtracted: {pred}")
                    if generated_length >= GENERATION_MAX_NEW_TOKENS - 5: # Small buffer
                         logger.warning(f"[DEBUG] Raw output length close to max_new_tokens. May be truncated.")
                    debug_samples_logged += 1

                # 2. Log Question/Full Output/Gold for first sample of each batch
                # Check if 'i' is the start of a batch (index 0, batch_size, 2*batch_size, etc.)
                if i % batch_size == 0:
                    try:
                        question = dataset[i]["question"]
                        gold_answer = dataset[i]["gold_answer"]
                        batch_num = i // batch_size + 1
                        total_batches = (len(dataset) + batch_size - 1) // batch_size
                        logger.info(f"\n--- Sample from Batch {batch_num}/{total_batches} (Dataset Index {i}) ---"
                                    f"\nModel: {model_id}"
                                    f"\nQuestion: {question}"
                                    f"\nFull Model Output:\n{generated_text}"
                                    f"\nGold Answer: {gold_answer}"
                                    f"\nExtracted Answer: {pred}\n"
                                    f"------------------------------------")
                    except IndexError:
                         logger.warning(f"Could not retrieve data for batch sample log at index {i}.")
                    except Exception as log_e:
                         logger.warning(f"Error during batch sample logging at index {i}: {log_e}")
            # --- End Combined Debug Logging ---

        except (IndexError, KeyError, TypeError) as e:
            # Handle pipeline output parsing issues
            logger.warning(f"Pipeline output parsing error at index {i}: {e}. Raw output object: {output}. Prediction set to None.")
            # Log the raw text if possible, even on error
            raw_text_on_error = "N/A"
            try: raw_text_on_error = output[0]['generated_text']
            except Exception: pass
            if debug_sample_output: logger.warning(f"[DEBUG RAW ON PARSE ERROR]\n---\n{raw_text_on_error}\n---")

        except Exception as e:
            # Catch any other unexpected errors during processing a single sample
            logger.error(f"Unexpected error processing sample {i} for {model_id}: {e}", exc_info=True)
            # generated_text is already set to an error message
        finally:
            # Always append a prediction (None if extraction failed or error occurred)
            predictions.append(pred)

    generation_time = time.time() - generation_start_time
    samples_per_sec = len(dataset) / generation_time if generation_time > 0 else 0
    logger.info(f"Generation done in {generation_time:.2f}s ({samples_per_sec:.2f} samples/sec).")

    # Calculate Accuracy
    correct = 0
    total = 0
    mismatched_samples = []
    logger.info("Calculating accuracy...")
    for i, (pred, gold) in enumerate(zip(predictions, dataset["gold_answer"])):
        if gold is None: continue # Should be filtered, but check
        total += 1
        if pred is not None and pred == gold:
            correct += 1
        elif len(mismatched_samples) < MAX_MISMATCH_LOG:
             # Log mismatch details (question truncated)
             mismatched_samples.append({
                 "index": i, "question": dataset[i]["question"][:100]+"...",
                 "prediction": pred, "gold": gold
             })

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    # Ensure total isn't zero before logging division
    correct_str = f"{correct}/{total}" if total > 0 else "0/0"
    logger.info(f"Model {model_id} -- Accuracy: {accuracy:.2f}% ({correct_str})")


    if mismatched_samples:
        logger.warning(f"--- First {len(mismatched_samples)} Mismatched/Failed Samples ({model_id}) ---")
        for sample in mismatched_samples:
            logger.warning(f"Idx: {sample['index']}, Q: '{sample['question']}', Pred: '{sample['prediction']}', Gold: '{sample['gold']}'")
        logger.warning("--- End Mismatched Samples ---")

    # Cleanup
    logger.info(f"Cleaning up resources for {model_id}...")
    del pipe
    # Explicitly clear cache if needed, might help prevent OOM in loops
    torch.cuda.empty_cache()
    logger.info(f"Finished cleanup for {model_id}.")

    return accuracy

# --- Tokenizer Cache Helper ---
tokenizer_cache = {}
def get_tokenizer(model_id_or_path: str, cache_dir: str):
    """Loads and caches tokenizer, ensuring pad token exists."""
    # Normalize path for consistent caching key
    normalized_path = os.path.abspath(model_id_or_path) if os.path.exists(model_id_or_path) else model_id_or_path

    if normalized_path in tokenizer_cache:
        logger.info(f"Using cached tokenizer for {model_id_or_path}")
        return tokenizer_cache[normalized_path]
    else:
        logger.info(f"Loading tokenizer for {model_id_or_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id_or_path,
                trust_remote_code=True,
                cache_dir=cache_dir,
                padding_side='left' # Set padding side during load
            )
            # Critical: Ensure pad token and ID are set for batching
            if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
                if tokenizer.eos_token is not None and tokenizer.eos_token_id is not None:
                    logger.warning(f"Tokenizer for {model_id_or_path} lacks pad token/ID. Setting to EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id}).")
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                else:
                    # This is a problem - no EOS token either! Add a default?
                    logger.error(f"CRITICAL: Tokenizer for {model_id_or_path} lacks BOTH pad and EOS tokens. Cannot set default padding. Batching will likely fail.")
                    # You might need to manually add a pad token here if this occurs:
                    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    # model.resize_token_embeddings(len(tokenizer)) # If model is loaded separately
                    # For pipeline, this is harder. Best to fix the tokenizer source.
                    return None # Indicate failure

            logger.info(f"Tokenizer loaded. Pad token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}, Padding side: {tokenizer.padding_side}")
            tokenizer_cache[normalized_path] = tokenizer # Cache it using normalized path
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer {model_id_or_path}: {e}", exc_info=True)
            return None

# --- Main Execution Logic ---
def main(args: SimpleNamespace):
    """Orchestrates dataset loading, model evaluation, and result reporting."""
    overall_start_time = time.time()
    results = {}

    # Load and prepare the dataset once
    try:
        dataset_full = get_gsm8k_dataset(cache_dir=args.cache_dir, split="test")
    except Exception:
        logger.error("Failed to load the dataset. Exiting.")
        return

    # Subset the dataset if requested
    total_samples = len(dataset_full)
    eval_size = total_samples
    if 0.0 < args.eval_fraction < 1.0:
        eval_size = max(1, int(total_samples * args.eval_fraction)) # Ensure at least 1
        logger.info(f"Selecting {eval_size} samples ({args.eval_fraction*100:.1f}%) from {total_samples} using seed={SEED}...")
        dataset = dataset_full.shuffle(seed=SEED).select(range(eval_size))
    elif args.eval_fraction == 1.0:
         logger.info(f"Evaluating on the full test set ({total_samples} samples).")
         dataset = dataset_full
    else:
        # This validation is also done at startup, but good to have defense in depth
        logger.error(f"Invalid eval_fraction: {args.eval_fraction}. Must be > 0.0 and <= 1.0. Exiting.")
        return

    if len(dataset) == 0:
        logger.error("Dataset is empty after potential filtering/subsetting. Exiting.")
        return

    # --- Evaluate Base Model ---
    logger.info(f"\n--- Evaluating Base Model: {args.base_model} ---")
    base_tokenizer = get_tokenizer(args.base_model, args.cache_dir)
    if base_tokenizer:
        # Prepare prompts using the *original (potentially subsetted) dataset* and base tokenizer
        base_dataset_prepared = prepare_prompts(dataset, base_tokenizer)
        if len(base_dataset_prepared) > 0:
             base_acc = evaluate_model(
                 args.base_model, base_dataset_prepared, args.batch_size,
                 base_tokenizer, args.debug_sample_output # Pass debug flag
             )
             results["base_model"] = base_acc
        else:
             logger.error("Base dataset preparation resulted in 0 samples. Skipping base model eval.")
             results["base_model"] = "Eval Skipped (0 samples)"
    else:
         logger.error("Skipping base model evaluation due to tokenizer load failure.")
         results["base_model"] = "Eval Failed (Tokenizer Load)"
    # ---------------------------

    # --- Evaluate Checkpoints ---
    logger.info(f"\n--- Evaluating Checkpoints in: {args.output_dir} ---")
    if not os.path.isdir(args.output_dir):
        logger.warning(f"Checkpoint directory not found: {args.output_dir}. Skipping checkpoint evaluation.")
    else:
        checkpoints = []
        try:
            ckpt_dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))]
            # Extract step number, handling potential errors, and sort
            valid_checkpoints = []
            for d in ckpt_dirs:
                try:
                    step = int(d.split("-")[-1])
                    valid_checkpoints.append((step, d))
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse step number from directory name: {d}")
            checkpoints = sorted(valid_checkpoints, key=lambda x: x[0]) # Sort by step number
            logger.info(f"Found {len(checkpoints)} valid checkpoint directories.")
        except OSError as e:
            logger.error(f"Cannot access checkpoint directory {args.output_dir}: {e}", exc_info=True)
            checkpoints = [] # Ensure checkpoints is empty list on error

        if not checkpoints:
             logger.warning(f"No valid 'checkpoint-<number>' directories found in {args.output_dir}.")

        # Simplified Checkpoint Loop
        for step, ckpt_name in checkpoints:
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
            logger.info(f"\n--- Evaluating Checkpoint: {ckpt_name} (Step: {step}) ---")

            # Determine tokenizer path: checkpoint's own or fallback to base
            # Check for tokenizer files within the specific checkpoint directory
            tokenizer_config_path = os.path.join(ckpt_path, "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                tokenizer_path = ckpt_path # Use checkpoint's tokenizer
                logger.info(f"Found tokenizer config in {ckpt_path}. Using checkpoint-specific tokenizer.")
            else:
                tokenizer_path = args.base_model # Fallback to base model tokenizer
                logger.info(f"No tokenizer config in {ckpt_path}. Using base model tokenizer ({args.base_model}).")

            # Load tokenizer (uses cache if path hasn't changed effectively)
            ckpt_tokenizer = get_tokenizer(tokenizer_path, args.cache_dir)
            if not ckpt_tokenizer:
                logger.error(f"Failed to load tokenizer from {tokenizer_path}. Skipping checkpoint {ckpt_name}.")
                results[ckpt_name] = "Eval Failed (Tokenizer Load)"
                continue # Skip this checkpoint

            # Prepare prompts using the *original (potentially subsetted) dataset* and the determined tokenizer
            # This ensures prompts are always correctly formatted for the specific model/tokenizer being evaluated
            logger.info(f"Preparing prompts for {ckpt_name} using tokenizer: {tokenizer_path}")
            ckpt_dataset_prepared = prepare_prompts(dataset, ckpt_tokenizer)

            if len(ckpt_dataset_prepared) == 0:
                logger.error(f"Dataset preparation for {ckpt_name} resulted in 0 samples (using tokenizer {tokenizer_path}). Skipping eval.")
                results[ckpt_name] = "Eval Skipped (0 samples)"
                continue # Skip this checkpoint

            # Evaluate the checkpoint using its specific path and the correctly prepared dataset/tokenizer
            acc = evaluate_model(
                ckpt_path,              # Model path is the checkpoint directory
                ckpt_dataset_prepared,  # Dataset prepared with the right tokenizer
                args.batch_size,
                ckpt_tokenizer,         # The tokenizer object itself
                args.debug_sample_output # Pass debug flag
            )
            results[ckpt_name] = acc
    # --------------------------

    # --- Print Final Summary ---
    print("\n" + "="*60)
    print(" " * 15 + "Overall Evaluation Results Summary")
    print("="*60)
    total_eval_time = time.time() - overall_start_time

    # Print Base Model Result First
    if "base_model" in results:
         base_res = results.pop("base_model") # Remove from dict to handle checkpoints separately
         model_name_disp = f"BASE: {args.base_model}"
         res_str = f"{base_res:.2f}% accuracy" if isinstance(base_res, float) else base_res
         print(f"{model_name_disp:<45}: {res_str}")

    # Print Checkpoint Results, sorted by step number (derived from the key)
    sorted_ckpt_results = sorted(
        [(k, v) for k, v in results.items() if k.startswith("checkpoint-")],
        key=lambda item: int(item[0].split('-')[-1]) # Sort by step number in the key string
    )
    for model_id, acc in sorted_ckpt_results:
        res_str = f"{acc:.2f}% accuracy" if isinstance(acc, float) else acc
        print(f"{model_id:<45}: {res_str}") # model_id is like "checkpoint-1000"

    print("-" * 60)
    logger.info(f"Total evaluation runtime: {total_eval_time:.2f} seconds ({total_eval_time/60:.2f} minutes).")
    print("="*60)

# --- Script Entry Point ---
if __name__ == "__main__":
    # Configuration Setup using SimpleNamespace and Environment Variables
    args = SimpleNamespace()
    args.base_model = os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL)
    args.output_dir = os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    args.cache_dir = os.environ.get("CACHE_DIR", DEFAULT_CACHE_DIR)
    try:
        args.eval_fraction = float(os.environ.get("EVAL_FRACTION", DEFAULT_EVAL_FRACTION))
    except ValueError:
        logger.warning(f"Invalid EVAL_FRACTION env var. Using default: {DEFAULT_EVAL_FRACTION}")
        args.eval_fraction = DEFAULT_EVAL_FRACTION
    try:
        args.batch_size = int(os.environ.get("BATCH_SIZE", DEFAULT_BATCH_SIZE))
    except ValueError:
        logger.warning(f"Invalid BATCH_SIZE env var. Using default: {DEFAULT_BATCH_SIZE}")
        args.batch_size = DEFAULT_BATCH_SIZE
    # Read debug flag from environment, converting common "true" values
    debug_env = os.environ.get("DEBUG_SAMPLE_OUTPUT", str(DEFAULT_DEBUG_SAMPLE_OUTPUT)).lower()
    args.debug_sample_output = debug_env in ['true', '1', 'yes', 'on']

    # Validation
    if not 0.0 < args.eval_fraction <= 1.0:
         logger.error(f"Configuration Error: eval_fraction ({args.eval_fraction}) must be > 0.0 and <= 1.0. Adjust script or EVAL_FRACTION env var.")
         exit(1) # Exit on invalid config
    if args.batch_size <= 0:
        logger.error(f"Configuration Error: batch_size ({args.batch_size}) must be positive. Adjust script or BATCH_SIZE env var.")
        exit(1) # Exit on invalid config

    # Log Final Configuration
    logger.info("--- Final Configuration ---")
    logger.info(f"Base Model         : {args.base_model}")
    logger.info(f"Checkpoints Dir    : {args.output_dir}")
    logger.info(f"Cache Dir          : {args.cache_dir}")
    logger.info(f"Eval Fraction      : {args.eval_fraction}")
    logger.info(f"Batch Size         : {args.batch_size}")
    logger.info(f"Debug Sample Output: {args.debug_sample_output} (Logs 1st sample/batch if True)")
    logger.info(f"Torch Dtype        : {TORCH_DTYPE}")
    logger.info(f"Attention Impl.    : {ATTN_IMPLEMENTATION}")
    logger.info(f"Max New Tokens     : {GENERATION_MAX_NEW_TOKENS}")
    logger.info("-------------------------")

    main(args)
