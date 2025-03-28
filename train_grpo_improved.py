# train_grpo_improved.py

import os
import re
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# Define cache directory
CACHE_DIR = "cai6307-henrykobs/cache"

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format, you have to adhere to the format, only output the final answer without **ANY** additional information in the "answer" box.

<think>
...
</think>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<think>
{think}
</think>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

def get_gsm8k_questions(split="train"):
    data = load_dataset('openai/gsm8k', 'main', cache_dir=CACHE_DIR)[split]  # specify cache directory
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs):
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# Choose model and set up output parameters
model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
# model_name = "Qwen/Qwen2.5-14B-Instruct-1M"
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"

if "Llama" in model_name:
    output_dir = "cai6307-henrykobs/model/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-gsm8k"
else:
    output_dir = "cai6307-henrykobs/model/Qwen-7B-GRPO-2nd"
    run_name = "Qwen-7B-GRPO-gsm8k-2nd"

# Training configuration optimized for an A100 80GB
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,  # Slightly increased learning rate (tune as needed)
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=8,      # Increased batch size for ample GPU memory
    gradient_accumulation_steps=4,
    num_generations=2,                   # Single generation per prompt
    max_prompt_length=256,
    max_completion_length=512,
    num_train_epochs=1,
    save_steps=20,                      # Save checkpoint every 100 steps
    max_grad_norm=0.1,
    report_to="none",                    # Change to "tensorboard" if you want native TensorBoard logging
    log_on_each_node=False,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

# Initialize the model loaded entirely on GPU (remove offloading)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,  # specify cache directory
    device_map="auto"
)

# Optionally, if you are using PyTorch 2.0+, you can compile the model for a speed boost:
# model = torch.compile(model)

# Gradient checkpointing is not required with 80GB, so it's been disabled for potentially faster execution
# model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token

# --- Custom Callback for Plotting Metrics Separately ---
class PlottingCallback(TrainerCallback):
    def __init__(self, model, tokenizer, sample_prompts, eval_steps=10, eval_max_length=50):
        """
        :param model: The current model.
        :param tokenizer: The tokenizer for processing prompts.
        :param sample_prompts: A list of sample prompts to evaluate response lengths.
                               Each prompt should be a list of messages (dict with "content").
        :param eval_steps: Frequency (in global steps) to evaluate and log response length.
        :param eval_max_length: Maximum additional tokens to generate for evaluation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sample_prompts = sample_prompts
        self.eval_steps = eval_steps
        self.eval_max_length = eval_max_length

        self.global_steps = []
        self.losses = []
        self.rewards = []
        self.avg_resp_lengths = []
        self.resp_plot_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Collect metrics if present.
        if "loss" in logs:
            self.global_steps.append(state.global_step)
            self.losses.append(logs["loss"])
        if "reward" in logs:
            self.rewards.append(logs["reward"])

        # Every eval_steps, compute average response length.
        if state.global_step % self.eval_steps == 0:
            avg_length = self.evaluate_response_length()
            self.avg_resp_lengths.append(avg_length)
            self.resp_plot_steps.append(state.global_step)
            self.plot_all_metrics()

    def evaluate_response_length(self):
        lengths = []
        self.model.eval()
        with torch.no_grad():
            for prompt in self.sample_prompts:
                # Use the last message's content as the prompt text.
                prompt_text = prompt[-1]["content"]
                encoded = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
                input_length = encoded.input_ids.shape[1]
                outputs = self.model.generate(
                    **encoded,
                    max_length=input_length + self.eval_max_length,
                    do_sample=True
                )
                # Compute number of generated tokens (excluding the prompt)
                gen_length = outputs.shape[1] - input_length
                lengths.append(gen_length)
        self.model.train()
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        print(f"Evaluated average response length: {avg_length:.2f} tokens")
        return avg_length

    def plot_all_metrics(self):
        # Plot Loss
        plt.figure()
        plt.plot(self.global_steps, self.losses, label="Loss")
        plt.xlabel("Global Steps")
        plt.ylabel("Loss")
        plt.title("Loss over Time")
        plt.legend()
        plt.savefig("run2/loss_metrics.png")
        plt.close()
        print(f"Saved loss plot at step {self.global_steps[-1]}")

        # Plot Reward (if available)
        if self.rewards:
            plt.figure()
            # Make sure we plot rewards vs. steps corresponding to rewards.
            plt.plot(self.global_steps[:len(self.rewards)], self.rewards, label="Reward")
            plt.xlabel("Global Steps")
            plt.ylabel("Reward")
            plt.title("Reward over Time")
            plt.legend()
            plt.savefig("run2/reward_metrics.png")
            plt.close()
            print(f"Saved reward plot at step {self.global_steps[-1]}")

        # Plot Average Response Length
        plt.figure()
        plt.plot(self.resp_plot_steps, self.avg_resp_lengths, label="Avg Response Length (tokens)")
        plt.xlabel("Global Steps")
        plt.ylabel("Tokens")
        plt.title("Average Response Length over Time")
        plt.legend()
        plt.savefig("run2/response_length.png")
        plt.close()
        print(f"Saved response length plot at step {self.global_steps[-1]}")

# Select a few sample prompts from the dataset for evaluation.
# Here we take the first 5 samples and use their "prompt" field.
sample_prompts = [item["prompt"] for item in dataset.select(range(5))]

# Initialize the GRPOTrainer with the custom callback.
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        correctness_reward_func,
        soft_format_reward_func,
        int_reward_func
    ],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.add_callback(PlottingCallback(model, tokenizer, sample_prompts, eval_steps=10, eval_max_length=50))

# --- Checkpoint Recovery ---
# If the training was interrupted, resume from the latest checkpoint if available.
resume_checkpoint = None
if os.path.exists(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        # Sort checkpoints by the numeric step value appended at the end of the folder name.
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        resume_checkpoint = os.path.join(output_dir, checkpoints[-1])
        print(f"Resuming training from checkpoint: {resume_checkpoint}")

# --- Start Training ---
if __name__ == "__main__":
    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    except Exception as e:
        # If something goes wrong, save the current state so we can resume later.
        print("An error occurred during training:", e)
        trainer.save_state()
        raise
