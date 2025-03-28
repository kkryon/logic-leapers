import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def test_single_prompt(model_path, example):
    """
    Test a single FOLIO prompt and get the model's prediction
    
    Args:
        model_path (str): Path to the HuggingFace model
        example (dict): Single FOLIO example
        
    Returns:
        dict: Results including the model's raw response and parsed answer
    """
    # Define the system prompt - Make the format requirement more explicit
    SYSTEM_PROMPT = """You are a logical reasoning assistant. You must use the exact format below:
    <think>
    Think through the problem step by step, analyzing the premises and conclusion carefully.
    </think>
    <answer>
    Provide ONLY ONE of these three options: True, False, or Uncertain.
    </answer>
    """
    
    print(f"Loading model: {model_path}")
    
    # Load model in FP16 precision without quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use FP16 for efficiency
        device_map="auto"           # Let the library decide placement
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set model to evaluation mode
    model.eval()
    
    # Format the prompt
    premises = example["premises"]
    conclusion = example["conclusion"]
    
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
    
    # Apply chat template - add temperature parameter to encourage precise formatting
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print("\nGenerating prediction...")
    
    # Tokenize with explicit attention mask
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=2048
    ).to(model.device)
    
    # Generate with correct parameters and additional constraints
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # Deterministic generation
            temperature=1.0,  # Setting to 1.0 to avoid warning with do_sample=False
            top_p=None,       # Disable top_p to avoid warning
            top_k=None,       # Disable top_k to avoid warning
            repetition_penalty=1.2  # Add repetition penalty to improve output quality
        )
    
    # Decode and get only the new tokens
    completion = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Extract answer
    answer = extract_answer(completion)
    
    # Print results
    print("\nModel's raw response:")
    print(completion)
    print(f"\nParsed answer: {answer}")
    print(f"True label: {example['label']}")
    print(f"Match: {'Yes' if answer == example['label'] else 'No'}")
    
    return {
        "raw_response": completion,
        "parsed_answer": answer,
        "true_label": example["label"],
        "is_correct": answer == example["label"]
    }

def extract_answer(response):
    """Extract the answer from between <answer> tags"""
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
    
    # More thorough checking for specific phrases
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

# Example usage:
if __name__ == "__main__":
    # Define the model path
    model_path = "Qwen/Qwen2.5-1.5B-Instruct"  # Change to your model
    
    # Define the example
    example = {
        'story_id': 380, 
        'premises': 'People in this club who perform in school talent shows often attend and are very engaged with school events.\nPeople in this club either perform in school talent shows often or are inactive and disinterested community members.\nPeople in this club who chaperone high school dances are not students who attend the school.\nAll people in this club who are inactive and disinterested members of their community chaperone high school dances.\nAll young children and teenagers in this club who wish to further their academic careers and educational opportunities are students who attend the school. \nBonnie is in this club and she either both attends and is very engaged with school events and is a student who attends the school or is not someone who both attends and is very engaged with school events and is not a student who attends the school.',
        'premises-FOL': '∀x (InThisClub(x) ∧ PerformOftenIn(x, schoolTalentShow) → Attend(x, schoolEvent) ∧ VeryEngagedWith(x, schoolEvent))\n∀x (InThisClub(x) → PerformOftenIn(x, schoolTalentShow) ⊕ (InActive(x) ∧ Disinterested(x) ∧ MemberOf(x, community)))\n∀x (InThisClub(x) ∧ Chaperone(x, highSchoolDance) → ¬(Studen(x) ∧ AttendSchool(x)))\n∀x (InThisClub(x) ∧ (InActive(x) ∧ Disinterested(x) ∧ MemberOf(x, community)) → Chaperone(x, highSchoolDances))\n∀x (InThisClub(x) ∧ (YoungChildren(x) ⊕ Teenager(x)) ∧ WishToFurther(x, academicCareer)) → Studen(x) ∧ AttendSchool(x))\nInThisClub(bonnie) ∧ ¬((Attend(x, schoolEvent) ∧ VeryEngagedWith(bonnie, schoolEvent)) ⊕ (Studen(bonne) ∧ AttendSchool(bonnie)))',
        'conclusion': 'Bonnie performs in school talent shows often.',
        'conclusion-FOL': 'InThisClub(bonnie) ∧ (Perform(bonnie, schoolTalentShow))',
        'label': 'Uncertain',
        'example_id': 1014
    }
    
    # Try multiple models to see which performs better with the format
    model_options = [
        "Qwen/Qwen2.5-1.5B-Instruct",  # Your current model
        # Uncomment to test with other models that might follow instructions better
        # "meta-llama/Llama-2-7b-chat-hf",
        # "microsoft/Phi-2",
    ]
    
    model_path = model_options[0]  # Choose the first model as default
    
    # Test the model
    result = test_single_prompt(model_path, example)