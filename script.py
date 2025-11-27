import torch
import torch.nn.functional as F
import re
import json
import math
from transformers import RobertaTokenizer, RobertaForMaskedLM

# Setup
model_name = "./json_diff_model" #"roberta-base"
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained(model_name)
model.eval()

def expand_template(json_template):
    # Same helper as before
    return re.sub(r'\[(\d+)\]', lambda m: " ".join(["<mask>"] * int(m.group(1))), json_template)


def sanitize_json_value(raw_value):
    """
    The 'Smart Padding' Logic.
    If the model generates: "Paris " , } ..."
    We cut it at "Paris".
    """
    if isinstance(raw_value, list):
        out = []
        for elem in raw_value:
            out.append(sanitize_json_value(elem))
        return out

    # 1. If the model predicted a closing quote inside the value, cut there.
    if '"' in raw_value:
        raw_value = raw_value.split('"')[0]
    
    # 2. If the model predicted a comma or brace (start of next field), cut there.
    # This happens if the model "finished" the word early and started writing the next key.
    for stop_char in [',', '}', '{']:
        if stop_char in raw_value:
            raw_value = raw_value.split(stop_char)[0]
            
    return raw_value.strip()


def extract_parallel(source_text, instruction, json_template_shorthand, steps=1):
    # 1. Prepare Input
    json_template = expand_template(json_template_shorthand)
    full_prompt = f"{source_text} {tokenizer.sep_token} {instruction}: {json_template}"
    
    
    inputs = tokenizer(full_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f'Prompt of size {len(input_ids[0])}')
    mask_token_id = tokenizer.mask_token_id
    
    # 2. Identify initial masks
    # We find all indices that need filling
    initial_mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]
    total_masks = len(initial_mask_indices)
    
    if total_masks == 0:
        return "No masks found."

    print(f"--- Parallel Extraction ({steps} steps for {total_masks} masks) ---")
    
    # 3. The Parallel Loop
    # We determine how many tokens to fill per step
    chunk_size = math.ceil(total_masks / steps)
    
    # We keep a set of indices we have already filled to avoid re-filling them
    filled_indices = set()

    for step in range(steps):
        # Current active masks (excluding ones we filled in previous loops)
        current_mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]
        
        if len(current_mask_indices) == 0:
            break

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Get probabilities for currently masked positions
        mask_logits = logits[0, current_mask_indices, :]
        mask_probs = F.softmax(mask_logits, dim=-1)
        
        # Get top token and confidence for each mask
        top_confidences, top_ids = torch.max(mask_probs, dim=-1)
        
        # --- The Parallel Selection Strategy ---
        
        # We want to pick the 'chunk_size' best tokens from the CURRENT masks
        # Sort by confidence (descending)
        sorted_indices = torch.argsort(top_confidences, descending=True)
        
        # Determine how many to lock in this round
        # In the last step, we must lock everything remaining.
        if step == steps - 1:
            n_to_lock = len(current_mask_indices)
        else:
            n_to_lock = min(chunk_size, len(current_mask_indices))
            
        # Select the indices of the masks we want to fill
        indices_to_fill_local = sorted_indices[:n_to_lock]
        
        # Convert local indices back to global input_ids indices
        indices_to_fill_global = current_mask_indices[indices_to_fill_local]
        tokens_to_fill = top_ids[indices_to_fill_local]
        confs_to_fill = top_confidences[indices_to_fill_local]
        
        # Apply updates
        input_ids[0, indices_to_fill_global] = tokens_to_fill
        
        # Visualization
        filled_words = [tokenizer.decode([t]).strip() for t in tokens_to_fill]
        avg_conf = confs_to_fill.mean().item()
        print(f"Step {step+1}: Locked {len(filled_words)} tokens (Avg Conf: {avg_conf:.2f}) -> {filled_words}")

    # 4. Extract
    full_output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    try:
        result = full_output.split(f"{instruction}:")[1].strip()
    except:
        result = full_output
        
    return result


def run_stress_test(name, text, instruction, template):
    print(f"\n🔰 SCENARIO: {name}")
    print(f"📄 Input: {text[:160]}..." if len(text) > 60 else f"📄 Input: {text}")
    
    raw_output = extract_parallel(text, instruction, template, steps=10)
    
    print("\n🔍 Parsed & Sanitized:")
    # Simple regex parser for the demo
    for key, val in json.loads(raw_output).items():
        print(f"   • {key}: {sanitize_json_value(val)}")
    print("-" * 40)

# ==========================================
# 1. THE "IMPLICIT CLASSIFIER"
# Challenge: The words "Positive" or "Negative" are NOT in the text.
# The model must infer the sentiment and fill the mask with a label.
# ==========================================
text_1 = "I waited 40 minutes for my food and it was cold. I am never coming back."
instr_1 = "Analyze the review. Extract the food status."
temp_1 = '{"food_status": "[1]", "sentiment": "[1]"}'

run_stress_test("Sentiment Inference", text_1, instr_1, temp_1)

text_1 = "I waited 40 minutes for my food and it was cold. I am never coming back."
instr_1 = "Analyze the review. Classify sentiment as Positive or Negative."
temp_1 = '{"food_status": "[1]", "sentiment": "[1]"}'

run_stress_test("Sentiment Inference", text_1, instr_1, temp_1)


# ==========================================
# 2. THE "MESSY LOG PARSER"
# Challenge: Extracting specific technical formats (IPs, Error Codes)
# from a noisy server log.
# ==========================================
text_2 = "2023-10-27 14:02:11 [ERROR] ConnectionRefused from 192.168.1.55 : Port 8080 blocked."
instr_2 = "Extract timestamp, error type, and IP address"
# We use [3] for timestamp to capture date+time, [5] for IP to capture the segments
temp_2 = '{"time": "[4]", "error": "[3]", "ip_address": "[5]"}'

#run_stress_test("Log Parsing", text_2, instr_2, temp_2)


# ==========================================
# 3. THE "MEETING SCHEDULER" (Named Entity Recognition)
# Challenge: Handling multiple entities (names) and parsing a date.
# ==========================================
text_3 = "Hey, let's set up a sync with Sarah and Mike next Tuesday regarding the Q4 roadmap."
instr_3 = "Extract participants, day the meeting is scheduled and topic"
# Note: "participants" needs multiple slots to catch both names if possible
temp_3 = '{"participants": ["[2]", "[2]"], "topic": "[3]", "schedule": "[2]"}'

#run_stress_test("Meeting Extraction", text_3, instr_3, temp_3)


# 2. Test Input
test_text = "We are excited to welcome Dr. Sarah Connor to our Paris office."
test_instr = "Extract details"
# We can use the same generic template with overshot masks
test_template = '{"name": "[2]", "job": "[1]", "city": "[1]"}'

# 3. Run Inference (Reuse your extract_parallel function)
# Note: Ensure extract_parallel uses the global 'model' variable which is now fine-tuned
result = extract_parallel(test_text, test_instr, test_template, steps=5)

print(f"\nfine-tuned Result: {result}")

test_text = "We are excited to welcome Dr. Sarah Connor as Senior Data Scientist to our Paris office."
test_instr = "Extract details"
# We can use the same generic template with overshot masks
test_template = '{"name": "[3]", "job": "[3]", "city": "[1]"}'

# 3. Run Inference (Reuse your extract_parallel function)
# Note: Ensure extract_parallel uses the global 'model' variable which is now fine-tuned
result = extract_parallel(test_text, test_instr, test_template, steps=15)

print(f"\nfine-tuned Result: {result}")
