import torch
import torch.nn.functional as F
import re
import json
import math
from transformers import RobertaForMaskedLM, AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel
from diffuberta.helpers import extract_parallel, sanitize_json_value

# Setup
model_name = "./json_diff_model" #"roberta-base"

model_id = "answerdotai/ModernBERT-base"

# Load base model, then apply LoRA adapters
base_model = AutoModelForMaskedLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(base_model, model_name)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

def run_stress_test(name, text, instruction, template, steps=10):
    print(f"\n🔰 SCENARIO: {name}")
    print(f"📄 Input: {text[:160]}..." if len(text) > 60 else f"📄 Input: {text}")
    
    raw_output = extract_parallel(tokenizer, model, text, instruction, template, steps=steps)
    
    print(f"\n🔍 Parsed & Sanitized: {raw_output}")
    # Simple regex parser for the demo
    for key, val in raw_output.items():
        print(f"   • {key}: {val}")
    print("-" * 40)

# ==========================================
# 1. THE "IMPLICIT CLASSIFIER"
# Challenge: The words "Positive" or "Negative" are NOT in the text.
# The model must infer the sentiment and fill the mask with a label.
# ==========================================
text_1 = "I waited 40 minutes for my food and it was cold. I am never coming back."
instr_1 = "Analyze the review. Classify sentiment as Positive or Negative. And extract the food status."
temp_1 = {"food_status": "[1]", "sentiment": "[1]"}

run_stress_test("Sentiment Inference", text_1, instr_1, temp_1, -1)
run_stress_test("Sentiment Inference", text_1, instr_1, temp_1, 5)
run_stress_test("Sentiment Inference", text_1, instr_1, temp_1, 25)

text_1 = "I waited 40 minutes for my food and it was cold. I am never coming back."
instr_1 = "Analyze the review. Classify sentiment as Positive or Negative."
temp_1 = {"food_status": "[10]", "sentiment": "[10]"}

run_stress_test("Sentiment Inference", text_1, instr_1, temp_1, -1)


# ==========================================
# 2. THE "MESSY LOG PARSER"
# Challenge: Extracting specific technical formats (IPs, Error Codes)
# from a noisy server log.
# ==========================================
text_2 = "2023-10-27 14:02:11 [ERROR] ConnectionRefused from 192.168.1.55 : Port 8080 blocked."
instr_2 = "Extract timestamp, error type, and IP address"
# We use [3] for timestamp to capture date+time, [5] for IP to capture the segments
temp_2 = {"time": "[4]", "error": "[3]", "ip_address": "[5]"}

run_stress_test("Log Parsing", text_2, instr_2, temp_2)


# ==========================================
# 3. THE "MEETING SCHEDULER" (Named Entity Recognition)
# Challenge: Handling multiple entities (names) and parsing a date.
# ==========================================
text_3 = "Hey, let's set up a sync with Sarah and Mike next Tuesday regarding the Q4 roadmap."
instr_3 = "Extract participants, day the meeting is scheduled and topic"
# Note: "participants" needs multiple slots to catch both names if possible
temp_3 = {"participants": [{"name": "[2]"}], "topic": "[3]", "schedule": "[2]"}

run_stress_test("Meeting Extraction", text_3, instr_3, temp_3)


# 2. Test Input
test_text = "We are excited to welcome Dr. Sarah Connor to our Paris office."
test_instr = "Extract details"
# We can use the same generic template with overshot masks
test_template = {"name": "[10]", "job": "[1]", "city": "[1]"}

# 3. Run Inference (Reuse your extract_parallel function)
# Note: Ensure extract_parallel uses the global 'model' variable which is now fine-tuned
result = extract_parallel(tokenizer, model, test_text, test_instr, test_template, steps=5)

print(f"\nfine-tuned Result: {result}")

test_text = "We are excited to welcome Dr. Sarah Connor as Senior Data Scientist to our Paris office."
test_instr = "Extract details"
# We can use the same generic template with overshot masks
test_template = {"name": "[3]", "job": "[3]", "city": "[1]"}

# 3. Run Inference (Reuse your extract_parallel function)
# Note: Ensure extract_parallel uses the global 'model' variable which is now fine-tuned
result = extract_parallel(tokenizer, model, test_text, test_instr, test_template, steps=5)

print(f"\nfine-tuned Result: {result}")

text = "I booked a flight to Paris, Texas, not the one in France."
instruction = "Extract destination"
template = {"city": "[10]", "country": "[10]"}

# Run with more steps to allow for "Mind Changing"
result = extract_parallel(tokenizer, model, text, instruction, template, steps=3)
print(result)
