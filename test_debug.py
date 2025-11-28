import torch
import json
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel
from diffuberta.helpers import extract_diffusion, expand_template

# Setup
model_name = "./json_diff_model"
model_id = "answerdotai/ModernBERT-base"

base_model = AutoModelForMaskedLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(base_model, model_name)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

# Test log parsing with verbose output
text_2 = "2023-10-27 14:02:11 [ERROR] ConnectionRefused from 192.168.1.55 : Port 8080 blocked."
instr_2 = "Extract timestamp, error type, and IP address"
temp_2 = {"time": "[4]", "error": "[3]", "ip_address": "[5]"}

print("=" * 80)
print("Testing Log Parsing")
print("=" * 80)
result = extract_diffusion(tokenizer, model, text_2, instr_2, json.dumps(temp_2), steps=10, verbose=True)
print(f"\nRaw model output: {repr(result)}")
print(f"Length: {len(result)}")

# Try to parse it
try:
    parsed = json.loads(result)
    print(f"\nParsed JSON:")
    for key, val in parsed.items():
        print(f"  {key}: {repr(val)}")
except Exception as e:
    print(f"Parse error: {e}")
