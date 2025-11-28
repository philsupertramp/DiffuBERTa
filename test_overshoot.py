import torch
import json
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel
from diffuberta.helpers import extract_diffusion

# Setup
model_name = "./json_diff_model"
model_id = "answerdotai/ModernBERT-base"

base_model = AutoModelForMaskedLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(base_model, model_name)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

# Test with OVERSHOOT (more slots than needed)
text = "2023-10-27 14:02:11 [ERROR] ConnectionRefused from 192.168.1.55 : Port 8080 blocked."
instr = "Extract timestamp, error type, and IP address"

print("=" * 80)
print("Testing OVERSHOOT - Allocating MORE slots than needed")
print("=" * 80)

# Allocate 10 slots for each field (way more than needed)
temp_overshoot = {"time": "[10]", "error": "[10]", "ip_address": "[10]"}

result = extract_diffusion(tokenizer, model, text, instr, json.dumps(temp_overshoot), steps=10, verbose=True)
print(f"\nRaw model output: {repr(result)}")

try:
    parsed = json.loads(result)
    print(f"\nParsed JSON:")
    for key, val in parsed.items():
        print(f"  {key}: {repr(val)}")
        print(f"    Contains PAD: {tokenizer.pad_token in val}")
except Exception as e:
    print(f"Parse error: {e}")

print("\n" + "=" * 80)
print("Testing a simple short value with overshoot")
print("=" * 80)

text2 = "The city is Paris"
instr2 = "Extract city"
temp2 = {"city": "[10]"}  # "Paris" is 1 token, give it 10 slots

result2 = extract_diffusion(tokenizer, model, text2, instr2, json.dumps(temp2), steps=10, verbose=True)
print(f"\nRaw model output: {repr(result2)}")

try:
    parsed2 = json.loads(result2)
    print(f"\nParsed JSON:")
    for key, val in parsed2.items():
        print(f"  {key}: {repr(val)}")
        print(f"    Contains PAD: {tokenizer.pad_token in val}")
        # Count tokens
        tokens = tokenizer.tokenize(val)
        print(f"    Token count: {len(tokens)}")
        print(f"    Tokens: {tokens}")
except Exception as e:
    print(f"Parse error: {e}")
