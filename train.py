import random
import json
import torch
import torch.nn.functional as F
import math
import re
from transformers import Trainer, TrainingArguments, RobertaForMaskedLM, RobertaTokenizer, AutoModelForMaskedLM, AutoTokenizer
from diffuberta.helpers import extract_parallel, sanitize_json_value
from diffuberta.data import UniversalGenerator, OvershootDataset
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import ModelCard, ModelCardData
import datetime

# --- Configuration ---
model_id = "answerdotai/ModernBERT-base"
hub_model_id = 'philipp-zettl/DiffuBERTa'

# --- Model & Tokenizer Setup ---
model = AutoModelForMaskedLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SLOT_LEN = 10 

# --- Data Generation ---
# Generate 20000 diverse samples (as per your script)
gen = UniversalGenerator()
train_data = [gen.get_sample() for _ in range(20000)]
test_data = train_data[:int(len(train_data)*0.2)]
train_data = train_data[len(test_data):]
print(f"Sample 0: {train_data[0]}")

dataset = OvershootDataset(train_data, tokenizer)
test_dataset = OvershootDataset(test_data, tokenizer)

# --- LoRA Config ---
if 'ModernBERT-base' in model_id:
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        inference_mode=False,
        r=16,                  
        lora_alpha=32,         
        lora_dropout=0.1,      
        bias="none",
        target_modules=["Wqkv", "Wo", "W1", "W2"] 
    )
else:
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False, 
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"]
    )

# Wrap the model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir="./json_diff_model",
    num_train_epochs=3,
    per_device_train_batch_size=12,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="no",
    report_to="wandb",
    hub_model_id=hub_model_id,
    push_to_hub=True,
    run_name='DiffuBERTa',
    do_eval=True,
    eval_strategy='epoch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=test_dataset,
)

print("Starting Fine-Tuning...")
train_result = trainer.train()
print("Training Complete.")

# Save and Push initial files
trainer.save_model('./json_diff_model')
trainer.push_to_hub()

# --- Post-Training Evaluation & Inference ---

# 1. Switch to CPU/Eval mode
model = model.to('cpu')
model.eval()

# 2. Test Input for Validation
test_text = "We are excited to welcome Dr. Sarah to our Paris office as Senior Data Scientist."
test_instr = "Extract details"
test_template = '{"name": "[1]", "job": "[2]", "city": "[1]"}'

# 3. Run Inference
result = extract_parallel(tokenizer, model, test_text, test_instr, test_template, steps=5)
print(f"\nFine-tuned Result: {result}")

# ---------------------------------------------------------
#    AUTOMATED MODEL CARD UPDATE
# ---------------------------------------------------------
print("Updating Model Card...")

# 1. Extract Training Metrics
# trainer.state.log_history contains a list of dicts with logs
train_loss = "N/A"
eval_loss = "N/A"

# Iterate backwards to find the last recorded values
for log in reversed(trainer.state.log_history):
    if "loss" in log and train_loss == "N/A":
        train_loss = log["loss"]
    if "eval_loss" in log and eval_loss == "N/A":
        eval_loss = log["eval_loss"]

# 2. Define Card Metadata (YAML)
card_data = ModelCardData(
    language="en",
    license="apache-2.0",
    library_name="peft",
    tags=["json-extraction", "modernbert", "lora", "diffuberta"],
    datasets=["generated-json-pairs"],
    metrics=[
        {"name": "train_loss", "value": train_loss},
        {"name": "eval_loss", "value": eval_loss},
    ]
)

# 3. Define the Content (Markdown)
# We include the specific result we just calculated to prove it works
readme_content = f"""
---
{card_data.to_yaml()}
---

# DiffuBERTa: JSON Extraction Adapter

This model is a Fine-tuned version of **{model_id}** using LoRA. It is designed to extract structured JSON data from unstructured text using a parallel decoding approach.

## Model Performance
- **Final Training Loss**: {train_loss}
- **Final Evaluation Loss**: {eval_loss}
- **Training Epochs**: {training_args.num_train_epochs}
- **Date Trained**: {datetime.date.today()}

## 🚀 Live Demo Output
*(Generated automatically after training)*

**Input Text:**
> "{test_text}"

**Template:**
> `{test_template}`

**Model Output:**
```json
{json.dumps(result, indent=2)}
```

## Usage
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForMaskedLM.from_pretrained("{model_id}")
model = PeftModel.from_pretrained(base_model, "{hub_model_id}")
# ... use extract_parallel helper ...
```
"""

# 4. Push the Card
# We load the existing card (if any) to preserve other metadata, or create new if not found.
try:
    card = ModelCard.load(hub_model_id)
    card.text = readme_content # Replace content
    card.data = card_data      # Update metadata
except:
    card = ModelCard(readme_content)

card.push_to_hub(hub_model_id)
print("Model Card updated successfully on the Hub!")
