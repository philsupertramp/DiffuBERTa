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
# Generate more diverse samples for better learning
gen = UniversalGenerator()
train_data = [gen.get_sample() for _ in range(10000)]  # 10k samples - balanced size for GPU stability
test_data = [gen.get_sample() for _ in range(1000)]  # Separate test set (10%)
print(f"Sample 0: {train_data[0]}")

# Use variable_slots=True for adaptive slot sizing during training
dataset = OvershootDataset(train_data, tokenizer, slot_len=10, variable_slots=True)
test_dataset = OvershootDataset(test_data, tokenizer, slot_len=10, variable_slots=True)

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
    num_train_epochs=5,  # Increased from 3 to 5 for better learning
    per_device_train_batch_size=2,  # Minimum batch size to avoid CUDA issues
    gradient_accumulation_steps=8,  # Effective batch size = 16
    learning_rate=3e-5,  # Slightly lower for more stable training
    weight_decay=0.01,
    max_grad_norm=1.0,  # Gradient clipping to prevent instability
    logging_steps=50,
    save_strategy="epoch",  # Save checkpoints each epoch
    save_total_limit=2,  # Keep only best 2 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="wandb",
    hub_model_id=hub_model_id,
    push_to_hub=True,
    run_name='DiffuBERTa-v2-CPU',
    do_eval=True,
    eval_strategy='epoch',
    warmup_steps=500,  # Warmup for stable training
    fp16=False,  # Disable mixed precision to avoid instability
    dataloader_num_workers=0,  # Avoid multiprocessing issues
    no_cuda=True,  # Force CPU training for stability
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=test_dataset,
)

print("Starting Fine-Tuning...")

# Clear CUDA cache before training to avoid memory issues
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("CUDA cache cleared")

try:
    train_result = trainer.train()
    print("Training Complete.")
except RuntimeError as e:
    if "CUDA" in str(e):
        print(f"⚠️ CUDA Error occurred: {e}")
        print("Try reducing batch_size or sequence length")
    raise

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
test_template = {"name": "[1]", "job": "[2]", "city": "[1]"}

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
