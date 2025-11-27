import random
import json
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import json
import math
import re
from transformers import Trainer, TrainingArguments, RobertaForMaskedLM, RobertaTokenizer

# Standardize: Every value gets exactly 10 slots.
SLOT_LEN = 10 


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


class UniversalGenerator:
    def __init__(self):
        # We define "Templates" for different logic types
        self.domains = [
            self.generate_product,
            self.generate_event,
            self.generate_science,
            self.generate_social
        ]

    def get_sample(self):
        # Pick a random domain logic
        generator = random.choice(self.domains)
        return generator()

    # --- Domain 1: E-Commerce (Price/Attribute extraction) ---
    def generate_product(self):
        items = ["Gaming Laptop", "Coffee Maker", "Running Shoes", "Drill"]
        prices = ["$500", "20 euros", "1500 JPY", "99 bucks"]
        colors = ["Red", "Matte Black", "Silver", "Neon"]
        
        i, p, c = random.choice(items), random.choice(prices), random.choice(colors)
        
        text = f"Flash sale! Get the new {c} {i} for only {p} today."
        # Randomize Keys to prevent memorization
        keys = random.choice([
            ("product", "cost", "finish"),
            ("item_name", "price_tag", "color_variant"),
            ("obj", "val", "col") 
        ])
        
        json_out = json.dumps({keys[0]: i, keys[1]: p, keys[2]: c})
        return {"text": text, "json": json_out, "type": "product"}

    # --- Domain 2: Events (Time/Location extraction) ---
    def generate_event(self):
        events = ["Conference", "Meetup", "Wedding", "Hackathon"]
        times = ["2 PM", "14:00", "midnight", "dawn"]
        locs = ["Room 101", "the Central Park", "Zoom", "Mars Base"]
        
        e, t, l = random.choice(events), random.choice(times), random.choice(locs)
        
        text = f"Don't miss the {e} happening at {l}. Doors open at {t}."
        json_out = json.dumps({"what": e, "where": l, "when": t})
        return {"text": text, "json": json_out, "type": "event"}

    # --- Domain 3: Science (Fact extraction) ---
    def generate_science(self):
        elements = ["Helium", "Iron", "Carbon", "Gold"]
        properties = ["lighter than air", "magnetic", "organic", "conductive"]
        
        e, p = random.choice(elements), random.choice(properties)
        
        text = f"Properties of {e}: It is known to be {p}."
        json_out = json.dumps({"element": e, "characteristic": p})
        return {"text": text, "json": json_out, "type": "science"}

    # --- Domain 4: Nonsense (Pure Copy-Paste Logic) ---
    # This is crucial. It forces the model to ignore meaning and just follow instructions.
    def generate_social(self):
        users = ["@user1", "@cool_guy", "@admin"]
        stats = ["5k", "1M", "0"]
        
        u, s = random.choice(users), random.choice(stats)
        
        text = f"User {u} just hit {s} followers!"
        json_out = json.dumps({"handle": u, "count": s})
        return {"text": text, "json": json_out, "type": "social"}

# Generate 2000 diverse samples
gen = UniversalGenerator()
train_data = [gen.get_sample() for _ in range(2000)]

print(f"Sample 0: {train_data[0]}")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


class OvershootDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.instruction = "Extract details"
        self.pad_id = tokenizer.pad_token_id
        self.mask_id = tokenizer.mask_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Tokenize Context + Instruction
        # "Context </s> Instruction: { "
        prefix_text = f"{item['text']} {self.tokenizer.sep_token} {self.instruction}: {{ "
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=True)[:-1] # Remove last sep/eos
        
        input_ids = list(prefix_ids)
        labels = [-100] * len(prefix_ids) # Ignore prefix in loss
        
        # 2. Parse the JSON to build the Structured Blocks
        # We assume item['json'] is a dict (modify your generator to return dict, not str)
        # If it's a string, load it:
        import json
        if isinstance(item['json'], str):
            json_obj = json.loads(item['json'])
        else:
            json_obj = item['json']
            
        keys = list(json_obj.keys())
        for i, key in enumerate(keys):
            val = str(json_obj[key])
            
            # --- A. The Key ---
            # '"key": "'
            key_text = f'"{key}": "'
            key_ids = self.tokenizer.encode(key_text, add_special_tokens=False)
            
            input_ids.extend(key_ids)
            labels.extend([-100] * len(key_ids)) # Don't train on keys
            
            # --- B. The Value (OVERSHOOT) ---
            val_ids = self.tokenizer.encode(val, add_special_tokens=False)
            
            # Truncate if too long (rare)
            if len(val_ids) > SLOT_LEN:
                val_ids = val_ids[:SLOT_LEN]
            
            # Calculate Padding needed
            pad_len = SLOT_LEN - len(val_ids)
            
            # INPUT: All MASKS
            input_ids.extend([self.mask_id] * SLOT_LEN)
            
            # LABEL: Actual IDs + PAD IDs
            # Crucial: We use self.pad_id, NOT -100. We WANT to learn this.
            labels.extend(val_ids + [self.pad_id] * pad_len)
            
            # --- C. Closure ---
            # Closing quote and comma/brace
            if i < len(keys) - 1:
                suffix = '", '
            else:
                suffix = '" }'
                
            suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
            input_ids.extend(suffix_ids)
            labels.extend([-100] * len(suffix_ids))
            
        # 3. Final Padding for Batching (Standard)
        # We need to pad the WHOLE sequence to a max length (e.g. 512) for the DataLoader
        # This is different from the "Value Padding" above.
        max_seq_len = 128
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            labels = labels[:max_seq_len]
        else:
            total_pad = max_seq_len - len(input_ids)
            input_ids.extend([self.pad_id] * total_pad)
            labels.extend([-100] * total_pad) # Ignore batch padding
            
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor([1 if i != self.pad_id else 0 for i in input_ids]), # Standard attention mask
            "labels": torch.tensor(labels)
        }

# UPDATE GENERATOR to return dicts
# (Small tweak to your UniversalGenerator to return the dict object in 'json' key, not string)

dataset = OvershootDataset(train_data, tokenizer)

from peft import get_peft_model, LoraConfig, TaskType
from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM.from_pretrained("roberta-base")

# Define LoRA Config
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, # or generic
    inference_mode=False, 
    r=8,            # Rank (smaller = less trainable parameters)
    lora_alpha=32,
    lora_dropout=0.1,
    # Apply to attention layers where the "copying" logic happens
    target_modules=["query", "value"] 
)

# Wrap the model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./json_diff_model",
    num_train_epochs=3,              # Short run for demo
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="no",              # Don't fill disk with checkpoints
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("Starting Fine-Tuning...")
trainer.train()
print("Training Complete.")
trainer.save_model('./json_diff_model')
trainer.push_to_hub('philipp-zettl/DiffuBERTa')

# 1. Switch to the fine-tuned model (it's in memory as 'model')
model = model.to('cpu')
model.eval()

# 2. Test Input
test_text = "We are excited to welcome Dr. Sarah to our Paris office as Senior Data Scientist."
test_instr = "Extract details"
# We can use the same generic template with overshot masks
test_template = '{"name": "[1]", "job": "[2]", "city": "[1]"}'

# 3. Run Inference (Reuse your extract_parallel function)
# Note: Ensure extract_parallel uses the global 'model' variable which is now fine-tuned
result = extract_parallel(test_text, test_instr, test_template, steps=5)

print(f"\nfine-tuned Result: {result}")
