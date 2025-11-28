import torch
import copy
import torch.nn.functional as F
import re
import json
import math

def expand_template(json_template, tokenizer):
    # Expands [N] notation to N mask tokens
    token = tokenizer.mask_token
    return re.sub(r'\[(\d+)\]', lambda m: " ".join([token] * int(m.group(1))), json_template)


def sanitize_json_value(raw_value, tokenizer=None):
    """
    The 'Smart Padding' Logic.
    If the model generates: "Paris [PAD] [PAD] ..."
    We cut it at "Paris".
    Also handles structural tokens like quotes, commas, braces, colons.
    """
    if isinstance(raw_value, list):
        out = []
        for elem in raw_value:
            out.append(sanitize_json_value(elem, tokenizer))
        return out

    # 0. Remove padding tokens if tokenizer is provided
    if tokenizer and tokenizer.pad_token:
        raw_value = raw_value.replace(tokenizer.pad_token, ' ')

    # 1. Smart structural token detection
    # Only cut at structural tokens if they appear in suspicious JSON-like patterns
    # Don't cut at colons/periods that are clearly part of data (timestamps, IPs, numbers)

    # First, handle obvious JSON structural tokens that should always stop extraction
    hard_structural = ['{', '}', '[', ']']
    for token in hard_structural:
        if token in raw_value:
            raw_value = raw_value.split(token)[0]
            break

    # For quotes and commas, only cut if followed by whitespace or another structural token
    # This prevents cutting "Paris, Texas" but does cut "Paris', 'country'"
    import re
    # Cut at comma followed by space and quote (JSON pattern: "Paris", "France")
    if re.search(r',\s*["\']', raw_value):
        raw_value = re.split(r',\s*["\']', raw_value)[0]
    # Cut at quote followed by comma or colon (JSON pattern: "Paris": or "Paris",)
    elif re.search(r'["\'][\s,:]', raw_value):
        raw_value = re.split(r'["\'][\s,:]', raw_value)[0]
    # Cut at standalone quote (but allow quotes in middle like "O'Brien")
    elif raw_value.endswith(('"', "'")):
        raw_value = raw_value.rstrip('"\'')

    # 2. Strip whitespace and common artifacts
    raw_value = raw_value.strip()

    # 3. Remove common padding artifacts (repeated dots, ellipsis, etc)
    if raw_value.endswith('...'):
        raw_value = raw_value[:-3].strip()

    return raw_value


def extract_diffusion(tokenizer, model, text, instruction, template_shorthand, steps=10, verbose=False):
    # 1. Expand Template
    # Regex to turn "[8]" into "[MASK] [MASK] [MASK]..."
    json_template = expand_template(template_shorthand, tokenizer)
    if verbose:
        print(json_template)
    
    full_prompt = f"{text} {tokenizer.sep_token} {instruction}: {json_template}"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    mask_token_id = tokenizer.mask_token_id
    
    # Identify Value Slots
    value_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]
    total_slots = len(value_indices)
    if steps == -1:
        steps = total_slots
    
    print(f"--- Diffusion Extraction ({steps} steps / {total_slots} slots) ---")

    for step in range(steps):
        # 0. Snapshot previous state for comparison
        prev_input_ids = input_ids.clone()

        # A. Schedule (How many to keep?)
        progress = (step + 1) / steps
        n_to_keep = int(total_slots * progress)
        if step == steps - 1: n_to_keep = total_slots

        # B. Forward Pass
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        
        # C. Probabilities & Ranking
        relevant_logits = logits[0, value_indices, :]
        probs = F.softmax(relevant_logits, dim=-1)
        token_confs, token_ids = torch.max(probs, dim=-1)
        
        # Sort by confidence
        sorted_indices = torch.argsort(token_confs, descending=True)
        
        # D. Update Step
        # 1. Keep/Write High Confidence
        indices_to_keep_local = sorted_indices[:n_to_keep]
        keep_global_indices = value_indices[indices_to_keep_local]
        input_ids[0, keep_global_indices] = token_ids[indices_to_keep_local]
        
        # 2. Re-Mask Low Confidence
        indices_to_mask_local = sorted_indices[n_to_keep:]
        if len(indices_to_mask_local) > 0:
            mask_global_indices = value_indices[indices_to_mask_local]
            input_ids[0, mask_global_indices] = mask_token_id
            
        # --- CHANGE TRACKING LOGIC ---
        changes = []
        # We iterate only over the value slots to see what flipped
        for idx in value_indices:
            old_id = prev_input_ids[0, idx]
            new_id = input_ids[0, idx]
            
            if old_id != new_id:
                old_tok = tokenizer.decode([old_id]).strip()
                new_tok = tokenizer.decode([new_id]).strip()
                
                # Filter out boring "<mask> -> <mask>" (shouldn't happen due to if check)
                # We mainly care about Token->Token or Mask->Token
                changes.append(f"'{old_tok}' -> '{new_tok}'")

        # Visualization
        avg_conf = token_confs.mean().item()
        
        # Format the output nicely
        if verbose:
            print(f"Step {step+1}: Avg Conf: {avg_conf:.2f}")
            if changes:
                # Show first 5 changes to avoid spamming console
                print(f"   📝 Changes: {', '.join(changes[:5])}" + ("..." if len(changes) > 5 else ""))
            else:
                print(f"   📝 No changes this step.")
            
    # Final Cleanup
    result_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    try:
        json_part = result_text.split(f"{instruction}:")[1]
        clean_json = json_part.replace(tokenizer.pad_token, "").replace(tokenizer.sep_token, "").replace("<s>", "").replace("</s>", "").strip()
        return clean_json
    except:
        return result_text


def create_mask_struct(schema_node, tokenizer):
    """
    Converts a schema definition into a Masked String structure.
    e.g. {"name": "[8]"} -> '{"name": "[MASK] [MASK]..."}'
    """
    mask_token = tokenizer.mask_token
    if isinstance(schema_node, str):
        # Convert "[8]" to 8 masks
        count = int(re.search(r'\[(\d+)\]', schema_node).group(1))
        return " ".join([mask_token] * count)
    
    elif isinstance(schema_node, dict):
        # Recursively build dict string
        parts = []
        for k, v in schema_node.items():
            val_str = create_mask_struct(v, tokenizer)
            parts.append(f'"{k}": {val_str}')
        return "{ " + ", ".join(parts) + " }"
    
    elif isinstance(schema_node, list):
        # For lists, the schema just defines the structure of A SINGLE ITEM
        # We don't return a list of masks here, we handle lists in the loop.
        # This function is only for Fixed Object structures.
        return "[]" # Placeholder

def is_padding_only(data_dict, pad_token='<pad>'):
    """
    Checks if a generated object is effectively empty/padding.
    """
    json_str = json.dumps(data_dict)
    # If the extracted values are all empty strings or <pad>
    # (Assuming the extract_diffusion function returns empty strings for pads)
    has_content = False
    
    def check(node):
        nonlocal has_content
        if isinstance(node, dict):
            for v in node.values(): check(v)
        elif isinstance(node, list):
            for v in node: check(v)
        elif isinstance(node, str):
            if len(node.strip()) > 0 and pad_token not in node:
                has_content = True
                
    check(data_dict)
    return not has_content


def extract_parallel(tokenizer, model, text, instruction, schema, current_data=None, steps=10, verbose=False):
    """
    The Master Function.
    Walks through the schema and iteratively fills it using the model.
    """
    if current_data is None:
        current_data = {}

    # 1. PHASE A: Fill Fixed Fields (Strings/Numbers at this level)
    # We build a partial template for just the keys at this level
    working_template = {}
    list_keys = []
    
    for key, value_schema in schema.items():
        if isinstance(value_schema, list):
            list_keys.append(key)
            # We will handle this later, put placeholder for now
            working_template[key] = [] 
        else:
            # It's a simple value or nested dict, prepare masks
            # If we already have data (from recursion), use it.
            if key in current_data:
                working_template[key] = current_data[key] # Keep existing
            else:
                working_template[key] = create_mask_struct(value_schema, tokenizer)

    # If there are any masks to fill at this level, run Diffusion
    # (We convert the working_template dict to a string, but leave the mask strings raw)
    # Note: We need a custom dumper that doesn't escape the <mask> strings
    
    def dict_to_template_str(d):
        # Quick hack to dump dict to string but keep <mask> unquoted if needed
        # Actually, our diffusion function expects standard string inputs.
        # We can just use json.dumps but we need to ensure <mask> isn't double escaped.
        # For simplicity, let's assume standard json.dumps works if masks are strings.
        s = json.dumps(d)
        # Cleanup: json.dumps might escape quotes inside the mask string. 
        # But our create_mask_struct returns raw mask strings. 
        # We might need to post-process to remove quotes around masks if the model expects that.
        # Let's assume input format: '{"key": "<mask> <mask>"}' is valid.
        return s.replace('"{', '{').replace('}"', '}') # Unquote nested objects if stringified

    # Only run if there are actual masks at this level
    if any(tokenizer.mask_token in str(v) for v in working_template.values()):
        template_str = dict_to_template_str(working_template)
        # Run Extraction
        filled_json_str = extract_diffusion(tokenizer, model, text, instruction, template_str, steps=steps, verbose=verbose)
        # Parse result
        if isinstance(filled_json_str, dict):
            return filled_json_str

        try:
            current_data = json.loads(filled_json_str)
        except:
            print(f"⚠️ JSON Parse Error at Root Level: {filled_json_str}")
            return current_data

    # 2. PHASE B: Handle Lists (The Expansion Loop)
    for list_key in list_keys:
        item_schema = schema[list_key][0] # Get the schema for items in this list
        
        # Ensure list exists
        if list_key not in current_data:
            current_data[list_key] = []
            
        print(f"--- Expanding List: {list_key} ---")
        
        # Expansion Loop
        while True:
            # Create a Deep Copy of current state to act as Context
            context_data = copy.deepcopy(current_data)
            
            # Append a NEW mask object to the list in the context
            # This is the "Probe"
            mask_item_str = create_mask_struct(item_schema, tokenizer)
            # We temporarily store this as a string, we'll need to handle it in template creation
            # Ideally, we inject the specific mask string.
            
            # Construct the prompt string manually to ensure correct format
            # " ... 'list_key': [ {existing}, {existing}, {NEW_MASKS} ] ... "
            
            # A. Serialize current known data
            prefix_data = copy.deepcopy(current_data)
            del prefix_data[list_key] # Remove the list we are working on
            
            # Serialize the list items we already found
            known_items_str = ", ".join([json.dumps(x) for x in current_data[list_key]])
            
            # B. Create the new Item Template
            new_item_template = mask_item_str
            
            # C. Combine
            if known_items_str:
                list_str = f"[ {known_items_str}, {new_item_template} ]"
            else:
                list_str = f"[ {new_item_template} ]"
                
            # D. Inject into parent JSON string
            # This is the hard part: Inserting the list string into the parent JSON.
            # For this MVP, let's just use the List String as the Target if the context is simple.
            # But to preserve parent context (like Company Name), we should include it.
            
            # Simplification: We prompt with the Parent Context String
            parent_json = json.dumps(prefix_data)
            # We hack the string: remove the last brace '}' and append our list
            if prefix_data:
                # Has other fields: {"field1": "val1", "list": [...]}
                full_template = parent_json[:-1] + f', "{list_key}": {list_str} }}'
            else:
                # Empty prefix: {"list": [...]}
                full_template = f'{{ "{list_key}": {list_str} }}'
            
            # RUN DIFFUSION
            # We expect the model to return the FULL string.
            print(f"Scanning for item #{len(current_data[list_key])+1}...")
            filled_str = extract_diffusion(tokenizer, model, text, instruction, full_template, steps=steps, verbose=verbose)
            
            try:
                full_obj = json.loads(filled_str)
                # Extract the LAST item from the target list
                new_item = full_obj[list_key][-1]
                
                # Check for Stop Condition (Padding)
                if is_padding_only(new_item, tokenizer.pad_token):
                    print(f"   -> Padding detected. List '{list_key}' complete.")
                    break
                
                # Check for Duplicates (Infinite loop guard)
                if new_item in current_data[list_key]:
                    print("   -> Duplicate detected. Stopping.")
                    break
                    
                print(f"   -> Found: {new_item}")
                current_data[list_key].append(new_item)
                
            except Exception as e:
                print(f"   -> Error parsing expansion: {e}")
                break

    return current_data
