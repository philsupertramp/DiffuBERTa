import random
import json
import string
import datetime
import torch
from torch.utils.data import Dataset

class MockDatabase:
    """
    Holds vocabulary lists to generate variety without external dependencies.
    """
    def __init__(self):
        self.first_names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi"]
        self.last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller"]
        self.cities = ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Mumbai"]
        self.medical_conditions = ["migraine", "hypertension", "insomnia", "nausea", "fracture"]
        self.iot_levels = ["INFO", "WARN", "ERROR", "CRITICAL", "DEBUG"]
        self.currencies = ["USD", "EUR", "JPY", "GBP", "CAD"]
        self.products = ["Laptop", "Mouse", "Keyboard", "Monitor", "Headset", "Webcam"]
        self.cities = [("New York", "JFK"), ("London", "LHR"), ("Berlin", "BER"), ("Tokyo", "HND"), ("Paris", "CDG")]
        
    def name(self): return f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
    def city(self): return random.choice(self.cities)


class UniversalGenerator:
    def __init__(self):
        self.db = MockDatabase()
        
        # We define "Templates" for different logic types
        self.domains = [
            self.generate_product,
            self.generate_event,
            self.generate_finance, # New: Nested Objects
            self.generate_iot_log, # New: Structured Logs
            self.generate_medical, # New: Array Extraction
            self.generate_navigation, # New: Sequential Instructions

            self.generate_shopping_cart,     # List of Objects (Items)
            self.generate_flight_itinerary,  # Ordered List (Route)
            self.generate_hr_record,          # Deep Nesting (Manager -> Employee)

            # Edge case generators
            self.generate_short_values,      # Single-token values with padding
            self.generate_empty_or_single,   # Empty/single item lists
            self.generate_numbers_only,      # Numeric extraction
        ]

    def get_sample(self, add_noise=True):
        """
        Main entry point. 
        add_noise: If True, randomly adds typos or formatting errors to input text.
        """
        generator = random.choice(self.domains)
        data = generator()
        
        if add_noise and random.random() < 0.3: # 30% chance to add noise
            data['text'] = self._inject_noise(data['text'])
            
        return data

    def _inject_noise(self, text):
        """Simulates real-world messy data (typos, extra spaces, missing chars)."""
        if not text: return text
        aug_type = random.choice(['swap', 'drop', 'space'])
        chars = list(text)
        idx = random.randint(0, len(chars) - 2)
        
        if aug_type == 'swap':
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        elif aug_type == 'drop':
            chars.pop(idx)
        elif aug_type == 'space':
            chars.insert(idx, " ")
            
        return "".join(chars)

    # --- Domain 1: E-Commerce (Standard Extraction) ---
    def generate_product(self):
        items = ["Gaming Laptop", "Coffee Maker", "Running Shoes", "Drill", "OLED TV"]
        adjectives = ["New", "Refurbished", "Used", "Open-Box"]
        prices = [f"${random.randint(10, 2000)}", f"{random.randint(10, 2000)} USD"]
        
        item = random.choice(items)
        adj = random.choice(adjectives)
        price = random.choice(prices)
        
        # Varying template structure
        templates = [
            f"Check out this {adj} {item} for only {price}.",
            f"Price drop: {item} ({adj}) is now {price}!",
            f"{price} is the asking price for this {adj} {item}."
        ]
        
        # Randomize Keys (Prevent overfitting to 'product_name')
        k_item = random.choice(["product", "item_name", "desc"])
        k_price = random.choice(["price", "cost", "val"])
        k_cond = random.choice(["condition", "state", "type"])
        
        return {
            "text": random.choice(templates),
            "json": json.dumps({k_item: item, k_price: price, k_cond: adj}),
            "type": "ecommerce"
        }

    # --- Domain 2: Events (Contextual Extraction) ---
    def generate_event(self):
        events = ["Conference", "Meetup", "Wedding", "Hackathon"]
        time_str = f"{random.randint(1,12)}:{random.choice(['00','30','15','45'])} {random.choice(['AM','PM'])}"
        loc = self.db.city()[0]
        
        templates = [
            f"Don't miss the {events[0]} in {loc} at {time_str}.",
            f"Schedule update: {events[0]} @ {loc} starts at {time_str}.",
            f"{loc} is hosting a {events[0]} today. Doors: {time_str}."
        ]
        text = random.choice(templates)
        
        # Distractor: Add a random date that isn't in the JSON to confuse the model
        if random.random() > 0.5:
            text += f" (Sent on {random.randint(1,28)}th)"

        return {
            "text": text,
            "json": json.dumps({"event": events[0], "location": loc, "time": time_str}),
            "type": "event"
        }

    # --- Domain 3: Finance (Nested JSON Objects) ---
    def generate_finance(self):
        """Generates nested JSON which is harder for models to learn."""
        sender = self.db.name()
        receiver = self.db.name()
        amount = random.randint(100, 50000)
        currency = random.choice(self.db.currencies)
        
        text = f"Transfer authorization: {sender} sent {amount} {currency} to {receiver}."
        
        # Nested Structure
        json_out = {
            "transaction": {
                "amount": amount,
                "currency": currency,
                "parties": {
                    "from": sender,
                    "to": receiver
                }
            },
            "status": "pending"
        }
        
        return {
            "text": text,
            "json": json.dumps(json_out),
            "type": "finance"
        }

    # --- Domain 4: IoT Logs (Structured Data + Arrays) ---
    def generate_iot_log(self):
        level = random.choice(self.db.iot_levels)
        device_id = f"DEV-{random.randint(1000, 9999)}"
        temp = random.randint(20, 90)
        
        text = f"[{level}] Device {device_id} (versin: v1.0) reported temperature: {temp}C. Connection stable."
        
        # Mixed types (Int and String)
        json_out = {
            "log_level": level,
            "metadata": [device_id, "v1.0"], # Array inside object
            "metrics": {"temperature": temp}
        }
        
        return {
            "text": text,
            "json": json.dumps(json_out),
            "type": "iot_log"
        }

    # --- Domain 5: Medical (List Extraction / NER) ---
    def generate_medical(self):
        """
        Simulates extracting a list of entities.
        """
        patient = self.db.name()
        # Pick 1 to 3 random conditions
        conditions = random.sample(self.db.medical_conditions, k=random.randint(1, 3))
        
        cond_str = ", ".join(conditions)
        
        templates = [
            f"Patient {patient} complains of {cond_str}.",
            f"Diagnosis for {patient}: {cond_str}.",
            f"Subjects presents symptoms including {cond_str}."
        ]
        
        # The goal is to extract the LIST of symptoms, ignoring the patient name
        return {
            "text": random.choice(templates),
            "json": json.dumps({"symptoms": conditions}), 
            "type": "medical"
        }

    # --- Domain 6: Navigation (Sequential Instructions) ---
    def generate_navigation(self):
        """
        Generates a list of instructions.
        """
        actions = ["Turn left", "Turn right", "Go straight", "Make a U-turn"]
        distances = [f"{random.randint(100,900)} meters", f"{random.randint(1,10)} miles"]
        
        step1_act = random.choice(actions)
        step1_dist = random.choice(distances)
        step2_act = random.choice(actions)
        
        text = f"{step1_act} for {step1_dist}, then {step2_act}."
        
        json_out = {
            "route": [
                {"action": step1_act, "distance": step1_dist},
                {"action": step2_act, "distance": None}
            ]
        }
        
        return {
            "text": text,
            "json": json.dumps(json_out),
            "type": "navigation"
        }
    def generate_shopping_cart(self):
        num_items = random.randint(1, 3)
        items = []
        text_parts = []
        
        for _ in range(num_items):
            prod = random.choice(self.db.products)
            qty = random.randint(1, 5)
            price = random.randint(10, 100)
            
            # Create a mini-sentence for this item
            if qty == 1:
                item_text = f"a {prod} for ${price}"
            else:
                item_text = f"{qty} {prod}s at ${price} each"
            
            text_parts.append(item_text)
            
            # Add to structured list
            items.append({
                "product": prod,
                "quantity": qty,
                "unit_price": price,
                "total": qty * price
            })

        # Combine text naturally
        if len(text_parts) == 1:
            full_text = f"I want to buy {text_parts[0]}."
        else:
            full_text = f"Add {', '.join(text_parts[:-1])} and {text_parts[-1]} to my cart."

        return {
            "text": full_text,
            "json": json.dumps({"intent": "cart_add", "items": items}),
            "type": "shopping_cart"
        }

    def generate_flight_itinerary(self):
        # Pick 2 or 3 distinct cities
        route_len = random.choice([2, 3])
        route_cities = random.sample(self.db.cities, route_len)
        
        airline = random.choice(["Delta", "United", "Lufthansa"])
        flight_num = f"{random.randint(100, 999)}"
        
        segments = []
        
        # Build text and json structure together
        origin_city, origin_code = route_cities[0]
        dest_city, dest_code = route_cities[-1]
        
        text = f"Book {airline} flight {flight_num} from {origin_city} ({origin_code}) to {dest_city} ({dest_code})"
        
        # Add metadata for Origin
        segments.append({"type": "departure", "airport": origin_code, "city": origin_city})

        # Handle Layover if it exists
        if route_len == 3:
            mid_city, mid_code = route_cities[1]
            text += f" with a layover in {mid_city}"
            segments.append({"type": "layover", "airport": mid_code, "city": mid_city})
            
        text += "."
        
        # Add metadata for Destination
        segments.append({"type": "arrival", "airport": dest_code, "city": dest_city})

        return {
            "text": text,
            "json": json.dumps({"flight": f"{airline} {flight_num}", "route": segments}),
            "type": "flight_itinerary"
        }

    def generate_hr_record(self):
        manager = self.db.name()
        employee = self.db.name()
        dept = random.choice(["Engineering", "Sales", "HR"])
        
        text = f"{employee} has joined the {dept} department. They will report directly to {manager}."
        
        structure = {
            "department": {
                "name": dept,
                "manager": {
                    "name": manager,
                    "role": "Lead"
                },
                "new_hire": {
                    "name": employee,
                    "status": "Onboarding"
                }
            }
        }
        
        return {
            "text": text,
            "json": json.dumps(structure),
            "type": "hr_record"
        }

    def generate_short_values(self):
        """
        EDGE CASE: Very short values (1-2 tokens) to teach proper padding.
        This is critical for the overshoot problem.
        """
        templates = [
            ("The status is {status}.", ["active", "pending", "done", "failed"]),
            ("Color: {color}", ["red", "blue", "green", "yellow"]),
            ("Size: {size}", ["S", "M", "L", "XL"]),
            ("Code: {code}", ["OK", "ERR", "404", "200"]),
            ("City: {city}", ["Paris", "Tokyo", "Berlin", "NYC"]),
        ]

        template, options = random.choice(templates)
        value = random.choice(options)
        text = template.format(**{template.split("{")[1].split("}")[0]: value})
        key = template.split("{")[1].split("}")[0]

        return {
            "text": text,
            "json": json.dumps({key: value}),
            "type": "short_value"
        }

    def generate_empty_or_single(self):
        """
        EDGE CASE: Empty lists or single-item lists.
        Teaches model to handle edge cases in list extraction.
        """
        case = random.choice(["empty", "single"])

        if case == "empty":
            texts = [
                "No items found.",
                "The list is empty.",
                "Nothing to display.",
            ]
            return {
                "text": random.choice(texts),
                "json": json.dumps({"items": []}),
                "type": "empty_list"
            }
        else:
            # Single item
            item = random.choice(self.db.products)
            text = f"Only one item available: {item}."
            return {
                "text": text,
                "json": json.dumps({"items": [item]}),
                "type": "single_item"
            }

    def generate_numbers_only(self):
        """
        EDGE CASE: Numeric values (ages, prices, quantities, years).
        Numbers tokenize differently than text.
        """
        num_type = random.choice(["age", "price", "quantity", "year", "temperature"])

        if num_type == "age":
            value = random.randint(18, 75)
            text = f"Patient is {value} years old."
        elif num_type == "price":
            value = random.randint(5, 500)
            text = f"Total: ${value}"
        elif num_type == "quantity":
            value = random.randint(1, 100)
            text = f"Stock: {value} units"
        elif num_type == "year":
            value = random.randint(2000, 2025)
            text = f"Released in {value}"
        else:  # temperature
            value = random.randint(-20, 45)
            text = f"Temperature: {value}°C"

        return {
            "text": text,
            "json": json.dumps({num_type: str(value)}),  # Keep as string for consistency
            "type": "numeric"
        }


class OvershootDataset(Dataset):
    def __init__(self, data, tokenizer, slot_len=10, variable_slots=True):
        self.data = data
        self.tokenizer = tokenizer
        self.instruction = "Extract details"
        self.pad_id = tokenizer.pad_token_id
        self.mask_id = tokenizer.mask_token_id
        self.SLOT_LEN = slot_len
        self.variable_slots = variable_slots  # Use adaptive slot sizing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
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
                val = json.dumps(json_obj[key])
                
                # --- A. The Key ---
                # '"key": "'
                key_text = f'"{key}": "'
                key_ids = self.tokenizer.encode(key_text, add_special_tokens=False)
                
                input_ids.extend(key_ids)
                labels.extend([-100] * len(key_ids)) # Don't train on keys
                
                # --- B. The Value (OVERSHOOT) ---
                val_ids = self.tokenizer.encode(val, add_special_tokens=False)

                # Adaptive slot sizing: actual length + random buffer
                if self.variable_slots:
                    # Add 1-5 extra tokens as buffer (teaches variable padding)
                    buffer = random.randint(1, 5)
                    current_slot_len = min(len(val_ids) + buffer, self.SLOT_LEN)
                else:
                    current_slot_len = self.SLOT_LEN

                # Truncate if too long (rare)
                if len(val_ids) > current_slot_len:
                    val_ids = val_ids[:current_slot_len]

                # Calculate Padding needed
                pad_len = current_slot_len - len(val_ids)

                # INPUT: All MASKS
                input_ids.extend([self.mask_id] * current_slot_len)

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

                # Validate token IDs are within vocabulary range
                vocab_size = self.tokenizer.vocab_size
                input_ids = [min(max(0, tid), vocab_size - 1) for tid in input_ids]
                labels = [min(max(-100, l), vocab_size - 1) if l != -100 else -100 for l in labels]

                # Create tensors
                input_tensor = torch.tensor(input_ids, dtype=torch.long)
                label_tensor = torch.tensor(labels, dtype=torch.long)
                attn_mask = torch.tensor([int(i != self.pad_id) for i in input_ids], dtype=torch.long)

                # Final validation - check for invalid values
                assert input_tensor.min() >= 0 and input_tensor.max() < vocab_size, f"Invalid input_ids: min={input_tensor.min()}, max={input_tensor.max()}, vocab_size={vocab_size}"
                assert label_tensor[label_tensor != -100].min() >= 0, f"Invalid labels: min non-pad={label_tensor[label_tensor != -100].min()}"
                assert not torch.isnan(input_tensor).any(), "NaN in input_ids"
                assert not torch.isnan(label_tensor).any(), "NaN in labels"

                return {
                    "input_ids": input_tensor,
                    "attention_mask": attn_mask,
                    "labels": label_tensor
                }

        except Exception as e:
            # Fallback: Return a simple valid example if data generation fails
            print(f"⚠️ Error in __getitem__ at idx {idx}: {e}")
            # Create a minimal valid sample
            simple_text = "Error sample"
            simple_json = json.dumps({"value": "error"})

            prefix_text = f"{simple_text} {self.tokenizer.sep_token} {self.instruction}: {{ "
            prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=True)[:max_seq_len]

            # Validate prefix_ids
            vocab_size = self.tokenizer.vocab_size
            prefix_ids = [min(max(0, tid), vocab_size - 1) for tid in prefix_ids]

            input_ids = prefix_ids + [min(self.pad_id, vocab_size - 1)] * (max_seq_len - len(prefix_ids))
            labels = [-100] * max_seq_len

            return {
                "input_ids": torch.tensor(input_ids[:max_seq_len], dtype=torch.long),
                "attention_mask": torch.tensor([1] * len(prefix_ids) + [0] * (max_seq_len - len(prefix_ids)), dtype=torch.long)[:max_seq_len],
                "labels": torch.tensor(labels[:max_seq_len], dtype=torch.long)
            }
