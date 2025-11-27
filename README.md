# ⚡ DiffuBERTa: Non-Autoregressive Structured Extraction

> **Strict JSON generation using BERT as a Discrete Diffusion Model.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30%2B-yellow)](https://huggingface.co/docs/transformers/index)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)](https://github.com/huggingface/peft)

**DiffuBERTa** is a proof-of-concept framework that treats data extraction not as "text generation," but as **schema-constrained inpainting**. By using `roberta-base` with a custom parallel decoding strategy, this approach guarantees 100% valid JSON syntax and leverages bidirectional context to solve complex extraction tasks.

Inspired by [Nathan.rs: BERT is just a Single Text Diffusion Step](https://nathan.rs/posts/roberta-diffusion/).

---

## 🚀 Why Use This?

Standard LLMs (GPT-4, Llama) generate JSON token-by-token (left-to-right). This is slow and prone to syntax errors (missing brackets, unescaped quotes).

**DiffuBERTa is different:**
* **Guaranteed Syntax:** The structure (brackets, keys) is hard-coded. The model only fills the values.
* **Parallel Decoding:** We generate all fields simultaneously, refining them in iterative steps.
* **Bidirectional Attention:** The model can see the *end* of the JSON structure to inform the *start* of the value.
* **Overshoot & Truncate:** Handles variable-length entities naturally via trained padding.

---

## 📦 Installation

```bash
git clone https://github.com/philsupertramp/DiffuBERTa.git
cd DiffuBERTa
pip install -r requirements.txt
````

# Details
## 🧠 Approach

### 1\. The "Overshoot" Strategy

BERT models operate on fixed-length sequences. To handle variable extraction (e.g., a name that could be "Al" or "Alexander"), we use an **Overshoot Template**. We allocate a buffer of masks (e.g., 8 tokens) for every field.

  * **Input:** `{"city": <mask> <mask> <mask> <mask>}`
  * **Prediction:** `{"city": Paris <pad> <pad> <pad>}`

We fine-tune the model to explicitly predict `<pad>` tokens when the semantic meaning is complete, preventing hallucinations.

### 2\. Parallel Refinement

Instead of greedy decoding, we use a multi-step diffusion process:

1.  **Step 1:** Predict logits for all masks. Lock in the top 25% most confident tokens.
2.  **Step 2-4:** Re-read the partially filled sequence. Use the new context to resolve ambiguous tokens.
3.  **Final:** All masks are filled.

-----

## 🛠️ Usage

### 1\. Training (Fine-Tuning)

We use **LoRA (Low-Rank Adaptation)** to teach `roberta-base` the JSON schema without destroying its pre-trained world knowledge. The script generates synthetic multi-domain data on the fly.

```bash
python train.py
```

*This will create a `json_diff_model` directory with the LoRA adapters.*

### 2\. Inference (Extraction)

Use the fine-tuned model to extract data from raw text.

```bash
python script.py
```

**Example Code:**

```python
from script import extract_parallel

text = "The SpaceX Starship launch happened at Starbase, Texas. It stands 120m tall."
instruction = "Extract location and height"
template = '{"location": "[8]", "height": "[5]"}' # [N] denotes mask buffer size

result = extract_parallel(text, instruction, template, steps=5)
print(result)
# Output: {"location": "Starbase, Texas", "height": "120m"}
```

-----

## 📊 Performance vs. Autoregressive

| Feature | Autoregressive (GPT) | DiffuBERTa (Ours) |
| :--- | :--- | :--- |
| **Latency** | Linear $O(N)$ (Slow for long JSONs) | **Sub-linear / Constant** (via Parallelism) |
| **Syntax Errors** | Possible (Hallucinated braces) | **Impossible** (Fixed Template) |
| **Context Window** | Previous tokens only | **Full Sequence (Past & Future)** |

-----

## 🔮 Future Roadmap

  - [ ] Implement Confidence-Based Branching (Beam Search for specific slots).
  - [ ] Support nested lists via dynamic template expansion.
  - [ ] Port to T5 encoder-decoder for native variable-span masking.

-----

## 🤝 Contributing

Open an issue or submit a PR if you want to help solve the variable-length masking problem\!

## 📜 License

MIT

