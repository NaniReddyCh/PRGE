# PRGE: On-Device Fine-Tuning with LoRA and Zeroth-Order Optimization

This repository contains research code for **on-device, privacy-preserving fine-tuning** of large language models using LoRA adapters and zeroth-order (gradient-free) optimization.  
It is designed for efficient, parameter-efficient adaptation on user data, suitable for edge devices.

## Features

- **LoRA (Low-Rank Adaptation):** Parameter-efficient fine-tuning by updating only small adapter modules.
- **Zeroth-Order Optimization (PRGE):** Gradient-free, forward-only optimization for environments where backpropagation is expensive or unavailable.
- **On-Device Focus:** All computation and data remain on the user's device for privacy.
- **Streamlit UI:** Simple web interface for running and visualizing fine-tuning.
- **Model Saving:** Fine-tuned models and tokenizers are saved for later use.

## Directory Structure

```
PRGE/
├── main.py           # Streamlit UI entry point
├── train.py          # Training loop and evaluation
├── prge_optimizer.py # Zeroth-order optimizer implementation
├── model_utils.py    # Model preparation and LoRA utilities
├── dataset_utils.py  # Data loading and preprocessing
├── test.py           # Script to test the fine-tuned model
├── plots.py          # Plotting utilities
├── requirements.txt  # Python dependencies
├── .gitignore
└── README.md
```

## Quick Start

1. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**

   ```sh
   streamlit run main.py
   ```

3. **Fine-tune a model:**

   - Select model and hyperparameters in the UI.
   - Click "Start Training".
   - Fine-tuned models are saved in timestamped folders.

4. **Test your model:**
   - Use `test.py` to load and generate text with your fine-tuned model.

## Example: Testing a Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "fine_tuned_model_YYYYMMDD_HHMMSS"  # replace with your folder
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

prompt = "Your test prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Notes

- For gated models (like Gemma), you must have access and be authenticated with Hugging Face.
- Results are saved to `results.json` during training.
- Only LoRA adapter weights are updated for efficiency.

## License

This code is for research and educational purposes only.  
Built as part of the **Samsung EnnovateX Hackathon 2025**.

---

## Authors

**Team ByteBots**

**PRGE** — Parameter-efficient, privacy-preserving, on-device fine-tuning
