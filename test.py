from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "fine_tuned_model_YYYYMMDD_HHMMSS"  # use the actual folder name
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

prompt = "Your test prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))