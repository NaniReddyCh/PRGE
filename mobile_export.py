# mobile_export.py
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model
base_model = "gpt2"   # use same model as training
peft_model_path = "finetuned_model"  # path where LoRA/PEFT is saved

print("Loading base + LoRA weights...")
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, peft_model_path)
model = model.merge_and_unload()  # merge LoRA into base for export
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model)

# Dummy input for tracing
example_input = tokenizer("Hello, how are you?", return_tensors="pt")["input_ids"]

# Export TorchScript
print("Exporting to TorchScript...")
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_mobile.pt")

print("TorchScript model saved as model_mobile.pt")

# (Optional) Convert to ExecuTorch format
try:
    from torch.export import export
    from torch._export import capture_pre_autograd_graph
    print("Exporting to ExecuTorch format...")
    exported_program = export(model, (example_input,))
    exported_program.save("model_mobile.pte")
    print("ExecuTorch model saved as model_mobile.pte")
except Exception as e:
    print("ExecuTorch export failed (install latest PyTorch nightly). Error:", e)
