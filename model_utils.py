import torch.nn as nn
from lora_fa_layer import LoRAFALayer

def prepare_model_for_prge(model: nn.Module, rank=16, lora_alpha=32, target_modules=["q_proj","v_proj"]):
    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            parts = name.split('.')
            parent_name, child_name = ".".join(parts[:-1]), parts[-1]
            parent_module = model.get_submodule(parent_name)
            new_layer = LoRAFALayer(module, rank, lora_alpha)
            setattr(parent_module, child_name, new_layer)
            print(f"Replaced {name} with LoRAFALayer.")
    return model

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
