import torch
import torch.nn as nn
import math

class LoRAFALayer(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int = 16, lora_alpha: int = 32):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.lora_alpha = lora_alpha

        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_A.requires_grad = False

        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_features))
        self.lora_B.requires_grad = True

        self.scaling = self.lora_alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original_layer(x)
        lora_update = (x @ self.lora_A.to(x.dtype) @ self.lora_B.to(x.dtype)) * self.scaling
        return original_output + lora_update
