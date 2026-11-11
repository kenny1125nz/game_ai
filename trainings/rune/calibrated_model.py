import torch
import torch.nn as nn

class CalibratedModel(nn.Module):
    def __init__(self, base_model, temperature=1.0):
        super().__init__()
        self.base_model = base_model
        # Use register_buffer so it's part of the model state but not a "parameter"
        self.register_buffer('T', torch.tensor([temperature], dtype=torch.float32))
    
    def forward(self, x):
        logits = self.base_model(x)
        return logits / self.T