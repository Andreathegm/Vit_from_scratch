import os
import torch

def load_weights(model: torch.nn.Module, path: str, device: torch.device, strict: bool = True):
    if not os.path.exists(path):

        raise RuntimeError(f"weights not found at {path}")

    checkpoint = torch.load(path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        raise ValueError("Complex checkpoint passed to this function.It only accepts .pt that only has weights")
    
    model.load_state_dict(checkpoint, strict=strict)

    print("weights loaded succesfully")
    return model

def load_weights_from_complex_checkpoint(model: torch.nn.Module, path: str, device: torch.device, strict: bool = True):
    if not os.path.exists(path):

        raise RuntimeError(f"weights not found at {path}")

    checkpoint = torch.load(path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("weights loaded succesfully")

    else:
        raise ValueError("simple  checkpoint consisting only of weights passed to this function.It only accepts dict-like .pt ")
    
    return model