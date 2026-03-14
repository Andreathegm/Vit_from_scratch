# utils/device.py
import torch

def get_device():
    """Return the available device GPU if present otherwise CPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device