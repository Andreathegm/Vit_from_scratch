import torch
def get_default_optimizers(model,optimizer:str):
    match(optimizer.name):
        case "AdamW":
            torch.optim.AdamW(model.parameters(), weight_decay=optimizer.weight_decay)
