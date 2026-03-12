import torch
from utils import accuracy,accuracy_topk

def train_one_epoch(model , loader , optimizer, criterion, epoch: int, device) -> float:
    model.train()
    running_loss = 0.0

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  
        logits = model(x)                     
        loss = criterion(logits, y)            
        loss.backward()                       
        optimizer.step()                       

        running_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"Epoch {epoch} Step {step+1}/{len(loader)} Train Loss {loss.item():.4f}")

    return running_loss / len(loader)

@torch.inference_mode()
def evaluate(model, loader, criterion,device,split = ""):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        total_acc += accuracy(logits, y)

    avg_loss = total_loss / len(loader)
    avg_acc  = total_acc  / len(loader)

    print(f"[{split}] Loss: {avg_loss:.4f}  Acc: {avg_acc*100:.2f}%")

    return avg_loss, avg_acc

@torch.inference_mode()
def evaluate_top_k(model, loader, criterion,device,split = "",k=5):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_acck = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        total_acc += accuracy(logits, y)
        total_acck += accuracy_topk(logits,y,k)

    avg_loss = total_loss / len(loader)
    avg_acc  = total_acc  / len(loader)
    avg_acc_topk = total_acck / len(loader)

    print(f"[{split}] Loss: {avg_loss:.4f}  Acc1: {avg_acc*100:.2f}%  Acc{k}: {avg_acc_topk*100:.2f}%")

    return avg_loss, avg_acc,avg_acc_topk