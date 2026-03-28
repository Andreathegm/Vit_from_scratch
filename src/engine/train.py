import torch
from torch.amp import autocast
from src.utils.metrics import accuracy,accuracy_topk

def train_one_epoch(model , loader , optimizer, criterion, epoch: int, device,scaler = None) -> float:
    model.train()
    running_loss = 0.0

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast(device_type="cuda"):
                logits = model(x)
                loss = criterion(logits,y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(x)                     
            loss = criterion(logits, y)            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()                       

        running_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"Epoch {epoch} Step {step+1}/{len(loader)} Train Loss {loss.item():.4f}")

    return running_loss / len(loader)

@torch.inference_mode()
def evaluate(model, loader, criterion,device,split = "",k = 1):
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

@torch.inference_mode()
def evaluate_top_k_per_class(model, loader, criterion, device, split="", k=5):
    num_classes = 100
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_acck = 0.0

    # --- per classe ---
    correct_per_class = torch.zeros(num_classes, device=device)
    total_per_class   = torch.zeros(num_classes, device=device)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        total_acc  += accuracy(logits, y)
        total_acck += accuracy_topk(logits, y, k)

        preds = logits.argmax(dim=1)
        for cls in range(num_classes):
            mask = (y == cls)
            total_per_class[cls] += mask.sum()
            correct_per_class[cls] += (preds[mask] == cls).sum()

    avg_loss = total_loss / len(loader)
    avg_acc  = total_acc  / len(loader)
    avg_acc_topk = total_acck / len(loader)

    class_acc = (correct_per_class / total_per_class).cpu().numpy()

    print(f"[{split}] Loss: {avg_loss:.4f}  Acc1: {avg_acc*100:.2f}%  Acc{k}: {avg_acc_topk*100:.2f}%")

    return avg_loss, avg_acc, avg_acc_topk, class_acc



def get_default_evaluation_action(evaluation_action):
    match(evaluation_action):
        case 1: 
            return evaluate
        case 2:
            return evaluate_top_k
        case _:
            return evaluate_top_k_per_class