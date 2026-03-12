def accuracy_topk(logits, targets,k) -> float:
    if targets.dim() == 2:
        targets = targets.argmax(dim=1)
    top5 = logits.topk(k=k, dim=1).indices
    return (top5 == targets.view(-1, 1)).any(dim=1).float().mean().item()


def accuracy(logits, targets) -> float:
    preds = logits.argmax(dim=1)

    # targets can have (B,) shape or (B, num_classes)
    if targets.dim() == 2:
        targets = targets.argmax(dim=1)

    return (preds == targets).float().mean().item()