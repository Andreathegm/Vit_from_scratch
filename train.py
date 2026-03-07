def train_one_epoch(model , loader , optimizer, criterion, epoch: int, device) -> float:
    model.train()
    running_loss = 0.0

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # clears old gradients
        logits = model(x)                      # forward pass
        loss = criterion(logits, y)            # compute loss
        loss.backward()                        # backward pass computes gradients
        optimizer.step()                       # update parameters

        running_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"Epoch {epoch} Step {step+1}/{len(loader)} Train Loss {loss.item():.4f}")

    return running_loss / len(loader)