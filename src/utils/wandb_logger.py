import wandb

def init_wandb(config: dict, run_name: str, project: str = "vit-imagenet100"):
    
    wandb.init(
        project=project,
        name=run_name,
        resume="allow",       
        config=config,     
    )
    print(f"wandb initialized → {wandb.run.url}")


def log_epoch(epoch, train_loss, val_loss, val_acc, lr):

    wandb.log({
        "epoch":      epoch,
        "train_loss": train_loss,
        "val_loss":   val_loss,
        "val_acc":    val_acc * 100,
        "lr":         lr,
    }, step=epoch)


def log_best(best_val_acc, best_epoch):

    wandb.run.summary["best_val_acc"] = best_val_acc * 100
    wandb.run.summary["best_epoch"]   = best_epoch


def log_test(test_acc, test_loss):

    wandb.run.summary["test_acc"]  = test_acc * 100
    wandb.run.summary["test_loss"] = test_loss


def finish_wandb():
    
    wandb.finish()