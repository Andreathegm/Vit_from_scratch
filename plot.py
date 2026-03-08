import os 
import matplotlib.pyplot as plt
import pandas as pd
from utils.wandb_fetcher import fetch_multiple_runs,fetch_run_history

def plot_single_run(df,save_path,title):
    """
    Plotta loss, accuracy e lr di un singolo run.
    """
    os.makedirs("plots", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # loss
    axes[0].plot(df["epoch"], df["train_loss"], label="Train", color="steelblue")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val",   color="tomato")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # accuracy
    axes[1].plot(df["epoch"], df["val_acc"], color="seagreen", label="Val Acc")
    axes[1].axhline(y=70, color="gray", linestyle="--", alpha=0.6, label="Target 70%")
    best_epoch = df.loc[df["val_acc"].idxmax(), "epoch"]
    best_acc   = df["val_acc"].max()
    axes[1].axvline(x=best_epoch, color="orange", linestyle="--", alpha=0.6, label=f"Best ep {int(best_epoch)}")
    axes[1].annotate(
        f"{best_acc:.1f}%",
        xy=(best_epoch, best_acc),
        xytext=(best_epoch + 5, best_acc - 2),
        fontsize=9,
        color="orange"
    )
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # learning rate
    axes[2].plot(df["epoch"], df["lr"], color="mediumpurple", label="LR")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    #axes[2].set_yscale("log")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot salvato → {save_path}")


def plot_compare_runs(run_paths: dict, save_path: str = "plots/comparison.png"):
    """
    Confronta val_acc e val_loss di più run sullo stesso grafico.
    Utile per confrontare run1 (ColorJitter) vs run2 (RandAugment puro).

    run_paths: dizionario nome → run_path
    """
    histories = fetch_multiple_runs(run_paths)
    os.makedirs("plots", exist_ok=True)

    colors = ["steelblue", "tomato", "seagreen", "mediumpurple", "orange"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Run Comparison", fontsize=14, fontweight="bold")

    for i, (name, df) in enumerate(histories.items()):
        color = colors[i % len(colors)]

        # val accuracy
        axes[0].plot(df["epoch"], df["val_acc"], label=name, color=color)

        # val loss
        axes[1].plot(df["epoch"], df["val_loss"], label=name, color=color)

    axes[0].axhline(y=70, color="gray", linestyle="--", alpha=0.5, label="Target 70%")
    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Comparison Saved → {save_path}")

def plot_run_history(path, run_name = None):
    if not run_name :
        df = pd.read_csv(path)
    else:
        RUN = "innocentiandrea6-/vit-imagenet100/ik3412d8"
        df = fetch_run_history(RUN, save_csv=True)
    
    plot_single_run(
        df=df,
        save_path="plots/run1.png",
        title="Vit-tiny on imagenet-100"
    )


RUN_NAME = "innocentiandrea6-/vit-imagenet100/ik3412d8"
path = "plots/vit-tiny-200ep_history.csv"
plot_run_history(path)


# # confronto tra run1 e fine-tuning
# plot_compare_runs(
#     run_paths={
#         "Run1 - ColorJitter":     "username/vit-imagenet100/abc123",
#         "Run2 - Finetune":        "username/vit-imagenet100/def456",
#     },
#     save_path="plots/comparison.png"
# )