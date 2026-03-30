import os 
import matplotlib.pyplot as plt
import pandas as pd
from src.utils.wandb_fetcher import fetch_multiple_runs,fetch_run_history
from src.utils.csv_manager import list_from_csv
import numpy as np

plt.rcParams.update({
    "font.size": 16,        # font generale
    "axes.titlesize": 18,   # titolo assi
    "axes.labelsize": 16,   # label assi
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

def plot_single_run(df,save_path,title):
    """
    Plot loss, accuracy e lr of a single run.
    """
    os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)


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

    histories = fetch_multiple_runs(run_paths)
    os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)

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


def plot_run_history(save_path,title,run_name = None,csv_path=None):
    if csv_path :
        df = pd.read_csv(save_path)
    else:
        RUN = run_name
        df = fetch_run_history(RUN, save_csv=True)
    
    plot_single_run(
        df=df,
        save_path=save_path,
        title=title
    )


def plot_training_trend(file,title):
    rows = list_from_csv(file)

    x_labels = [row[0] for row in rows]

    numeric_cols = list(zip(*[list(map(float, row[1:])) for row in rows]))

    plt.figure(figsize=(10, 6))

    for i, col in enumerate(numeric_cols):
        plt.plot(x_labels, col, marker="o", label=f"Top {4*i+1}")

    plt.title(title)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/trend.png")

import matplotlib.pyplot as plt

def plot_class_accuracy(class_acc,save_path, title="Accuracy per class",ylabel="Accuracy"):
    plt.figure(figsize=(10,5))
    plt.bar(range(len(class_acc)), class_acc)
    plt.xlabel("Class index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)


# paths=["class_accuracy.npy","class_accuracy_topk.npy"]
# save_paths = ["cls_acc.png","cls_acc_topk"]
# titles = ["Top1 Accuracy","Top5 Accuracy"]
# accuracys = []
# for path,save_path,title in zip(paths,save_paths,titles):
#     accuracy = np.load(path)
#     accuracys.append(accuracy)
#     plot_class_accuracy(class_acc=accuracy,save_path=save_path,title=title)
# diff = accuracys[1] - accuracys[0]
# sum = accuracys[1] + accuracys[0] 
# plot_class_accuracy(class_acc=diff,save_path="diff.png",title="",ylabel="Accuracy differnce Top5 - Top1")
# plot_class_accuracy(class_acc=sum,save_path="sum.png",title="",ylabel="Accuracy sum Top5 + Top1")


plot_training_trend("csv_results/model_performance.csv","Test Set Accuracy during training")


# p = "innocentiandrea6-/vit-imagenet100/"
# vit_pretrain0 = "ik3412d8"
# vit_finetune1 = "8staqny1"
# vit_finetune_mix_up2 = "w4t1m77m"
# vit_fine_tune_mix_up3 = "onoy1old"
# vit_fine_tune_early_stopping4 = "96ep10zp"

# run_ids = [
#            vit_pretrain0,
#            vit_finetune1,
#            vit_finetune_mix_up2,
#            vit_fine_tune_mix_up3,
#            vit_fine_tune_early_stopping4
#            ]

# titles = ["Vit Pretrain",
#           "Vit fine Tune",
#           "Vit fine Tune + MixUp & CutMix",
#           "Vit fine Tune + MixUp & CutMix second try",
#           "Vit fine Tune , early stopping"]
# titles_s = ["".join(title.split(" ")) for title in titles]
# print(titles_s)

# RUNS_NAMES = [p + run_id for run_id in run_ids]

# for i,run_name in enumerate(RUNS_NAMES):
#     plot_run_history(save_path="plots/"+ titles[i] +"/" + run_ids[i] ,run_name= run_name,title=titles[i])

