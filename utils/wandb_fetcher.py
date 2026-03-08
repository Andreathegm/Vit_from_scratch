import wandb
import os


def fetch_run_history(run_path: str, save_csv: bool = True):
    """
    Scarica la history completa di un run da wandb.

    run_path: "username/project/run_id"
              lo trovi su wandb.ai → run → Overview → Run path
    """
    api = wandb.Api()
    run = api.run(run_path)

    df = run.history()
    df = df.dropna(subset=["train_loss", "val_loss", "val_acc"])
    df = df.sort_values("epoch").reset_index(drop=True)

    if save_csv:
        os.makedirs("plots", exist_ok=True)
        csv_path = f"plots/{run.name}_history.csv"
        df.to_csv(csv_path, index=False)
        print(f"History saved → {csv_path}")

    return df


def fetch_multiple_runs(run_paths) -> dict:

    histories = {}
    for name, path in run_paths.items():
        print(f"Download {name}...")
        histories[name] = fetch_run_history(path, save_csv=False)
    return histories


