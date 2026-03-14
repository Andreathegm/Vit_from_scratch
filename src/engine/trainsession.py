import os
import torch
from tqdm import tqdm
from time import time

from .train import train_one_epoch, evaluate
from  src.utils.wandb_logger import log_epoch, log_best, log_test, init_wandb, finish_wandb


class TrainSession:
    """
    Encapsulates a training session for a PyTorch model.
    Handles training loop, checkpointing, and experiment tracking via wandb.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        config,
        run_name,
        epochs,        
        device,
        weights_path
    ):
        self.model           = model
        self.optimizer       = optimizer
        self.scheduler       = scheduler
        self.criterion       = criterion
        self.config          = config
        self.run_name        = run_name
        self.epochs          = epochs
        self.device          = device
        self.checkpoint_last = f"checkpoints/{self.run_name}/last.pt"
        self.checkpoint_best = f"checkpoints/{self.run_name}/best.pt"

        if weights_path is not None:
            self.load_weights(weights_path)
            self.weights_path = weights_path
        else:
            self.weights_path = None
    
    def __str__(self) -> str:
        """Returns a human-readable summary of the training session."""
        
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        current_lr   = self.optimizer.param_groups[0]["lr"]
        weight_decay = self.optimizer.param_groups[0]["weight_decay"]
        if self.weights_path is not None :
            w_path = self.weights_path
        else:
            w_path = "-"
        return (
            f"TrainSession\n"
            f"{'─' * 40}\n"
            f"{self.config.to_dict()}"
            f"  run name:        {self.run_name}\n"
            f"  epochs:          {self.epochs}\n"
            f"  device:          {self.device}\n"
            f"\n"
            f"  model:           {self.model.__class__.__name__}\n"
            f"  parameters:      {total_params:.2f}M\n"
            f"\n"
            f"  optimizer:       {self.optimizer.__class__.__name__}\n"
            f"      lr:              {current_lr:.2e}\n"
            f"      weight decay:    {weight_decay}\n"
            f"\n"
            f"  scheduler:       {self.scheduler.__class__.__name__}\n"
            f"  criterion:       {self.criterion.__class__.__name__}\n"
            f"\n"
            f"  When training last checkpoint will be in : {self.checkpoint_last}\n"
            f"  When training best checkpoint will be in : {self.checkpoint_best}\n"
            f"  weights loaded from {w_path}"
        )


        

    # ──────────────────────────────────────────
    # Checkpoint
    # ──────────────────────────────────────────

    def save_checkpoint(self, epoch, train_loss, val_loss, val_acc, best_val_acc, path):
        """Saves model, optimizer, scheduler state and training metrics to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch":                epoch,
            "train_loss":           train_loss,
            "val_loss":             val_loss,
            "val_acc":              val_acc,
            "best_val_acc":         best_val_acc,
        }, path)
        print(f"Checkpoint saved → {path} (epoch {epoch}, val_acc {val_acc*100:.2f}%)")

    def load_checkpoint(self, path, weights_only: bool = False):
        """
        Loads a checkpoint from disk.

        Args:
            path:         path to the checkpoint file
            weights_only: if True  → loads only model weights
                          if False → loads model + optimizer + scheduler
        Returns:
            start_epoch:  epoch to resume from
            best_val_acc: best validation accuracy seen so far
        """
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return 0, 0.0

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if not weights_only:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch        = checkpoint["epoch"]
        best_val_acc = checkpoint["best_val_acc"]

        print(
            f"Loaded checkpoint from {path} "
            f"{'(weights only)' if weights_only else '(full state)'}\n"
            f"  Last epoch saved:  {epoch}\n"
            f"  Best val accuracy: {best_val_acc*100:.2f}%"
        )
        return epoch + 1, best_val_acc

    def load_weights(self, path: str):
        """
        Loads only model weights from any checkpoint file.
        Used for external weights not necessarily from last/best checkpoints.
        """
        checkpoint = torch.load(path, map_location=self.device)
        # supporta sia checkpoint completi che semplici state_dict
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("no weights loaded")
        print(f"Weights loaded from {path}")

    def load_optimizer_state(self, path: str, new_lr: float = None,no_wd_load=False):
        """
        Loads optimizer state dict from a checkpoint — preserves m and v moments.
        Optionally overrides lr after loading, keeping the accumulated moments.

        Args:
            path:   path to the checkpoint file
            new_lr: if provided, overrides lr after loading state dict
                    moments m and v are preserved — only step size changes
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if new_lr is not None:
            for group in self.optimizer.param_groups:
                old_lr = group["lr"]
                group["lr"] = new_lr
                if no_wd_load is True :
                    group["weight_decay"] = self.config.training.weight_decay
            print(f"Optimizer state loaded — old_lr = {old_lr} overridden to {new_lr}")
        else:
            print("Optimizer state loaded — lr unchanged")

    def load_scheduler_state(self, path: str):
        """
        Loads scheduler state dict from a checkpoint.
        Restores last_epoch so the scheduler knows where it is in the curve.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Scheduler state loaded from {path}")

    # ──────────────────────────────────────────
    # Core training loop
    # ──────────────────────────────────────────

    def _run_loop(self, train_loader, val_loader, start_epoch, best_val_acc):
        """
        Internal training loop.
        Not meant to be called directly — use train(), resume() instead.
        """
        start      = time()
        epoch_pbar = tqdm(range(start_epoch, self.epochs + 1), desc="Training")

        for epoch in epoch_pbar:
            train_loss = train_one_epoch(
                self.model, train_loader, self.optimizer,
                self.criterion, epoch, self.device
            )
            val_loss, val_acc = evaluate(
                self.model, val_loader, self.criterion,
                self.device, split="Val"
            )
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            log_epoch(epoch, train_loss, val_loss, val_acc, current_lr)

            self.save_checkpoint(
                epoch, train_loss, val_loss, val_acc,
                best_val_acc, self.checkpoint_last
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(
                    epoch, train_loss, val_loss, val_acc,
                    best_val_acc, self.checkpoint_best
                )
                log_best(best_val_acc, epoch)

            epoch_pbar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                acc=f"{val_acc*100:.2f}%",
                best=f"{best_val_acc*100:.2f}%",
                lr=f"{current_lr:.2e}"
            )

        elapsed = (time() - start) / 60
        print(f"\nTraining completed in {elapsed:.1f} minutes")

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def train(self, train_loader, val_loader):
        """Starts a fresh training session from epoch 1."""
        init_wandb(self.config, run_name=self.run_name)
        self._run_loop(train_loader, val_loader, start_epoch=1, best_val_acc=0.0)
        finish_wandb()

    def resume(self, train_loader, val_loader, weights_only: bool = False):
        """
        Resumes training from the last saved checkpoint.

        Args:
            weights_only: if False → full resume including optimizer and scheduler
                          if True  → loads only model weights,
                                     use when optimizer/scheduler were set up externally
        """
        init_wandb(self.config, run_name=self.run_name)
        start_epoch, best_val_acc = self.load_checkpoint(
            self.checkpoint_last, weights_only=weights_only
        )
        self._run_loop(train_loader, val_loader, start_epoch, best_val_acc)
        finish_wandb()

    def train_and_test(self, train_loader, val_loader, test_loader):
        """Runs full training then evaluates on the test set."""
        init_wandb(self.config, run_name=self.run_name)
        self._run_loop(train_loader, val_loader, start_epoch=1, best_val_acc=0.0)
        self._test(test_loader)
        finish_wandb()

    def resume_and_test(self, train_loader, val_loader, test_loader, weights_only: bool = False):
        """Resumes training then evaluates on the test set."""
        init_wandb(self.config, run_name=self.run_name)
        start_epoch, best_val_acc = self.load_checkpoint(
            self.checkpoint_last, weights_only=weights_only
        )
        self._run_loop(train_loader, val_loader, start_epoch, best_val_acc)
        self._test(test_loader)
        finish_wandb()

    def test(self, test_loader):
        """Evaluates the best saved model on the test set."""
        init_wandb(self.config, run_name=self.run_name)
        self._test(test_loader)
        finish_wandb()

    def _test(self, test_loader):
        """Internal test evaluation — loads best checkpoint and computes metrics."""
        print("Loading best model for final test evaluation...")
        self.load_weights(self.checkpoint_best)

        test_loss, test_acc = evaluate(
            self.model, test_loader, self.criterion,
            self.device, split="Test"
        )
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc:  {test_acc*100:.2f}%")
        log_test(test_acc, test_loss)

    def test_checkpoint(self,test_loader):
        print("Testing preloaded weights on test set...")

        test_loss, test_acc = evaluate(
            self.model, test_loader, self.criterion,
            self.device, split="Test"
        )
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc:  {test_acc*100:.2f}%")