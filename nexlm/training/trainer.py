import os
import time
import math
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nexlm.model.gpt import NexLM
from nexlm.config import NexLMConfig
from nexlm.training.lr_schedule import get_lr
from nexlm.training.dataset import TokenizedDataset


def get_default_device() -> torch.device:
    """Auto-detect optimal device (MPS on Apple Silicon, CUDA on NVIDIA, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainingConfig:
    """Hyperparameters and settings for NexLM training."""
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    weight_decay: float = 0.1
    warmup_steps: int = 50
    max_steps: int = 500
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_steps: int = 10
    checkpoint_interval: int = 250
    checkpoint_dir: str = "checkpoints"
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NexLMTrainer:
    """Complete training orchestrator for NexLM models."""

    def __init__(
        self,
        model: NexLM,
        train_dataset: TokenizedDataset,
        val_dataset: Optional[TokenizedDataset] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()

        # Resolve device
        if self.config.device is not None:
            self.device = torch.device(self.config.device)
        else:
            self.device = get_default_device()

        self.model.to(self.device)

        # Setup optimizer
        device_type = "cuda" if "cuda" in self.device.type else ("mps" if "mps" in self.device.type else "cpu")
        self.optimizer = self.model.configure_optimizers(
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            device_type=device_type,
        )

        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        self.step = 0
        self.best_val_loss = float("inf")

    def save_checkpoint(self, path: str, val_loss: Optional[float] = None) -> None:
        """Save model weights, optimizer, and metadata to disk."""
        checkpoint = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config.to_dict(),
            "training_config": self.config.to_dict(),
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Resume training from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint.get("step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed checkpoint from {path} at step {self.step}")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Compute average validation loss and perplexity."""
        if self.val_dataset is None:
            return {}

        self.model.eval()
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=True,
        )

        total_loss = 0.0
        steps = 0

        for x, y in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss, _ = self.model(x, targets=y)
            total_loss += loss.item()
            steps += 1
            if steps >= self.config.eval_steps:
                break

        avg_loss = total_loss / max(1, steps)
        perplexity = math.exp(min(avg_loss, 20.0))  # guard against overflow

        self.model.train()
        return {"val_loss": avg_loss, "val_perplexity": perplexity}

    def train(self) -> None:
        """Main training loop."""
        self.model.train()
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        loader_iter = iter(train_loader)

        print(f"Starting training on device: {self.device}")
        print(f"Model parameters: {self.model.get_num_params():,} non-embedding")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")

        start_time = time.time()
        running_loss = 0.0

        while self.step < self.config.max_steps:
            # Update learning rate according to cosine schedule
            lr = get_lr(
                step=self.step,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.config.max_steps,
                max_lr=self.config.learning_rate,
                min_lr=self.config.min_learning_rate,
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient accumulation steps
            self.optimizer.zero_grad()
            step_loss = 0.0

            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    x, y = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    x, y = next(loader_iter)

                x, y = x.to(self.device), y.to(self.device)
                _, loss, _ = self.model(x, targets=y)
                # Scale loss by gradient accumulation factor
                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()
                step_loss += loss.item()

            # Gradient clipping to stabilize training
            if self.config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()
            self.step += 1
            running_loss += step_loss / self.config.gradient_accumulation_steps

            # Logging & Evaluation
            if self.step % 10 == 0 or self.step == 1:
                elapsed = time.time() - start_time
                tokens_processed = (
                    self.step
                    * self.config.batch_size
                    * self.config.gradient_accumulation_steps
                    * self.model.config.max_seq_len
                )
                tok_per_sec = tokens_processed / max(1e-5, elapsed)
                avg_train_loss = running_loss / (10 if self.step > 1 else 1)
                print(
                    f"Step {self.step:4d}/{self.config.max_steps} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Speed: {tok_per_sec:.0f} tok/s"
                )
                running_loss = 0.0

            # Periodic Evaluation
            if self.step % self.config.eval_interval == 0:
                metrics = self.evaluate()
                if metrics:
                    val_loss = metrics["val_loss"]
                    val_ppl = metrics["val_perplexity"]
                    print(f"  >>> Eval @ Step {self.step}: Val Loss = {val_loss:.4f} | Val PPL = {val_ppl:.2f}")

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
                        self.save_checkpoint(best_path, val_loss=val_loss)

            # Periodic Checkpointing
            if self.step % self.config.checkpoint_interval == 0:
                ckpt_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_step_{self.step}.pt")
                self.save_checkpoint(ckpt_path)

        # Final save
        final_path = os.path.join(self.config.checkpoint_dir, "final_model.pt")
        self.save_checkpoint(final_path)
        print("Training finished successfully!")
