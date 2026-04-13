"""
PyTorch classifier models for LLM-generated text classification.
  - DNN : fully-connected dense network (for pooled embeddings)
  - RNN : bidirectional GRU (for BERT token-level sequences)
Classifier wrapper handles training, checkpointing, evaluation, and saving.
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── DNN ──────────────────────────────────────────────────────────────────────

class DNN(nn.Module):
    """Feed-forward: input_dim → [512 → 256 → 128] → num_classes."""
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── RNN ──────────────────────────────────────────────────────────────────────

class RNN(nn.Module):
    """
    Bidirectional GRU over token sequences.
    Input:  (B, seq_len, 768)
    Output: (B, num_classes)
    """
    def __init__(self, input_dim: int = 768, num_classes: int = 12,
                 hidden: int = 256, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0,
        )
        # Bidirectional → hidden * 2
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, seq_len, 768)
        out, _ = self.gru(x)       # (B, seq_len, hidden*2)
        out = out[:, -1, :]        # take last timestep: (B, hidden*2)
        return self.fc(out)


# ── Classifier wrapper ───────────────────────────────────────────────────────

class Classifier:
    """
    Wraps DNN or RNN with training loop, checkpointing, evaluation, and saving.

    Args:
        model:      DNN or RNN instance
        output_dir: path to save checkpoints, models, and stats
        lr, batch_size, epochs: training hyperparameters
        ckpt_every: save checkpoint every N epochs (overwrites same file)
    """
    def __init__(self, model: nn.Module, output_dir: str,
                 lr: float = 1e-3, batch_size: int = 64,
                 epochs: int = 50, ckpt_every: int = 10):
        self.model      = model.to(DEVICE)
        self.output_dir = output_dir
        self.optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion  = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.epochs     = epochs
        self.ckpt_every = ckpt_every
        self.label2id:  dict = {}
        self.id2label:  dict = {}
        self.best_val_acc = 0.0
        self.history: list   = []  # per-epoch stats

        # Create output dirs
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

    # ── Label encoding ───────────────────────────────────────────────────

    def _encode(self, labels):
        """Build label vocab on first call, then convert to LongTensor."""
        if not self.label2id:
            unique = sorted(set(labels))
            self.label2id = {l: i for i, l in enumerate(unique)}
            self.id2label = {i: l for l, i in self.label2id.items()}
        return torch.tensor([self.label2id[l] for l in labels], dtype=torch.long)

    # ── DataLoader ───────────────────────────────────────────────────────

    def _loader(self, X, y, shuffle=True):
        ds = TensorDataset(X.float().to(DEVICE), y.to(DEVICE))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    # ── Accuracy helper ──────────────────────────────────────────────────

    def _accuracy(self, X, y) -> float:
        self.model.eval()
        preds_all = []
        # Evaluate in batches to avoid OOM
        loader = DataLoader(TensorDataset(X.float(), y),
                            batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for xb, yb in loader:
                preds_all.append(self.model(xb.to(DEVICE)).argmax(1).cpu())
        preds = torch.cat(preds_all)
        return (preds == y).float().mean().item()

    # ── Checkpoint save/load ─────────────────────────────────────────────

    def _save_checkpoint(self, epoch):
        """Overwrite single checkpoint file with latest state."""
        path = f"{self.output_dir}/checkpoints/checkpoint.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "label2id": self.label2id,
            "id2label": self.id2label,
        }, path)
        tqdm.write(f"    [checkpoint saved: epoch {epoch}]")

    def _save_best(self):
        """Save current model as best model."""
        path = f"{self.output_dir}/best_model.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_val_acc": self.best_val_acc,
            "label2id": self.label2id,
            "id2label": self.id2label,
        }, path)

    # ── Training ─────────────────────────────────────────────────────────

    def fit(self, X_train, y_train_raw, X_val, y_val_raw):
        y_train = self._encode(y_train_raw)
        y_val   = self._encode(y_val_raw)
        loader  = self._loader(X_train, y_train)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Trainable params: {n_params:,}  |  batches/epoch: {len(loader)}  |  device: {DEVICE}")

        epoch_bar = tqdm(range(1, self.epochs + 1), desc="  Epochs", unit="ep")
        for epoch in epoch_bar:
            self.model.train()
            total_loss, correct = 0.0, 0

            batch_bar = tqdm(loader, desc=f"    ep{epoch:02d}", unit="batch", leave=False)
            for xb, yb in batch_bar:
                self.optimizer.zero_grad()
                out  = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(xb)
                correct    += (out.argmax(1) == yb).sum().item()
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_acc = correct / len(y_train)
            val_acc   = self._accuracy(X_val, y_val)

            # Track history
            self.history.append({
                "epoch": epoch,
                "loss": round(total_loss / len(y_train), 4),
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
            })

            epoch_bar.set_postfix(
                loss=f"{total_loss/len(y_train):.4f}",
                train_acc=f"{train_acc:.3f}",
                val_acc=f"{val_acc:.3f}",
            )

            # Update best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_best()

            # Checkpoint every N epochs (overwrite)
            if epoch % self.ckpt_every == 0:
                self._save_checkpoint(epoch)

        print("  Training complete.")

    # ── Evaluation ───────────────────────────────────────────────────────

    def evaluate(self, X_test, y_test_raw) -> dict:
        """Evaluate on test set, print report, return stats dict."""
        print("  Running evaluation on test set...")
        y_test = self._encode(y_test_raw)
        self.model.eval()

        preds_all = []
        loader = DataLoader(TensorDataset(X_test.float(), y_test),
                            batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for xb, yb in loader:
                preds_all.append(self.model(xb.to(DEVICE)).argmax(1).cpu())
        preds = torch.cat(preds_all)

        names  = [self.id2label[i] for i in range(len(self.id2label))]
        report = classification_report(y_test.cpu(), preds, target_names=names,
                                       digits=3, output_dict=True)
        report_str = classification_report(y_test.cpu(), preds, target_names=names, digits=3)
        print(report_str)

        return {
            "test_accuracy": round(report["accuracy"], 4),
            "per_class": {k: {m: round(v, 4) for m, v in report[k].items()}
                          for k in names},
            "report_text": report_str,
        }

    # ── Save everything ──────────────────────────────────────────────────

    def save(self, test_stats: dict):
        """Save latest model, label mappings, training history, and test stats."""
        # Latest model
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "label2id": self.label2id,
            "id2label": self.id2label,
        }, f"{self.output_dir}/latest_model.pt")

        # Stats JSON
        stats = {
            "best_val_acc": round(self.best_val_acc, 4),
            "test_stats": test_stats,
            "training_history": self.history,
            "label2id": self.label2id,
        }
        with open(f"{self.output_dir}/stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"  Saved: latest_model.pt, best_model.pt, stats.json → {self.output_dir}/")
