"""
PyTorch classifier models for LLM-generated text classification.
  - DNN : fully-connected dense network
  - CNN : 1D convolution over the embedding treated as a signal
Both expose  .fit(X_train, y_train, X_val, y_val)
         and .evaluate(X_test, y_test)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Architectures ─────────────────────────────────────────────────────────────

class DNN(nn.Module):
    """Feed-forward network: input_dim → [512 → 256 → 128] → num_classes."""
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


class CNN(nn.Module):
    """
    1D CNN treating the embedding vector as a 1-channel signal.
    Conv1d with padding=1 keeps dimensions stable; AdaptiveAvgPool
    collapses to a fixed size regardless of input_dim.
    """
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            # (B, 1, input_dim)
            nn.Conv1d(1,  32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),   # → (B, 128, 64) regardless of input_dim
        )
        self.fc = nn.Sequential(
            nn.Flatten(),               # → (B, 128*64 = 8192)
            nn.Linear(128 * 64, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)             # add channel dim: (B, 1, input_dim)
        return self.fc(self.conv(x))


# ── Classifier wrapper ────────────────────────────────────────────────────────

class Classifier:
    """
    Wraps a DNN or CNN with training and evaluation logic.
    Usage:
        clf = Classifier(DNN(input_dim, num_classes))
        clf.fit(X_train, y_train, X_val, y_val)
        clf.evaluate(X_test, y_test)
    """
    def __init__(self, model: nn.Module, lr: float = 1e-3,
                 batch_size: int = 64, epochs: int = 10):
        self.model      = model.to(DEVICE)
        self.optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion  = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.epochs     = epochs
        self.label2id: dict = {}
        self.id2label:  dict = {}

    def _encode(self, labels):
        """Convert string labels → LongTensor, building vocab on first call."""
        if not self.label2id:
            unique = sorted(set(labels))
            self.label2id = {l: i for i, l in enumerate(unique)}
            self.id2label = {i: l for l, i in self.label2id.items()}
        return torch.tensor([self.label2id[l] for l in labels], dtype=torch.long)

    def _loader(self, X, y, shuffle: bool = True):
        ds = TensorDataset(X.float().to(DEVICE), y.to(DEVICE))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _accuracy(self, X, y) -> float:
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X.float().to(DEVICE)).argmax(1).cpu()
        return (preds == y).float().mean().item()

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

            val_acc = self._accuracy(X_val, y_val)
            train_acc = correct / len(y_train)
            epoch_bar.set_postfix(
                loss=f"{total_loss/len(y_train):.4f}",
                train_acc=f"{train_acc:.3f}",
                val_acc=f"{val_acc:.3f}",
            )

        print("  Training complete.")

    def evaluate(self, X_test, y_test_raw):
        print("  Running evaluation on test set...")
        y_test = self._encode(y_test_raw)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_test.float().to(DEVICE)).argmax(1).cpu()
        names = [self.id2label[i] for i in range(len(self.id2label))]
        print(classification_report(y_test.cpu(), preds, target_names=names, digits=3))
