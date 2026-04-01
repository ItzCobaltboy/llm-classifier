"""
Step 3: Train all embedding × model combinations and print results.
Combinations: 4 embeddings × 2 models × 2 dataset sizes = 16 runs
No checkpoints saved — all results printed to stdout.
"""
import torch
from itertools import product
from tqdm import tqdm
from models import DNN, CNN, Classifier

EMB_DIR    = "./embeddings"
SIZES      = ["small", "medium"]
EMBEDDINGS = ["qwen3", "minilm", "bert", "e5"]
NUM_CLASSES = 12
EPOCHS      = 10


def load(size, split, emb):
    base = f"{EMB_DIR}/{size}/{split}"
    X = torch.load(f"{base}/{emb}.pt", weights_only=True)
    y = torch.load(f"{base}/labels.pt", weights_only=False)
    return X, y


ARCH = [("DNN", DNN), ("CNN", CNN)]
combos = list(product(SIZES, EMBEDDINGS, ARCH))
total_runs = len(combos)

print(f"Starting training: {total_runs} combinations total")
print(f"Sizes: {SIZES}  |  Embeddings: {EMBEDDINGS}  |  Models: DNN, CNN\n")

outer_bar = tqdm(combos, desc="Combinations", unit="run")
for run_idx, (size, emb, (model_name, ModelClass)) in enumerate(outer_bar, 1):
    outer_bar.set_description(f"[{run_idx}/{total_runs}] {size}|{emb}|{model_name}")

    tqdm.write(f"\nLoading {emb} embeddings for {size}...")
    X_train, y_train = load(size, "train", emb)
    X_val,   y_val   = load(size, "val",   emb)
    X_test,  y_test  = load(size, "test",  emb)
    tqdm.write(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    input_dim = X_train.shape[1]

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"  Run {run_idx}/{total_runs}:  {size.upper()} | {emb} | {model_name}")
    tqdm.write(f"  input_dim={input_dim}  train_samples={len(y_train)}")
    tqdm.write(f"{'='*60}")

    model = ModelClass(input_dim=input_dim, num_classes=NUM_CLASSES)
    clf   = Classifier(model, epochs=EPOCHS)

    clf.fit(X_train, y_train, X_val, y_val)

    tqdm.write("\n--- Test Set Report ---")
    clf.evaluate(X_test, y_test)

tqdm.write(f"\nAll {total_runs} runs complete.")
