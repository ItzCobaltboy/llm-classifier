"""
Step 3: Train DNN on all 4 pooled embeddings.
Combinations: 4 embeddings × N sizes = runs
BERT RNN training is in train_bert.py (separate pipeline).
"""
import torch
from tqdm import tqdm
from classifier_models import DNN, Classifier

EMB_DIR     = "./embeddings"
OUTPUT_DIR  = "./output"
SIZES       = ["medium"]
# EMBEDDINGS  = ["qwen3", "minilm", "bert", "e5"]
EMBEDDINGS  = ["bert"]
NUM_CLASSES = 12
EPOCHS      = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}\n")


def load(size, split, emb):
    base = f"{EMB_DIR}/{size}/{split}"
    X = torch.load(f"{base}/{emb}.pt", weights_only=True)
    y = torch.load(f"{base}/labels.pt", weights_only=False)
    return X, y


combos = [(s, e) for s in SIZES for e in EMBEDDINGS]
total  = len(combos)

print(f"Starting training: {total} DNN runs")
print(f"Sizes: {SIZES}  |  Embeddings: {EMBEDDINGS}\n")

outer_bar = tqdm(combos, desc="Runs", unit="run")
for run_idx, (size, emb) in enumerate(outer_bar, 1):
    full_name = f"{size}_{emb}_dnn"
    outer_bar.set_description(f"[{run_idx}/{total}] {full_name}")

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"  Run {run_idx}/{total}:  {full_name}")
    tqdm.write(f"{'='*60}")

    X_train, y_train = load(size, "train", emb)
    X_val,   y_val   = load(size, "val",   emb)
    X_test,  y_test  = load(size, "test",  emb)
    tqdm.write(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    model   = DNN(input_dim=X_train.shape[1], num_classes=NUM_CLASSES)
    out_dir = f"{OUTPUT_DIR}/{full_name}"
    clf     = Classifier(model, output_dir=out_dir, epochs=EPOCHS, ckpt_every=10)

    clf.fit(X_train, y_train, X_val, y_val)

    tqdm.write("\n--- Test Set Report ---")
    test_stats = clf.evaluate(X_test, y_test)
    clf.save(test_stats)

    outer_bar.update()

tqdm.write(f"\nAll {total} DNN runs complete. Results in {OUTPUT_DIR}/")
