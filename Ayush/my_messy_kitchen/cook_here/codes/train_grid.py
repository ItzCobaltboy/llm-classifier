"""
train_grid.py — Grid-search training for a configurable transformer model.

Replaces train_roberta.py, train_distilbert.py, and train_deberta.py.

Hyperparameter grid (36 runs per model):
  max_len  : [256, 512, 128]
  lr       : [2e-5, 3e-5, 1e-5]
  batch    : [16, 32]
  lr_mode  : [uniform, differential]

Multi-GPU: DataParallel is enabled automatically when >1 GPU is detected.

Usage:
    python train_grid.py --model roberta-base
    python train_grid.py --model distilbert-base-uncased
    python train_grid.py --model microsoft/deberta-v3-base

Data path is read from the DATA_PATH environment variable (default: ./data).
"""

import argparse
import glob
import os
import subprocess
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import torch

from utils import (
    SEED, TEXT_COL, LABEL_COL, STYLO_DIM,
    set_seed, get_device, wrap_model_multi_gpu, unwrap_model,
    compute_stylo_matrix, make_loader,
    LLMClassifier, build_criterion, build_optimizer,
    train_model, evaluate_model, print_eval_report,
    plot_training_curves, plot_confusion_matrix,
    predict_text, run_shap_explanation, save_artifacts,
)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Grid-search training for a transformer model.")
parser.add_argument(
    "--model",
    default="roberta-base",
    help="HuggingFace model ID (e.g. roberta-base, distilbert-base-uncased).",
)
args = parser.parse_args()

MODEL_NAME = args.model
MODEL_SLUG = MODEL_NAME.split("/")[-1]   # safe filename prefix

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH   = os.getenv("DATA_PATH", "./data")
EPOCHS      = 8
DROPOUT     = 0.3

MAX_LENS    = [256, 512, 128]
LRS         = [2e-5, 3e-5, 1e-5]
BATCH_SIZES = [16, 32]
LR_MODES    = [False, True]   # False = uniform LR,  True = differential LR

# ── Setup ──────────────────────────────────────────────────────────────────────
set_seed(SEED)
DEVICE = get_device()
print(f"Model  : {MODEL_NAME}")
print(f"Device : {DEVICE}  |  GPUs available: {torch.cuda.device_count()}")

# ── Load & prepare data (once, outside the grid loop) ─────────────────────────
print("\n" + "="*60)
print("  Loading data")
print("="*60)

train_df = pd.read_parquet(f"{DATA_PATH}/train.parquet")
val_df   = pd.read_parquet(f"{DATA_PATH}/val.parquet")
test_df  = pd.read_parquet(f"{DATA_PATH}/test.parquet")

for split_df, tag in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
    split_df["split_source"] = tag

combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
df = combined_df[[TEXT_COL, LABEL_COL, "split_source"]].dropna().reset_index(drop=True)
df["text_len"] = df[TEXT_COL].str.len()

print(f"  Total rows after dropping nulls: {len(df)}")
print(f"  Class distribution:\n{df[LABEL_COL].value_counts().to_string()}")

# ── Encode labels ──────────────────────────────────────────────────────────────
le = LabelEncoder()
df["label"] = le.fit_transform(df[LABEL_COL])
NUM_CLASSES = len(le.classes_)
print(f"\n  Classes ({NUM_CLASSES}): {list(le.classes_)}")

# ── Restore splits ─────────────────────────────────────────────────────────────
train_df = df[df["split_source"] == "train"].drop(columns=["split_source"]).copy()
val_df   = df[df["split_source"] == "val"].drop(columns=["split_source"]).copy()
test_df  = df[df["split_source"] == "test"].drop(columns=["split_source"]).copy()
print(f"  Train: {len(train_df):>5}  |  Val: {len(val_df):>5}  |  Test: {len(test_df):>5}")

# ── Stylometric features (computed once, reused across all grid runs) ──────────
print("\n" + "="*60)
print("  Computing stylometric features")
print("="*60)

train_stylo_raw = compute_stylo_matrix(train_df[TEXT_COL].tolist())
val_stylo_raw   = compute_stylo_matrix(val_df[TEXT_COL].tolist())
test_stylo_raw  = compute_stylo_matrix(test_df[TEXT_COL].tolist())

scaler      = StandardScaler()
train_stylo = scaler.fit_transform(train_stylo_raw).astype(np.float32)
val_stylo   = scaler.transform(val_stylo_raw).astype(np.float32)
test_stylo  = scaler.transform(test_stylo_raw).astype(np.float32)
print(f"  Stylometric matrix shape: {train_stylo.shape}")

# ── Tokenizer ──────────────────────────────────────────────────────────────────
print(f"\n  Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"  Vocabulary size: {tokenizer.vocab_size:,}")


# ══════════════════════════════════════════════════════════════════════════════
#  Grid-search loop
# ══════════════════════════════════════════════════════════════════════════════
run_index = 0

for max_len in MAX_LENS:
    for lr in LRS:
        for batch_size in BATCH_SIZES:
            for use_differential_lr in LR_MODES:

                lr_tag   = "diff_lr" if use_differential_lr else "uniform_lr"
                run_name = f"{MODEL_SLUG}__maxlen{max_len}__lr{lr}__bs{batch_size}__{lr_tag}"
                checkpoint_path = f"model_{run_name}.pt"

                print("\n" + "="*60)
                print(f"  RUN {run_index + 1}/36 — {run_name}")
                print("="*60)

                # ── DataLoaders ────────────────────────────────────────────
                train_loader = make_loader(train_df, train_stylo, tokenizer, max_len, batch_size, shuffle=True)
                val_loader   = make_loader(val_df,   val_stylo,   tokenizer, max_len, batch_size)
                test_loader  = make_loader(test_df,  test_stylo,  tokenizer, max_len, batch_size)

                # ── Model ──────────────────────────────────────────────────
                model = LLMClassifier(MODEL_NAME, STYLO_DIM, NUM_CLASSES, DROPOUT).to(DEVICE)
                model = wrap_model_multi_gpu(model)

                total_params = sum(p.numel() for p in model.parameters())
                print(f"  Total params: {total_params:,}")

                # ── Loss, optimizer, scheduler ─────────────────────────────
                criterion = build_criterion(train_df["label"].values, NUM_CLASSES, DEVICE)
                optimizer = build_optimizer(model, lr, use_differential_lr)

                total_steps  = len(train_loader) * EPOCHS
                warmup_steps = int(0.1 * total_steps)
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps   = warmup_steps,
                    num_training_steps = total_steps,
                )
                print(f"  Training steps: {total_steps}  |  Warmup: {warmup_steps}")

                # ── Training ───────────────────────────────────────────────
                history = train_model(
                    model, train_loader, val_loader,
                    criterion, optimizer, scheduler,
                    EPOCHS, checkpoint_path, DEVICE,
                )

                plot_training_curves(history, run_name)

                # ── Evaluation ─────────────────────────────────────────────
                unwrap_model(model).load_state_dict(
                    torch.load(checkpoint_path, map_location=DEVICE)
                )
                all_labels, all_preds = evaluate_model(model, test_loader, DEVICE)
                print_eval_report(all_labels, all_preds, le.classes_)
                plot_confusion_matrix(all_labels, all_preds, list(le.classes_), run_name)

                # ── Demo prediction ────────────────────────────────────────
                sample = (
                    "Hi, my name is Ayush Yadav. I am a human. "
                    "Let's see if the model can correctly predict me as a human."
                )
                result = predict_text(sample, model, tokenizer, scaler, le, max_len, DEVICE)
                print(f"  Demo prediction → {result['prediction']}  ({result['confidence']})")
                print(f"  All probs: {result['all_probs']}")

                # ── SHAP ───────────────────────────────────────────────────
                run_shap_explanation(
                    model, tokenizer, scaler, train_df, test_df,
                    max_len, batch_size, DEVICE,
                )

                # ── Save artifacts ─────────────────────────────────────────
                save_artifacts(le, scaler, run_name)

                print(f"\n  ✔ Run {run_name} complete.\n")
                run_index += 1

# ── Zip all outputs ────────────────────────────────────────────────────────────
zip_name = f"{MODEL_SLUG}_outputs.zip"
output_files = glob.glob(f"*{MODEL_SLUG}*")
if output_files:
    subprocess.run(["zip", "-j", zip_name] + output_files)
    print(f"\nAll runs complete. Outputs zipped → {zip_name}")
