"""
Step 2: Generate embeddings for cleaned small + medium datasets.
Models: Qwen3-0.6B, MiniLM, BERT (mean-pool), E5
Saves embeddings + labels as .pt files under embeddings/
"""
import os
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

CLEAN_DIR = "./../Dataset_cleaned"
EMB_DIR   = "./embeddings"
# SIZES     = ["small", "medium"]
SIZES     = ["small"]
SPLITS    = ["train", "val", "test"]
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BATCH     = 64


# ── Encoding helpers ─────────────────────────────────────────────────────────

def encode_st(model, texts):
    """Encode with SentenceTransformer → numpy (N, dim)."""
    return model.encode(texts, batch_size=BATCH, show_progress_bar=True,
                        convert_to_numpy=True)


def encode_bert(texts, model_name="google-bert/bert-base-uncased"):
    """Mean-pool BERT last hidden state → numpy (N, 768)."""
    print(f"    Loading BERT tokenizer + model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    all_embs = []
    batches = [texts[i : i + BATCH] for i in range(0, len(texts), BATCH)]
    for batch in tqdm(batches, desc="    BERT batches", unit="batch", leave=False):
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=256, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        # Weighted mean over token dim using attention mask
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)


# ── Load models once ─────────────────────────────────────────────────────────

print(f"\nDevice: {DEVICE}")
print("Loading Qwen3-Embedding-0.6B...")
qwen3  = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",  device=DEVICE)
print("Loading all-MiniLM-L6-v2...")
minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
print("Loading intfloat/e5-base...")
e5     = SentenceTransformer("intfloat/e5-base", device=DEVICE)
print("All sentence-transformer models loaded.\n")

MODELS = {
    "qwen3":  lambda texts: encode_st(qwen3,  [f"passage: {t}" for t in texts]),
    "minilm": lambda texts: encode_st(minilm, texts),
    "bert":   lambda texts: encode_bert(texts),
    "e5":     lambda texts: encode_st(e5, [f"passage: {t}" for t in texts]),
}

# ── Generate and save ─────────────────────────────────────────────────────────

total_jobs = len(SIZES) * len(SPLITS) * len(MODELS)
outer_bar  = tqdm(total=total_jobs, desc="Overall progress", unit="job")

for size in SIZES:
    tqdm.write(f"\n{'='*50}\nDataset: {size.upper()}\n{'='*50}")
    for split in SPLITS:
        df     = pd.read_parquet(f"{CLEAN_DIR}/Dataset_{size}/{split}/{split}.parquet")
        texts  = df["text"].tolist()
        labels = df["generated_by"].tolist()

        out_dir = f"{EMB_DIR}/{size}/{split}"
        os.makedirs(out_dir, exist_ok=True)
        torch.save(labels, f"{out_dir}/labels.pt")

        tqdm.write(f"\n[{size}/{split}] {len(texts)} samples — encoding with 4 models")
        for name, encode_fn in MODELS.items():
            outer_bar.set_description(f"{size}/{split} → {name}")
            tqdm.write(f"  → {name}...")
            emb = encode_fn(texts)
            # Pad to nearest multiple of 8 (helps CNN kernel sizing)
            pad = (8 - emb.shape[1] % 8) % 8
            if pad:
                emb = np.pad(emb, ((0, 0), (0, pad)))
            torch.save(torch.tensor(emb), f"{out_dir}/{name}.pt")
            tqdm.write(f"  ✓ {name:8s}: shape={emb.shape}  saved to {out_dir}/{name}.pt")
            outer_bar.update(1)

outer_bar.close()
tqdm.write("\nAll embeddings generated.")
