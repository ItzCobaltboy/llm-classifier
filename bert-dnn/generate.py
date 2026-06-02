"""
Step 2: Generate pooled embeddings for cleaned datasets.
Models: Qwen3-0.6B, MiniLM, BERT (mean-pool), E5
All output (N, dim) vectors — no sequence embeddings stored.
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
SIZES     = ["small", "medium"]
SPLITS    = ["train", "val", "test"]
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BATCH     = 8
MAX_LEN   = 256


# ── Encoding helpers ─────────────────────────────────────────────────────────

def encode_st(model, texts):
    """Encode with SentenceTransformer → numpy (N, dim)."""
    return model.encode(texts, batch_size=BATCH, show_progress_bar=True,
                        convert_to_numpy=True)


def encode_bert_pooled(texts, tokenizer, model):
    """Mean-pool BERT last hidden state → numpy (N, 768)."""
    all_embs = []
    batches = [texts[i : i + BATCH] for i in range(0, len(texts), BATCH)]
    for batch in tqdm(batches, desc="    BERT pooled", unit="batch", leave=False):
        enc = tokenizer(batch, truncation=True, padding="max_length",
                        max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
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

print("Loading google-bert/bert-base-uncased...")
bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert_model     = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(DEVICE).eval()
print("All models loaded.\n")


# ── Generate and save ─────────────────────────────────────────────────────────

total_jobs = len(SIZES) * len(SPLITS) * 4  # qwen3, minilm, e5, bert
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

        tqdm.write(f"\n[{size}/{split}] {len(texts)} samples")

        # ── Qwen3 ───────────────────────────────────────────────────────
        outer_bar.set_description(f"{size}/{split} → qwen3")
        tqdm.write(f"  → qwen3...")
        emb = encode_st(qwen3, [f"passage: {t}" for t in texts])
        pad = (8 - emb.shape[1] % 8) % 8
        if pad:
            emb = np.pad(emb, ((0, 0), (0, pad)))
        torch.save(torch.tensor(emb), f"{out_dir}/qwen3.pt")
        tqdm.write(f"  ✓ qwen3:  {emb.shape}")
        outer_bar.update(1)

        # ── MiniLM ──────────────────────────────────────────────────────
        outer_bar.set_description(f"{size}/{split} → minilm")
        tqdm.write(f"  → minilm...")
        emb = encode_st(minilm, texts)
        pad = (8 - emb.shape[1] % 8) % 8
        if pad:
            emb = np.pad(emb, ((0, 0), (0, pad)))
        torch.save(torch.tensor(emb), f"{out_dir}/minilm.pt")
        tqdm.write(f"  ✓ minilm: {emb.shape}")
        outer_bar.update(1)

        # ── E5 ──────────────────────────────────────────────────────────
        outer_bar.set_description(f"{size}/{split} → e5")
        tqdm.write(f"  → e5...")
        emb = encode_st(e5, [f"passage: {t}" for t in texts])
        pad = (8 - emb.shape[1] % 8) % 8
        if pad:
            emb = np.pad(emb, ((0, 0), (0, pad)))
        torch.save(torch.tensor(emb), f"{out_dir}/e5.pt")
        tqdm.write(f"  ✓ e5:     {emb.shape}")
        outer_bar.update(1)

        # ── BERT pooled ─────────────────────────────────────────────────
        outer_bar.set_description(f"{size}/{split} → bert")
        tqdm.write(f"  → bert (pooled)...")
        emb = encode_bert_pooled(texts, bert_tokenizer, bert_model)
        torch.save(torch.tensor(emb), f"{out_dir}/bert.pt")
        tqdm.write(f"  ✓ bert:   {emb.shape}")
        outer_bar.update(1)

outer_bar.close()
tqdm.write("\nAll embeddings generated.")
