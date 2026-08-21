"""
Dhwani (ध्वनि) - MSMARCO-XI 5-Way Hybrid Vector Index Builder
Generates atomic proposition micro-units, hierarchical parent-child spans,
semantic sentence boundaries, intent-conditioned chunks, and cross-lingual aligned units.
"""

import os
import sys
import argparse
import requests
import lancedb
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Optimize PyTorch CPU threading
try:
    torch.set_num_threads(os.cpu_count() or 8)
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.core.chunking import process_record_5way_chunking

parquet_filename = os.getenv("PARQUET_FILE", os.path.join(PROJECT_ROOT, "hinval.parquet"))
db_path = os.getenv("LANCEDB_PATH", os.path.join(PROJECT_ROOT, "lancedb_msmarco"))
url = os.getenv(
    "DATASET_URL",
    "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"
)

def download_dataset():
    print("--> Downloading MSMARCO-XI validation dataset (~462MB)...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024

    with open(parquet_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    done_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"\r--> Progress: {done_mb:.1f}MB / {total_mb:.1f}MB ({downloaded * 100 / total_size:.1f}%)", end="", flush=True)
    print("\n--> Download complete!")

def verify_parquet():
    if os.path.exists(parquet_filename):
        try:
            _ = pd.read_parquet(parquet_filename, engine="pyarrow", columns=["query_id"]).head(1)
            print(f"--> Using verified local file: {parquet_filename}")
            return True
        except Exception as e:
            print(f"--> Local dataset invalid ({e}). Redownloading...")
            try:
                os.remove(parquet_filename)
            except OSError:
                pass
    return False

def main():
    parser = argparse.ArgumentParser(description="Dhwani 5-Way Hybrid Vector Index Builder")
    parser.add_argument("--rows", type=int, default=500, help="Number of query rows to process (default: 500)")
    parser.add_argument("--all", action="store_true", help="Index the full dataset")
    args = parser.parse_args()

    if not verify_parquet():
        download_dataset()

    print("--> Loading MSMARCO-XI records from Parquet...")
    df = pd.read_parquet(parquet_filename, engine="pyarrow")
    if not args.all:
        df = df.head(args.rows)
    print(f"--> Loaded {len(df)} query documents into memory.")

    # 1. Apply 5-Way Hybrid Chunking
    all_chunks = []
    print(f"--> Executing 5-Way Hybrid Chunking on {len(df)} documents...")
    for record in df.to_dict("records"):
        chunks = process_record_5way_chunking(record)
        all_chunks.extend(chunks)

    print(f"--> 5-Way Hybrid Chunking Finished! Total Generated Units: {len(all_chunks)}")

    # Strategy breakdown
    strategy_counts = {}
    for c in all_chunks:
        st = c.get("chunk_strategy", "other")
        strategy_counts[st] = strategy_counts.get(st, 0) + 1
    for st, count in strategy_counts.items():
        print(f"    - {st}: {count} chunks")

    # 2. Embedding Generation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Initializing BAAI/bge-small-en-v1.5 embedding model on {device.upper()}...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    texts_to_embed = [c["text"] for c in all_chunks]
    batch_size = 256 if device == "cuda" else 64

    print(f"--> Generating normalized dense embeddings (batch_size={batch_size})...")
    embeddings = model.encode(
        texts_to_embed,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    for i, c in enumerate(all_chunks):
        c["vector"] = embeddings[i]

    # 3. LanceDB Ingestion & IVF-PQ Indexing
    print(f"--> Connecting to LanceDB at {db_path}...")
    db = lancedb.connect(db_path)
    table = db.create_table("msmarco_vector_store", data=all_chunks, mode="overwrite")

    print("--> Creating IVF-PQ Cosine Vector Index in LanceDB...")
    try:
        table.create_index(metric="cosine", num_partitions=32, num_sub_vectors=16)
    except Exception as e:
        print(f"--> Index notice: {e}")

    print("--> [SUCCESS] Dhwani 5-Way Hybrid Vector Store Indexed Successfully!")

if __name__ == "__main__":
    main()