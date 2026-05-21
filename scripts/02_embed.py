"""
Step 02: Generate sentence embeddings using CTranslate2 INT8 quantized model.

Optimized for CPU-only execution on consumer hardware (4+ cores, 8GB+ RAM).
Uses INT8 quantization to leverage AVX2/AVX-VNNI instructions for ~3-5x speedup
over vanilla PyTorch SentenceTransformers.

Usage:
    python scripts/02_embed.py --config config/default.yaml --inst-id 01tmp8f25 --mode full
"""

# ═══════════════════════════════════════════════════════════════════════════════
# THREAD LIMITING — Must be set BEFORE any library imports to prevent CPU freeze.
# Reserves threads for the OS so the machine remains responsive during inference.
# ═══════════════════════════════════════════════════════════════════════════════
import os

_MAX_THREADS = max(1, min(os.cpu_count() - 2, 8))

os.environ["OMP_NUM_THREADS"]       = str(_MAX_THREADS)
os.environ["MKL_NUM_THREADS"]       = str(_MAX_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]  = str(_MAX_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(_MAX_THREADS)
os.environ["NUMEXPR_NUM_THREADS"]   = str(_MAX_THREADS)
# Disable ONNX/tokenizers parallelism to avoid fork bombs
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import time
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import ctranslate2

from transformers import AutoTokenizer
from semantic_research_atlas.utils import load_config

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CT2_MODEL_DIR       = "data/models/compiled_miniLM"
CHUNK_SIZE          = 50_000
ENCODE_BATCH_SIZE   = 64


def ensure_ct2_model(model_name: str, ct2_dir: str) -> str:
    """
    Ensure the CTranslate2 INT8 model exists on disk.
    If not, download the HuggingFace model and convert it automatically.
    Returns the path to the compiled model directory.
    """
    marker = os.path.join(ct2_dir, "model.bin")
    if os.path.exists(marker):
        print(f" CT2 INT8 model found at: {ct2_dir}")
        return ct2_dir

    print(f"⚙ First-time setup: converting {model_name} → CTranslate2 INT8...")
    print("  This downloads ~500MB and takes ~2 minutes. Subsequent runs are instant.")

    import subprocess
    import sys

    os.makedirs(ct2_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", model_name,
        "--quantization", "int8",
        "--output_dir", ct2_dir,
        "--force",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Fallback: try the CLI tool directly
        cmd_alt = [
            "ct2-transformers-converter",
            "--model", model_name,
            "--quantization", "int8",
            "--output_dir", ct2_dir,
            "--force",
        ]
        result = subprocess.run(cmd_alt, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to convert model. stderr:\n{result.stderr}\n"
            f"Ensure ctranslate2 is installed: pip install ctranslate2 transformers"
        )

    # Copy tokenizer files to the same directory for self-contained deployment
    from transformers import AutoTokenizer as AT
    tokenizer = AT.from_pretrained(model_name)
    tokenizer.save_pretrained(ct2_dir)

    print(f"✓ Model compiled and saved to: {ct2_dir}")
    return ct2_dir


def mean_pooling(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Apply mean pooling to encoder hidden states, masked by attention_mask.
    This replicates the pooling behavior of sentence-transformers.
    """
    # hidden_states: (batch, seq_len, hidden_dim)
    # attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
    mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
    sum_embeddings = np.sum(hidden_states * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def normalize_l2(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each embedding vector (unit sphere)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def encode_batch(
    texts: list[str],
    tokenizer: AutoTokenizer,
    encoder: ctranslate2.Encoder,
    max_length: int = 128,
) -> np.ndarray:
    """
    Encode a batch of texts into normalized sentence embeddings using CT2 INT8.
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    # CTranslate2 expects token IDs as a StorageView
    input_ids = encoded["input_ids"]
    attention_mask = np.array(encoded["attention_mask"], dtype=np.int32)

    # Convert to list of lists of strings (CT2 encoder input format)
    tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

    # Run inference through the INT8 engine
    outputs = encoder.forward_batch(tokens)

    # Extract hidden states and convert to numpy
    # outputs is an EncoderForwardOutput with a last_hidden_state attribute
    hidden_states = np.array(outputs.last_hidden_state)

    # Pool + normalize
    pooled = mean_pooling(hidden_states, attention_mask)
    return normalize_l2(pooled)


def main(config_path: str, inst_id: str = "default", mode: str = "full"):
    cfg = load_config(config_path)

    # ── Resolve dynamic paths ──
    prefix = f"{inst_id}_{mode}"
    raw_dir = os.path.join(cfg["paths"]["raw"], prefix)
    proc_dir = os.path.join(cfg["paths"]["processed"], prefix)
    os.makedirs(proc_dir, exist_ok=True)

    in_path = os.path.join(raw_dir, "records.parquet")
    out_parquet = os.path.join(proc_dir, "records.parquet")
    out_npy = os.path.join(proc_dir, "embeddings.npy")

    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"Input not found: {in_path}\n"
            f"Run 01_ingest.py first with --inst-id {inst_id} --mode {mode}"
        )

    # ── Count records via DuckDB (zero RAM) ──
    conn = duckdb.connect()
    total = conn.execute(f"SELECT count(*) FROM '{in_path}'").fetchone()[0]
    print(f"Total records: {total:,}")
    print(f"Threads: {_MAX_THREADS} (of {os.cpu_count()} available)")

    # ── Load or compile the CT2 INT8 model ──
    model_name = cfg.get("embeddings", {}).get("model_name", DEFAULT_MODEL_NAME)
    ct2_dir = cfg.get("embeddings", {}).get("ct2_model_dir", CT2_MODEL_DIR)
    ct2_path = ensure_ct2_model(model_name, ct2_dir)

    print(f"Loading CTranslate2 INT8 encoder...")
    encoder = ctranslate2.Encoder(
        ct2_path,
        device="cpu",
        inter_threads=_MAX_THREADS,
        compute_type="int8",
    )
    tokenizer = AutoTokenizer.from_pretrained(ct2_path)
    print(f" Encoder loaded ({model_name} → INT8)")

    # ── Chunked processing ──
    batch_size = cfg.get("embeddings", {}).get("batch_size", ENCODE_BATCH_SIZE)
    embeddings_list = []
    writer = None
    offset = 0
    start = time.time()

    while offset < total:
        chunk_end = min(offset + CHUNK_SIZE, total)
        pct = chunk_end * 100 // total
        print(f"\n── Chunk {offset:,} → {chunk_end:,} ({pct}%) ──")

        df_chunk = conn.execute(
            f"SELECT * FROM '{in_path}' LIMIT {CHUNK_SIZE} OFFSET {offset}"
        ).df()

        texts = (
            df_chunk["title"].fillna("") + ". " + df_chunk["abstract"].fillna("")
        ).tolist()

        # Encode in micro-batches
        chunk_embeddings = []
        for i in range(0, len(texts), batch_size):
            micro = texts[i : i + batch_size]
            emb = encode_batch(micro, tokenizer, encoder)
            chunk_embeddings.append(emb)

            # Progress within chunk
            done = min(i + batch_size, len(texts))
            elapsed = time.time() - start
            total_done = offset + done
            rate = total_done / elapsed if elapsed > 0 else 0
            eta = (total - total_done) / rate if rate > 0 else 0
            print(
                f"  [{done}/{len(texts)}] "
                f"Total: {total_done:,}/{total:,} | "
                f"{rate:.0f} docs/s | "
                f"ETA: {eta/60:.1f}m",
                end="\r",
            )

        print()  # newline after progress

        chunk_emb = np.vstack(chunk_embeddings)
        embeddings_list.append(chunk_emb)

        # Stream records to Parquet
        table_chunk = pa.Table.from_pandas(df_chunk)
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, table_chunk.schema)
        writer.write_table(table_chunk)

        offset += CHUNK_SIZE

        # Free memory
        del df_chunk, texts, chunk_embeddings, chunk_emb, table_chunk
        gc.collect()

    if writer:
        writer.close()

    conn.close()

    if embeddings_list:
        final = np.vstack(embeddings_list)
        np.save(out_npy, final)
        total_time = time.time() - start
        print(f"\n{'='*60}")
        print(f" Saved {final.shape[0]:,} embeddings (dim={final.shape[1]})")
        print(f"  → {out_npy}")
        print(f"  → {out_parquet}")
        print(f"  Total: {total_time/60:.1f} min ({final.shape[0]/total_time:.0f} docs/s avg)")
        print(f"{'='*60}")
    else:
        print(" No records processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate INT8-quantized sentence embeddings (CTranslate2)"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--inst-id", default="default",
        help="Institution ID (e.g., 01tmp8f25 for UNAM). Used for path prefixing."
    )
    parser.add_argument(
        "--mode", default="full", choices=["full", "limited"],
        help="Pipeline mode: 'full' (all records) or 'limited' (15k cap)"
    )
    args = parser.parse_args()
    main(args.config, args.inst_id, args.mode)
