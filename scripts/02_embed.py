"""
Step 02: Generate embeddings for each record.

Supports two modes:
  - LOCAL (CPU):  python scripts/02_embed.py --config config/default.yaml
  - IMPORT (GPU): python scripts/02_embed.py --config config/default.yaml --import-npy data/processed/embeddings.npy

The import mode lets you generate embeddings on Google Colab (GPU) and
import the resulting .npy file back into the local pipeline.
"""
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import os
import time

from semantic_research_atlas.utils import load_config


def main(config_path: str, import_npy: str = None):
    cfg = load_config(config_path)
    in_path = f"{cfg['paths']['raw']}/records.parquet"
    out_parquet = f"{cfg['paths']['processed']}/records.parquet"
    out_npy = f"{cfg['paths']['processed']}/embeddings.npy"

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input {in_path} not found. Run ingest first.")

    conn = duckdb.connect()
    total = conn.execute(f"SELECT count(*) FROM '{in_path}'").fetchone()[0]
    print(f"Total records: {total:,}")

    # ── Mode: Import pre-computed embeddings from Colab ──
    if import_npy:
        print(f"Importing pre-computed embeddings from: {import_npy}")
        embeddings = np.load(import_npy)

        if embeddings.shape[0] != total:
            raise ValueError(
                f"Mismatch: embeddings has {embeddings.shape[0]:,} rows "
                f"but records.parquet has {total:,} rows. "
                f"Re-run ingest or re-generate embeddings."
            )

        # Copy embeddings to the canonical location
        if os.path.abspath(import_npy) != os.path.abspath(out_npy):
            np.save(out_npy, embeddings)

        # Copy records to processed
        table = pq.read_table(in_path)
        pq.write_table(table, out_parquet)

        print(f"✅ Imported {embeddings.shape[0]:,} embeddings (dim={embeddings.shape[1]})")
        print(f"   Saved: {out_npy}")
        print(f"   Saved: {out_parquet}")
        return

    # ── Mode: Compute embeddings locally (CPU) ──
    from sentence_transformers import SentenceTransformer

    model_name = cfg["embeddings"]["model_name"]
    batch_size_encode = cfg["embeddings"].get("batch_size", 64)
    chunk_size = 50_000

    print(f"Model: {model_name}")
    print(f"Device: CPU (for GPU, use the Colab notebook + --import-npy)")

    model = SentenceTransformer(model_name)

    embeddings_list = []
    writer = None
    offset = 0
    start = time.time()

    while offset < total:
        chunk_end = min(offset + chunk_size, total)
        print(f"Processing chunk: {offset:,} to {chunk_end:,} ({chunk_end*100//total}%)")

        df_chunk = conn.execute(
            f"SELECT * FROM '{in_path}' LIMIT {chunk_size} OFFSET {offset}"
        ).df()

        texts = (df_chunk["title"].fillna("") + ". " + df_chunk["abstract"].fillna("")).tolist()
        emb_chunk = model.encode(texts, batch_size=batch_size_encode, show_progress_bar=True)
        embeddings_list.append(emb_chunk)

        table_chunk = pa.Table.from_pandas(df_chunk)
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, table_chunk.schema)
        writer.write_table(table_chunk)

        offset += chunk_size

        # Progress estimate
        elapsed = time.time() - start
        rate = offset / elapsed
        remaining = (total - offset) / rate if rate > 0 else 0
        print(f"  Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m | {rate:.0f} docs/s")

    if writer:
        writer.close()

    if embeddings_list:
        final_embeddings = np.vstack(embeddings_list)
        np.save(out_npy, final_embeddings)
        total_time = time.time() - start
        print(f"\n✅ Saved {final_embeddings.shape[0]:,} embeddings (dim={final_embeddings.shape[1]})")
        print(f"   Total time: {total_time/60:.1f} minutes")
    else:
        print("No records processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or import embeddings")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--import-npy",
        default=None,
        help="Path to pre-computed embeddings.npy (from Colab GPU). "
             "Skips local computation and imports directly.",
    )
    args = parser.parse_args()
    main(args.config, args.import_npy)
