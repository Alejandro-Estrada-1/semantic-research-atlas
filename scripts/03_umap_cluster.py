"""
Step 03: UMAP dimensionality reduction + HDBSCAN clustering.

Optimized for low-memory execution (single-threaded, deferred loading).

Usage:
    python scripts/03_umap_cluster.py --config config/default.yaml --inst-id 01tmp8f25 --mode full

Outputs:
    data/processed/{inst_id}_{mode}/embeddings_2d.parquet
    data/processed/{inst_id}_{mode}/embeddings_2d.arrow
"""
import argparse
import gc
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import umap
import hdbscan

from semantic_research_atlas.utils import load_config


def main(config_path: str, inst_id: str = "default", mode: str = "full"):
    cfg = load_config(config_path)

    prefix = f"{inst_id}_{mode}"
    proc_dir = os.path.join(cfg["paths"]["processed"], prefix)

    embeddings_path = os.path.join(proc_dir, "embeddings.npy")
    records_path = os.path.join(proc_dir, "records.parquet")

    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path).astype(np.float32)

    if embeddings.size == 0 or embeddings.ndim != 2:
        raise ValueError(f"Invalid embeddings. Shape: {embeddings.shape}")

    total = embeddings.shape[0]
    print(f"Loaded {total:,} embeddings (dim={embeddings.shape[1]})")

    # ── UMAP ──
    print("Running UMAP (low memory, single-threaded)...")
    reducer = umap.UMAP(
        n_neighbors=cfg["umap"]["n_neighbors"],
        min_dist=cfg["umap"]["min_dist"],
        n_components=cfg["umap"]["n_components"],
        metric=cfg["umap"]["metric"],
        random_state=42,
        low_memory=True,
        n_jobs=1,
    )
    coords = reducer.fit_transform(embeddings)

    print("UMAP done. Freeing memory...")
    del embeddings, reducer
    gc.collect()

    # ── HDBSCAN ──
    print("Running HDBSCAN (single-threaded)...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg["clustering"]["min_cluster_size"],
        core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(coords)

    # ── Attach to DataFrame ──
    print("Loading records...")
    df = pd.read_parquet(records_path)

    if len(df) != total:
        raise ValueError(f"Records ({len(df)}) ≠ embeddings ({total})")

    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["cluster"] = labels

    out_parquet = os.path.join(proc_dir, "embeddings_2d.parquet")
    df.to_parquet(out_parquet, index=False)

    out_arrow = os.path.join(proc_dir, "embeddings_2d.arrow")
    table = pa.Table.from_pandas(df)
    with ipc.new_file(out_arrow, table.schema) as writer:
        writer.write(table)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = (labels == -1).sum()
    print(f" {n_clusters} clusters found ({noise:,} noise points)")
    print(f" Saved: {out_parquet}")
    print(f" Saved: {out_arrow}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inst-id", default="default")
    parser.add_argument("--mode", default="full", choices=["full", "limited"])
    args = parser.parse_args()
    main(args.config, args.inst_id, args.mode)
