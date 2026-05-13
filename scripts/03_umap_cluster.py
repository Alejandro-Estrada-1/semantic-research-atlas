"""
Step 03: UMAP + clustering.
Outputs:
- data/processed/unam_embeddings_2d.parquet
- data/processed/unam_embeddings_2d.arrow
"""
import argparse
import numpy as np
import pandas as pd
import umap
import hdbscan
import pyarrow as pa
import pyarrow.ipc as ipc
from semantic_research_atlas.utils import load_config


def main(config_path: str):
    import gc
    cfg = load_config(config_path)
    
    # 1. Load ONLY the embeddings to save RAM
    embeddings_path = f"{cfg['paths']['processed']}/embeddings.npy"
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)

    if embeddings.size == 0 or embeddings.ndim != 2:
        raise ValueError(f"Invalid embeddings array. Shape: {embeddings.shape}")

    total_records = embeddings.shape[0]
    print(f"Loaded {total_records:,} embeddings.")

    # 2. Cast to float32 to halve memory usage
    embeddings = embeddings.astype(np.float32)

    # 3. UMAP
    print("Running UMAP (low memory mode, single-threaded)...")
    reducer = umap.UMAP(
        n_neighbors=cfg["umap"]["n_neighbors"],
        min_dist=cfg["umap"]["min_dist"],
        n_components=cfg["umap"]["n_components"],
        metric=cfg["umap"]["metric"],
        random_state=42,       # Forces single-threaded (prevents RAM spikes)
        low_memory=True,       # Uses less RAM during k-NN graph construction
        n_jobs=1               # Explicitly prevent multi-processing
    )
    coords = reducer.fit_transform(embeddings)

    # 4. FREE RAM BEFORE HDBSCAN
    print("UMAP finished. Freeing memory...")
    del embeddings
    del reducer
    gc.collect()

    # 5. HDBSCAN
    print("Running HDBSCAN (single-threaded)...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg["clustering"]["min_cluster_size"],
        core_dist_n_jobs=1  # Prevent multi-processing memory duplication
    )
    labels = clusterer.fit_predict(coords)

    # 6. NOW load the DataFrame (we didn't need it before!)
    print("Loading records to attach coordinates...")
    df_path = f"{cfg['paths']['processed']}/records.parquet"
    df = pd.read_parquet(df_path)

    if len(df) != total_records:
        raise ValueError(f"Records count ({len(df)}) does not match embeddings ({total_records}).")

    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["cluster"] = labels

    out_path = f"{cfg['paths']['processed']}/unam_embeddings_2d.parquet"
    df.to_parquet(out_path, index=False)

    arrow_path = f"{cfg['paths']['processed']}/unam_embeddings_2d.arrow"
    table = pa.Table.from_pandas(df)
    with ipc.new_file(arrow_path, table.schema) as writer:
        writer.write(table)

    print(f" Saved: {out_path}")
    print(f" Saved: {arrow_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
