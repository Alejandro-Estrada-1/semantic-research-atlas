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
    cfg = load_config(config_path)
    df = pd.read_parquet(f"{cfg['paths']['processed']}/records.parquet")
    embeddings = np.load(f"{cfg['paths']['processed']}/embeddings.npy")

    reducer = umap.UMAP(
        n_neighbors=cfg["umap"]["n_neighbors"],
        min_dist=cfg["umap"]["min_dist"],
        n_components=cfg["umap"]["n_components"],
        metric=cfg["umap"]["metric"],
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=cfg["clustering"]["min_cluster_size"])
    labels = clusterer.fit_predict(embeddings)

    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["cluster"] = labels

    out_path = f"{cfg['paths']['processed']}/unam_embeddings_2d.parquet"
    df.to_parquet(out_path, index=False)

    arrow_path = f"{cfg['paths']['processed']}/unam_embeddings_2d.arrow"
    table = pa.Table.from_pandas(df)
    with ipc.new_file(arrow_path, table.schema) as writer:
        writer.write(table)

    print(f"Saved: {out_path}")
    print(f"Saved: {arrow_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
