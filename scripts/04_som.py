"""
Step 04: SOM training and mapping.
Outputs:
- data/processed/som_map.parquet
- data/processed/som_umatrix.parquet
- data/tiles/som_umatrix.json  (for WebGL frontend)
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from minisom import MiniSom
from semantic_research_atlas.utils import load_config


def main(config_path: str):
    cfg = load_config(config_path)
    df = pd.read_parquet(f"{cfg['paths']['processed']}/records.parquet")
    embeddings = np.load(f"{cfg['paths']['processed']}/embeddings.npy")

    # Subsample for training
    sample_size = min(cfg["som"]["train_sample_size"], len(embeddings))
    idx = np.random.choice(len(embeddings), size=sample_size, replace=False)
    train_data = embeddings[idx]

    som = MiniSom(
        x=cfg["som"]["grid_x"],
        y=cfg["som"]["grid_y"],
        input_len=embeddings.shape[1],
        sigma=cfg["som"]["sigma"],
        learning_rate=cfg["som"]["learning_rate"],
        random_seed=42,
    )
    som.random_weights_init(train_data)
    som.train_random(train_data, num_iteration=1000)

    # Map all points
    winners = np.array([som.winner(v) for v in embeddings])
    df["som_x"] = winners[:, 0]
    df["som_y"] = winners[:, 1]

    out_path = f"{cfg['paths']['processed']}/som_map.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    # U-Matrix
    umatrix = som.distance_map()
    umatrix_records = [
        {"som_x": x, "som_y": y, "distance": float(umatrix[x, y])}
        for x in range(umatrix.shape[0])
        for y in range(umatrix.shape[1])
    ]
    umatrix_df = pd.DataFrame(umatrix_records)
    umatrix_path = f"{cfg['paths']['processed']}/som_umatrix.parquet"
    umatrix_df.to_parquet(umatrix_path, index=False)
    print(f"Saved: {umatrix_path}")

    # ── Export JSON for WebGL frontend ──
    export_som_json(cfg, df, umatrix, som)


def export_som_json(cfg, df, umatrix, som):
    """Export SOM data as lightweight JSON for the React frontend."""
    out_path = "data/tiles/som_umatrix.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    grid_x = cfg["som"]["grid_x"]
    grid_y = cfg["som"]["grid_y"]

    # Convert U-Matrix to 2D list
    umatrix_2d = []
    for x in range(grid_x):
        row = []
        for y in range(grid_y):
            row.append(round(float(umatrix[x, y]), 4))
        umatrix_2d.append(row)

    # Sample ~5000 points to avoid huge JSON
    max_points = min(5000, len(df))
    sample_idx = np.random.RandomState(42).choice(len(df), size=max_points, replace=False)
    sample_df = df.iloc[sample_idx]

    points = []
    for _, row in sample_df.iterrows():
        points.append({
            "som_x": int(row["som_x"]),
            "som_y": int(row["som_y"]),
            "cluster": int(row["cluster"]) if "cluster" in row and pd.notna(row.get("cluster")) else -1,
            "title": str(row.get("title", ""))[:120],
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
        })

    # Count documents per cell for density overlay
    cell_counts = {}
    for _, row in df.iterrows():
        key = f"{int(row['som_x'])},{int(row['som_y'])}"
        cell_counts[key] = cell_counts.get(key, 0) + 1

    density = []
    for x in range(grid_x):
        row = []
        for y in range(grid_y):
            row.append(cell_counts.get(f"{x},{y}", 0))
        density.append(row)

    data = {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "umatrix": umatrix_2d,
        "density": density,
        "points": points,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Saved SOM frontend data: {out_path} ({len(points)} sample points)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
