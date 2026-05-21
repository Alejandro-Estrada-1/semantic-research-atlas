"""
Step 04: Self-Organizing Map (SOM) training, mapping, and frontend export.

Usage:
    python scripts/04_som.py --config config/default.yaml --inst-id 01tmp8f25 --mode full

Outputs:
    data/processed/{inst_id}_{mode}/som_map.parquet
    data/processed/{inst_id}_{mode}/som_umatrix.parquet
    data/tiles/{inst_id}_{mode}/som_umatrix.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from minisom import MiniSom

from semantic_research_atlas.utils import load_config


def main(config_path: str, inst_id: str = "default", mode: str = "full"):
    cfg = load_config(config_path)

    prefix = f"{inst_id}_{mode}"
    proc_dir = os.path.join(cfg["paths"]["processed"], prefix)
    tiles_dir = os.path.join("data/tiles", prefix)
    os.makedirs(tiles_dir, exist_ok=True)

    df = pd.read_parquet(os.path.join(proc_dir, "records.parquet"))
    embeddings = np.load(os.path.join(proc_dir, "embeddings.npy"))

    # Subsample for training
    sample_size = min(cfg["som"]["train_sample_size"], len(embeddings))
    idx = np.random.choice(len(embeddings), size=sample_size, replace=False)
    train_data = embeddings[idx]

    print(f"Training SOM ({cfg['som']['grid_x']}×{cfg['som']['grid_y']}) on {sample_size:,} samples...")
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
    print("Mapping all points to SOM neurons...")
    winners = np.array([som.winner(v) for v in embeddings])
    df["som_x"] = winners[:, 0]
    df["som_y"] = winners[:, 1]

    som_path = os.path.join(proc_dir, "som_map.parquet")
    df.to_parquet(som_path, index=False)
    print(f"✓ Saved: {som_path}")

    # U-Matrix
    umatrix = som.distance_map()
    umatrix_records = [
        {"som_x": x, "som_y": y, "distance": float(umatrix[x, y])}
        for x in range(umatrix.shape[0])
        for y in range(umatrix.shape[1])
    ]
    um_path = os.path.join(proc_dir, "som_umatrix.parquet")
    pd.DataFrame(umatrix_records).to_parquet(um_path, index=False)
    print(f"✓ Saved: {um_path}")

    # ── Export JSON for frontend ──
    _export_frontend_json(cfg, df, umatrix, tiles_dir)


def _export_frontend_json(cfg, df, umatrix, tiles_dir):
    """Export lightweight SOM data for the React frontend."""
    gx = cfg["som"]["grid_x"]
    gy = cfg["som"]["grid_y"]

    umatrix_2d = [[round(float(umatrix[x, y]), 4) for y in range(gy)] for x in range(gx)]

    # Sample ~5000 points
    n = min(5000, len(df))
    sample = df.iloc[np.random.RandomState(42).choice(len(df), size=n, replace=False)]

    points = [
        {
            "som_x": int(r["som_x"]), "som_y": int(r["som_y"]),
            "cluster": int(r["cluster"]) if "cluster" in r and pd.notna(r.get("cluster")) else -1,
            "title": str(r.get("title", ""))[:120],
            "year": int(r["year"]) if pd.notna(r.get("year")) else None,
        }
        for _, r in sample.iterrows()
    ]

    # Density per cell
    density = [[0] * gy for _ in range(gx)]
    for _, r in df.iterrows():
        density[int(r["som_x"])][int(r["som_y"])] += 1

    out = os.path.join(tiles_dir, "som_umatrix.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"grid_x": gx, "grid_y": gy, "umatrix": umatrix_2d,
                    "density": density, "points": points}, f, ensure_ascii=False)
    print(f"✓ Saved frontend SOM: {out} ({n} sample points)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inst-id", default="default")
    parser.add_argument("--mode", default="full", choices=["full", "limited"])
    args = parser.parse_args()
    main(args.config, args.inst_id, args.mode)
