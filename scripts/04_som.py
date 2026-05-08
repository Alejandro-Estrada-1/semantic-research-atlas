"""
Step 04: SOM training and mapping.
Outputs:
- data/processed/som_map.parquet
- data/processed/som_umatrix.parquet
"""
import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
