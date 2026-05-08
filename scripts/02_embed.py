"""
Step 02: Generate embeddings for each record.
Outputs data/processed/embeddings.npy and data/processed/records_with_embeddings.parquet
"""
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from semantic_research_atlas.utils import load_config


def main(config_path: str):
    cfg = load_config(config_path)
    in_path = f"{cfg['paths']['raw']}/records.parquet"
    df = pd.read_parquet(in_path)

    model = SentenceTransformer(cfg["embeddings"]["model_name"])
    texts = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).tolist()
    embeddings = model.encode(texts, batch_size=cfg["embeddings"]["batch_size"], show_progress_bar=True)

    np.save(f"{cfg['paths']['processed']}/embeddings.npy", embeddings)
    df.to_parquet(f"{cfg['paths']['processed']}/records.parquet", index=False)
    print("Embeddings saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
