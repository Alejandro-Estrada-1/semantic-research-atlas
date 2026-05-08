"""
Step 05: Build FAISS index for semantic search.
Outputs data/index/index.faiss
"""
import argparse
import numpy as np
import faiss
from semantic_research_atlas.utils import load_config


def main(config_path: str):
    cfg = load_config(config_path)
    embeddings = np.load(f"{cfg['paths']['processed']}/embeddings.npy").astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    out_path = f"{cfg['paths']['index']}/index.faiss"
    faiss.write_index(index, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
