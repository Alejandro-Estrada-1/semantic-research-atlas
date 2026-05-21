"""
Step 06: Generate semantic labels for HDBSCAN clusters via TF-IDF.

Usage:
    python scripts/06_generate_labels.py --config config/default.yaml --inst-id 01tmp8f25 --mode full

Outputs: data/tiles/{inst_id}_{mode}/cluster_labels.json
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from semantic_research_atlas.utils import load_config


def get_top_keywords(tfidf_matrix, vectorizer, top_n=4):
    """Extract top N keywords by summed TF-IDF score."""
    features = vectorizer.get_feature_names_out()
    scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    top_idx = scores.argsort()[-top_n:][::-1]
    return [features[i].capitalize() for i in top_idx]


STOPWORDS = set([
    "the", "and", "of", "to", "in", "for", "is", "on", "that", "by", "this", "with", "from",
    "as", "it", "are", "we", "an", "be", "was", "or", "which", "study", "analysis", "results",
    "using", "used", "paper", "based", "model", "data", "also", "were", "show", "can", "has",
    "effect", "effects", "different", "two", "method", "methods", "between", "these",
    "de", "la", "el", "en", "y", "a", "los", "se", "del", "las", "un", "por", "con", "no",
    "una", "su", "para", "es", "al", "lo", "como", "mas", "o", "pero", "sus", "le", "ya",
    "este", "esta", "estudio", "analisis", "resultados", "metodo", "desarrollo", "articulo",
    "mrow", "math", "xmlns", "inline", "msub", "mi", "mn", "mo", "msup", "msubsup",
    "msqrt", "mtext", "mathvariant", "bold", "display", "http", "www", "org", "1998",
])


def main(config_path: str, inst_id: str = "default", mode: str = "full"):
    cfg = load_config(config_path)

    prefix = f"{inst_id}_{mode}"
    proc_dir = os.path.join(cfg["paths"]["processed"], prefix)
    tiles_dir = os.path.join("data/tiles", prefix)
    os.makedirs(tiles_dir, exist_ok=True)

    in_path = os.path.join(proc_dir, "embeddings_2d.parquet")
    out_path = os.path.join(tiles_dir, "cluster_labels.json")

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing {in_path}. Run 03_umap_cluster.py first.")

    print(f"Loading {in_path}...")
    df = pd.read_parquet(in_path)

    valid_clusters = [c for c in df["cluster"].unique() if c != -1]
    print(f"Found {len(valid_clusters)} clusters. Extracting keywords...")

    vectorizer = TfidfVectorizer(
        max_df=0.8, min_df=1, max_features=1000,
        stop_words=list(STOPWORDS),
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z]{4,}\b",
    )

    clusters_data = []
    labels_data = []

    for cid in valid_clusters:
        cdf = df[df["cluster"] == cid]
        cx, cy = float(cdf["x"].mean()), float(cdf["y"].mean())
        count = len(cdf)

        raw = (cdf["title"].fillna("") + " " + cdf["abstract"].fillna("")).tolist()
        texts = [re.sub(r"<[^>]+>", " ", t) for t in raw]

        try:
            tfidf = vectorizer.fit_transform(texts)
            keywords = get_top_keywords(tfidf, vectorizer, top_n=5)
        except ValueError:
            keywords = ["Unknown"]

        label = keywords[0] if keywords else f"Cluster {cid}"

        clusters_data.append({
            "cluster": int(cid), "label": label, "keywords": keywords,
            "x": cx, "y": cy, "count": count,
        })
        labels_data.append({
            "cluster": int(cid), "text": label, "x": cx, "y": cy, "level": 1,
        })

        # Sub-topic labels via KMeans
        if count > 50 and len(keywords) > 1:
            try:
                coords = cdf[["x", "y"]].values
                n_sub = min(3, len(coords))
                km = KMeans(n_clusters=n_sub, n_init="auto", random_state=42)
                sub_labels = km.fit_predict(coords)

                used = {label}
                for i in range(n_sub):
                    sub_texts = [texts[j] for j, m in enumerate(sub_labels == i) if m]
                    if not sub_texts:
                        continue
                    sub_words = get_top_keywords(vectorizer.transform(sub_texts), vectorizer, 3)
                    chosen = next((w for w in sub_words if w not in used), None)
                    if chosen:
                        used.add(chosen)
                        labels_data.append({
                            "cluster": int(cid), "text": chosen,
                            "x": float(km.cluster_centers_[i][0]),
                            "y": float(km.cluster_centers_[i][1]),
                            "level": 2,
                        })
            except Exception:
                pass

    clusters_data.sort(key=lambda x: x["count"], reverse=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"clusters": clusters_data, "labels": labels_data}, f, ensure_ascii=False, indent=2)

    print(f"✓ {len(labels_data)} labels for {len(valid_clusters)} clusters → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inst-id", default="default")
    parser.add_argument("--mode", default="full", choices=["full", "limited"])
    args = parser.parse_args()
    main(args.config, args.inst_id, args.mode)
