"""
Step 07: Generate semantic labels for HDBSCAN clusters.
Reads data/processed/unam_embeddings_2d.parquet and outputs data/tiles/cluster_labels.json
Uses TF-IDF to find the most representative keywords for each cluster.
"""
import argparse
import pandas as pd
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from semantic_research_atlas.utils import load_config


def get_top_keywords(tfidf_matrix, vectorizer, top_n=4):
    """Extract top N keywords from a TF-IDF matrix for a single cluster's averaged tfidf profile"""
    feature_names = vectorizer.get_feature_names_out()
    # Sum the TF-IDF scores across all documents in the cluster
    summed_tfidf = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    top_indices = summed_tfidf.argsort()[-top_n:][::-1]
    return [feature_names[i].capitalize() for i in top_indices]


def main(config_path: str):
    cfg = load_config(config_path)
    in_path = f"{cfg['paths']['processed']}/unam_embeddings_2d.parquet"
    out_path = "data/tiles/cluster_labels.json"
    
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing {in_path}. Run 03_umap_cluster.py first.")

    print(f"Loading {in_path}...")
    df = pd.read_parquet(in_path)
    
    clusters_data = []
    labels_data = []
    
    # We ignore cluster -1 because it's HDBSCAN noise
    valid_clusters = [c for c in df['cluster'].unique() if c != -1]
    print(f"Found {len(valid_clusters)} semantic clusters. Extracting keywords via TF-IDF...")

    custom_stopwords = set([
        "the", "and", "of", "to", "in", "for", "is", "on", "that", "by", "this", "with", "from", 
        "as", "it", "are", "we", "an", "be", "was", "or", "which", "study", "analysis", "results",
        "using", "used", "paper", "based", "model", "data", "also", "were", "show", "can", "has",
        "effect", "effects", "different", "two", "method", "methods", "between", "these",
        "de", "la", "el", "en", "y", "a", "los", "se", "del", "las", "un", "por", "con", "no",
        "una", "su", "para", "es", "al", "lo", "como", "mas", "o", "pero", "sus", "le", "ya",
        "este", "esta", "estudio", "analisis", "resultados", "metodo", "desarrollo", "articulo",
        # MathML / XML leftovers
        "mrow", "math", "xmlns", "inline", "msub", "mi", "mn", "mo", "msup", "msubsup",
        "msqrt", "mtext", "mathvariant", "bold", "display", "http", "www", "org", "1998"
    ])

    from sklearn.cluster import KMeans
    import re

    # TF-IDF configured to strip accents to merge "Mexico" and "México"
    vectorizer = TfidfVectorizer(
        max_df=0.8,
        min_df=1,           
        max_features=1000,
        stop_words=list(custom_stopwords),
        strip_accents='unicode',
        token_pattern=r'(?u)\b[a-zA-Z]{4,}\b'
    )

    for cluster_id in valid_clusters:
        cluster_df = df[df['cluster'] == cluster_id]
        
        centroid_x = float(cluster_df['x'].mean())
        centroid_y = float(cluster_df['y'].mean())
        count = int(len(cluster_df))
        
        raw_texts = (cluster_df['title'].fillna('') + ' ' + cluster_df['abstract'].fillna('')).tolist()
        # Remove XML/HTML tags using regex
        texts = [re.sub(r'<[^>]+>', ' ', t) for t in raw_texts]
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            main_keywords = get_top_keywords(tfidf_matrix, vectorizer, top_n=5)
        except ValueError:
            main_keywords = ["Unknown"]
            
        main_label = main_keywords[0] if main_keywords else f"Cluster {cluster_id}"
        
        # 1. Save Cluster Metadata for Sidebar
        clusters_data.append({
            "cluster": int(cluster_id),
            "label": main_label,
            "keywords": main_keywords,
            "x": centroid_x,
            "y": centroid_y,
            "count": count
        })

        # 2. Save Level 1 Label (Main Topic)
        labels_data.append({
            "cluster": int(cluster_id),
            "text": main_label,
            "x": centroid_x,
            "y": centroid_y,
            "level": 1
        })

        # 3. Save Level 2 Labels (Sub-topics via KMeans on coordinates)
        if count > 50 and len(main_keywords) > 1:
            try:
                # Find 3 sub-regions inside this cluster based on X,Y coords
                coords = cluster_df[['x', 'y']].values
                n_sub = min(3, len(coords))
                kmeans = KMeans(n_clusters=n_sub, n_init="auto", random_state=42)
                sub_labels = kmeans.fit_predict(coords)
                
                # Assign one secondary keyword to each sub-region
                used_words = {main_label}
                for i in range(n_sub):
                    sub_mask = (sub_labels == i)
                    sub_texts = [texts[j] for j, mask in enumerate(sub_mask) if mask]
                    
                    if not sub_texts: continue
                    
                    sub_tfidf = vectorizer.transform(sub_texts)
                    sub_words = get_top_keywords(sub_tfidf, vectorizer, top_n=3)
                    
                    # Find a word not yet used
                    chosen_word = None
                    for w in sub_words:
                        if w not in used_words:
                            chosen_word = w
                            break
                            
                    if chosen_word:
                        used_words.add(chosen_word)
                        labels_data.append({
                            "cluster": int(cluster_id),
                            "text": chosen_word,
                            "x": float(kmeans.cluster_centers_[i][0]),
                            "y": float(kmeans.cluster_centers_[i][1]),
                            "level": 2
                        })
            except Exception as e:
                pass

    clusters_data.sort(key=lambda x: x["count"], reverse=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "clusters": clusters_data,
            "labels": labels_data
        }, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(labels_data)} map labels for {len(valid_clusters)} clusters.")
    print(f"Saved cluster metadata to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
