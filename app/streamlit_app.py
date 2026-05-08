import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Semantic Research Atlas", layout="wide")

st.title("Semantic Research Atlas")
st.caption("UMAP + SOM exploration with semantic search")

@st.cache_data
def load_umap(path: str):
    return pd.read_parquet(path)

@st.cache_data
def load_som(path: str):
    return pd.read_parquet(path)

@st.cache_resource
def load_model(model_name: str):
    return SentenceTransformer(model_name)

@st.cache_resource
def load_faiss_index(path: str):
    if not os.path.exists(path):
        return None
    return faiss.read_index(path)

umap_path = st.sidebar.text_input("UMAP parquet", "data/processed/unam_embeddings_2d.parquet")
som_path = st.sidebar.text_input("SOM parquet", "data/processed/som_map.parquet")
index_path = st.sidebar.text_input("FAISS index", "data/index/index.faiss")
model_name = st.sidebar.text_input(
    "Embedding model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

view = st.sidebar.radio("View", ["UMAP", "SOM"])

st.sidebar.subheader("Semantic search")
query = st.sidebar.text_input("Search topic")
top_k = st.sidebar.slider("Top K", min_value=5, max_value=100, value=20, step=5)

matches = None
if query:
    index = load_faiss_index(index_path)
    if index is None:
        st.warning("FAISS index not found. Build it with scripts/05_faiss.py")
    else:
        model = load_model(model_name)
        q_emb = model.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)
        _, idx = index.search(q_emb, top_k)
        matches = idx[0].tolist()

if view == "UMAP":
    df = load_umap(umap_path)
    color_col = "cluster"

    if matches is not None:
        df = df.copy()
        df["match"] = "Other"
        df.loc[matches, "match"] = "Match"
        color_col = "match"

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_col,
        hover_data=["title", "year", "faculty", "source"],
        height=700,
        render_mode="webgl",
    )
    st.plotly_chart(fig, use_container_width=True)

    if matches is not None:
        st.subheader("Top matches")
        st.dataframe(df.loc[matches, ["title", "year", "faculty", "source", "url"]])
else:
    df = load_som(som_path)
    fig = px.density_heatmap(
        df,
        x="som_x",
        y="som_y",
        height=700,
    )

    if matches is not None:
        som_matches = df.loc[matches]
        fig.add_scatter(
            x=som_matches["som_x"],
            y=som_matches["som_y"],
            mode="markers",
            marker=dict(color="red", size=6),
            name="Match",
        )

    st.plotly_chart(fig, use_container_width=True)

    if matches is not None:
        st.subheader("Top matches")
        st.dataframe(df.loc[matches, ["title", "year", "faculty", "source", "url"]])
