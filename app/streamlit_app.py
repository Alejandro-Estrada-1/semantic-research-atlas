import os
import streamlit as st
import pandas as pd
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

@st.cache_data
def load_umatrix(path: str):
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
umatrix_path = st.sidebar.text_input(
    "SOM U-Matrix parquet", "data/processed/som_umatrix.parquet"
)
index_path = st.sidebar.text_input("FAISS index", "data/index/index.faiss")
model_name = st.sidebar.text_input(
    "Embedding model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

view = st.sidebar.radio("View", ["UMAP", "SOM (U-Matrix)", "Compare"])

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


def build_umap_figure(df, matches_idx):
    color_col = "cluster"
    plot_df = df
    if matches_idx is not None:
        plot_df = df.copy()
        plot_df["match"] = "Other"
        plot_df.loc[matches_idx, "match"] = "Match"
        color_col = "match"

    return px.scatter(
        plot_df,
        x="x",
        y="y",
        color=color_col,
        hover_data=["title", "year", "faculty", "source"],
        height=650,
        render_mode="webgl",
    )


def build_umatrix_figure(umatrix_df, som_df, matches_idx):
    pivot = umatrix_df.pivot(index="som_y", columns="som_x", values="distance")
    fig = px.imshow(
        pivot,
        origin="lower",
        aspect="auto",
        color_continuous_scale="Viridis",
        height=650,
    )

    if matches_idx is not None:
        som_matches = som_df.loc[matches_idx]
        fig.add_scatter(
            x=som_matches["som_x"],
            y=som_matches["som_y"],
            mode="markers",
            marker=dict(color="red", size=6),
            name="Match",
        )

    return fig


if view == "UMAP":
    df = load_umap(umap_path)
    fig = build_umap_figure(df, matches)
    st.plotly_chart(fig, use_container_width=True)

    if matches is not None:
        st.subheader("Top matches")
        st.dataframe(df.loc[matches, ["title", "year", "faculty", "source", "url"]])

elif view == "SOM (U-Matrix)":
    som_df = load_som(som_path)
    umatrix_df = load_umatrix(umatrix_path)
    fig = build_umatrix_figure(umatrix_df, som_df, matches)
    st.plotly_chart(fig, use_container_width=True)

    if matches is not None:
        st.subheader("Top matches")
        st.dataframe(som_df.loc[matches, ["title", "year", "faculty", "source", "url"]])

else:
    left, right = st.columns(2)
    umap_df = load_umap(umap_path)
    som_df = load_som(som_path)
    umatrix_df = load_umatrix(umatrix_path)

    with left:
        st.subheader("UMAP")
        st.plotly_chart(build_umap_figure(umap_df, matches), use_container_width=True)

    with right:
        st.subheader("SOM U-Matrix")
        st.plotly_chart(
            build_umatrix_figure(umatrix_df, som_df, matches),
            use_container_width=True,
        )

    if matches is not None:
        st.subheader("Top matches")
        st.dataframe(umap_df.loc[matches, ["title", "year", "faculty", "source", "url"]])
