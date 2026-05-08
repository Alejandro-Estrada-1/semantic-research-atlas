import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Semantic Research Atlas", layout="wide")

st.title("Semantic Research Atlas")
st.caption("UMAP + SOM exploration with semantic search")

@st.cache_data
def load_umap(path: str):
    return pd.read_parquet(path)

@st.cache_data
def load_som(path: str):
    return pd.read_parquet(path)

umap_path = st.sidebar.text_input("UMAP parquet", "data/processed/unam_embeddings_2d.parquet")
som_path = st.sidebar.text_input("SOM parquet", "data/processed/som_map.parquet")

view = st.sidebar.radio("View", ["UMAP", "SOM"])

if view == "UMAP":
    df = load_umap(umap_path)
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_data=["title", "year", "faculty"],
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    df = load_som(som_path)
    fig = px.density_heatmap(
        df,
        x="som_x",
        y="som_y",
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Semantic Search (placeholder)")
query = st.text_input("Search topic")
if query:
    st.info("Hook FAISS search here to highlight nearest points.")
