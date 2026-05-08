import os
import streamlit as st
import pandas as pd
import plotly.express as px
import pyarrow as pa
import pyarrow.ipc as ipc
import pydeck as pdk
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Semantic Research Atlas", layout="wide")

st.title("Semantic Research Atlas")
st.caption("UMAP + SOM exploration with semantic search")

@st.cache_data
def load_umap(path: str):
    return pd.read_parquet(path)

@st.cache_data
def load_umatrix(path: str):
    return pd.read_parquet(path)

@st.cache_data
def load_som(path: str):
    return pd.read_parquet(path)

@st.cache_data
def load_arrow(path: str):
    with ipc.open_file(path) as reader:
        table = reader.read_all()
    return table.to_pandas()

@st.cache_resource
def load_model(model_name: str):
    return SentenceTransformer(model_name)

@st.cache_resource
def load_faiss_index(path: str):
    if not os.path.exists(path):
        return None
    return faiss.read_index(path)

umap_path = st.sidebar.text_input("UMAP parquet", "data/processed/unam_embeddings_2d.parquet")
arrow_path = st.sidebar.text_input("UMAP Arrow", "data/processed/unam_embeddings_2d.arrow")
som_path = st.sidebar.text_input("SOM parquet", "data/processed/som_map.parquet")
umatrix_path = st.sidebar.text_input(
    "SOM U-Matrix parquet", "data/processed/som_umatrix.parquet"
)
index_path = st.sidebar.text_input("FAISS index", "data/index/index.faiss")
model_name = st.sidebar.text_input(
    "Embedding model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

view = st.sidebar.radio("View", ["UMAP (Pydeck)", "SOM (U-Matrix)", "Compare"])

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


def build_umap_pydeck(df, matches_idx):
    plot_df = df
    color = [90, 90, 90]

    if matches_idx is not None:
        plot_df = df.copy()
        plot_df["is_match"] = False
        plot_df.loc[matches_idx, "is_match"] = True
        color = None

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position="[x, y]",
        get_radius=15,
        radius_min_pixels=1,
        radius_max_pixels=6,
        get_fill_color=("[255, 90, 90]" if matches_idx is not None else color),
        pickable=True,
    )

    view_state = pdk.ViewState(
        longitude=float(plot_df["x"].mean()),
        latitude=float(plot_df["y"].mean()),
        zoom=4,
        min_zoom=1,
        max_zoom=15,
    )

    tooltip = {
        "text": "{title}\n{faculty}\n{year}\n{source}",
    }

    return pdk.Deck(
        layers=[scatter],
        initial_view_state=view_state,
        tooltip=tooltip,
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


def render_top_matches(df, matches_idx):
    if matches_idx is not None:
        st.subheader("Top matches")
        st.dataframe(df.loc[matches_idx, ["title", "year", "faculty", "source", "url"]])


if view == "UMAP (Pydeck)":
    if os.path.exists(arrow_path):
        df = load_arrow(arrow_path)
    else:
        df = load_umap(umap_path)

    st.pydeck_chart(build_umap_pydeck(df, matches))
    render_top_matches(df, matches)

elif view == "SOM (U-Matrix)":
    som_df = load_som(som_path)
    umatrix_df = load_umatrix(umatrix_path)
    fig = build_umatrix_figure(umatrix_df, som_df, matches)
    st.plotly_chart(fig, use_container_width=True)
    render_top_matches(som_df, matches)

else:
    left, right = st.columns(2)

    if os.path.exists(arrow_path):
        umap_df = load_arrow(arrow_path)
    else:
        umap_df = load_umap(umap_path)

    som_df = load_som(som_path)
    umatrix_df = load_umatrix(umatrix_path)

    with left:
        st.subheader("UMAP (Pydeck)")
        st.pydeck_chart(build_umap_pydeck(umap_df, matches))

    with right:
        st.subheader("SOM U-Matrix")
        st.plotly_chart(
            build_umatrix_figure(umatrix_df, som_df, matches),
            use_container_width=True,
        )

    render_top_matches(umap_df, matches)
