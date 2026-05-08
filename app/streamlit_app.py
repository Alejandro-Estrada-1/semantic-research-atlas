import os
import streamlit as st
import pandas as pd
import plotly.express as px
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


def load_umap_data(arrow_path: str, parquet_path: str) -> pd.DataFrame:
    if os.path.exists(arrow_path):
        return load_arrow(arrow_path)
    return load_umap(parquet_path)


def apply_filters(df: pd.DataFrame, faculties, sources, year_range):
    filtered = df.copy()

    if faculties:
        filtered = filtered[filtered["faculty"].isin(faculties)]
    if sources:
        filtered = filtered[filtered["source"].isin(sources)]
    if year_range and "year" in filtered.columns:
        filtered = filtered[filtered["year"].between(year_range[0], year_range[1])]

    return filtered


def build_faculty_palette(values):
    palette = [
        [141, 211, 199],
        [255, 255, 179],
        [190, 186, 218],
        [251, 128, 114],
        [128, 177, 211],
        [253, 180, 98],
        [179, 222, 105],
        [252, 205, 229],
        [217, 217, 217],
        [188, 128, 189],
        [204, 235, 197],
        [255, 237, 111],
    ]
    mapping = {}
    for idx, val in enumerate(values):
        mapping[val] = palette[idx % len(palette)]
    return mapping


def add_density_radius(df: pd.DataFrame, bins: int = 80) -> pd.DataFrame:
    density_df = df.copy()
    x_bins = pd.cut(density_df["x"], bins=bins, labels=False, include_lowest=True)
    y_bins = pd.cut(density_df["y"], bins=bins, labels=False, include_lowest=True)
    density_df["density_bin"] = list(zip(x_bins, y_bins))
    counts = density_df["density_bin"].value_counts()
    density_df["density_count"] = density_df["density_bin"].map(counts).fillna(1)
    density_df["radius"] = density_df["density_count"].clip(1, 50)
    return density_df


def render_top_matches(df, matches_idx):
    if matches_idx is not None:
        st.subheader("Top matches")
        st.dataframe(df.loc[matches_idx, ["title", "year", "faculty", "source", "url"]])


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

st.sidebar.subheader("Map settings")
point_opacity = st.sidebar.slider("Point opacity", 0.1, 1.0, 0.7, 0.05)
show_legend = st.sidebar.checkbox("Show faculty legend", value=True)
match_mode = st.sidebar.radio(
    "Match mode",
    ["Context", "Only matches"],
    index=0,
    help="Context keeps background points translucent. Only matches hides the rest.",
)

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


umap_df = load_umap_data(arrow_path, umap_path)
umap_df = umap_df.reset_index(drop=True)
umap_df["row_id"] = umap_df.index

faculty_options = sorted(umap_df["faculty"].dropna().unique().tolist())
source_options = sorted(umap_df["source"].dropna().unique().tolist())

st.sidebar.subheader("Filters")
selected_faculties = st.sidebar.multiselect("Faculty", faculty_options)
selected_sources = st.sidebar.multiselect("Source", source_options)

year_range = None
if "year" in umap_df.columns and not umap_df["year"].dropna().empty:
    year_min = int(umap_df["year"].min())
    year_max = int(umap_df["year"].max())
    year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))

match_set = set(matches) if matches is not None else set()

faculty_palette = build_faculty_palette(faculty_options)

if show_legend and faculty_options:
    legend_rows = []
    for faculty in faculty_options:
        color = faculty_palette.get(faculty, [120, 120, 120])
        legend_rows.append(
            f"""
            <div style='display:flex;align-items:center;margin-bottom:4px;'>
                <div style='width:12px;height:12px;background-color:rgb({color[0]},{color[1]},{color[2]});margin-right:6px;border:1px solid #555;'></div>
                <span style='font-size:12px;'>{faculty}</span>
            </div>
            """
        )
    st.sidebar.markdown("".join(legend_rows), unsafe_allow_html=True)


def build_umap_pydeck(df, matches_idx, opacity_value: float):
    plot_df = df.copy()
    alpha = int(255 * opacity_value)

    plot_df["color"] = plot_df["faculty"].map(faculty_palette).fillna([120, 120, 120])
    plot_df["color"] = plot_df["color"].apply(lambda c: [c[0], c[1], c[2], alpha])

    plot_df["is_match"] = plot_df["row_id"].isin(match_set)
    plot_df["line_color"] = plot_df["is_match"].apply(lambda v: [255, 0, 0] if v else [0, 0, 0])
    plot_df["line_width"] = plot_df["is_match"].apply(lambda v: 3 if v else 0)

    if matches_idx is not None:
        if match_mode == "Context":
            plot_df.loc[plot_df["is_match"], "color"] = plot_df.loc[plot_df["is_match"], "color"].apply(
                lambda c: [255, 70, 70, 255]
            )
            plot_df.loc[~plot_df["is_match"], "color"] = plot_df.loc[~plot_df["is_match"], "color"].apply(
                lambda c: [c[0], c[1], c[2], int(255 * 0.15)]
            )
        else:
            plot_df = plot_df[plot_df["is_match"]]

    plot_df = add_density_radius(plot_df)

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position="[x, y]",
        get_radius="radius",
        radius_min_pixels=1,
        radius_max_pixels=10,
        get_fill_color="color",
        get_line_color="line_color",
        get_line_width="line_width",
        pickable=True,
    )

    view_state = pdk.ViewState(
        longitude=float(plot_df["x"].mean()) if not plot_df.empty else 0.0,
        latitude=float(plot_df["y"].mean()) if not plot_df.empty else 0.0,
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
        som_matches = som_df[som_df["row_id"].isin(match_set)]
        fig.add_scatter(
            x=som_matches["som_x"],
            y=som_matches["som_y"],
            mode="markers",
            marker=dict(color="red", size=6),
            name="Match",
        )

    return fig


def filter_matches(df):
    if matches is None:
        return df
    return df[df["row_id"].isin(match_set)]


if view == "UMAP (Pydeck)":
    filtered_umap = apply_filters(umap_df, selected_faculties, selected_sources, year_range)
    st.pydeck_chart(build_umap_pydeck(filtered_umap, matches, point_opacity))
    render_top_matches(filter_matches(filtered_umap), filter_matches(filtered_umap).index)

elif view == "SOM (U-Matrix)":
    som_df = load_som(som_path).reset_index(drop=True)
    som_df["row_id"] = som_df.index
    som_df = apply_filters(som_df, selected_faculties, selected_sources, year_range)

    umatrix_df = load_umatrix(umatrix_path)
    fig = build_umatrix_figure(umatrix_df, som_df, matches)
    st.plotly_chart(fig, use_container_width=True)

    render_top_matches(filter_matches(som_df), filter_matches(som_df).index)

else:
    left, right = st.columns(2)

    filtered_umap = apply_filters(umap_df, selected_faculties, selected_sources, year_range)

    som_df = load_som(som_path).reset_index(drop=True)
    som_df["row_id"] = som_df.index
    som_df = apply_filters(som_df, selected_faculties, selected_sources, year_range)

    umatrix_df = load_umatrix(umatrix_path)

    with left:
        st.subheader("UMAP (Pydeck)")
        st.pydeck_chart(build_umap_pydeck(filtered_umap, matches, point_opacity))

    with right:
        st.subheader("SOM U-Matrix")
        st.plotly_chart(
            build_umatrix_figure(umatrix_df, som_df, matches),
            use_container_width=True,
        )

    render_top_matches(filter_matches(filtered_umap), filter_matches(filtered_umap).index)
