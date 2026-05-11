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

DEFAULTS = {
    "point_opacity": 0.7,
    "base_radius": 3.0,
    "base_radius_preset": "Medium",
    "show_legend": True,
    "match_mode": "Context",
    "query": "",
    "top_k": 20,
    "color_mode": "Faculty",
    "show_noise": True,
    "cluster_keywords_top_n": 5,
    "selected_faculties": [],
    "selected_sources": [],
    "selected_clusters": [],
    "year_range": None,
    "view": "UMAP (Pydeck)",
}

if "point_opacity" not in st.session_state:
    st.session_state.point_opacity = DEFAULTS["point_opacity"]
if "base_radius" not in st.session_state:
    st.session_state.base_radius = DEFAULTS["base_radius"]
if "base_radius_preset" not in st.session_state:
    st.session_state.base_radius_preset = DEFAULTS["base_radius_preset"]
if "show_legend" not in st.session_state:
    st.session_state.show_legend = DEFAULTS["show_legend"]
if "match_mode" not in st.session_state:
    st.session_state.match_mode = DEFAULTS["match_mode"]
if "query" not in st.session_state:
    st.session_state.query = DEFAULTS["query"]
if "top_k" not in st.session_state:
    st.session_state.top_k = DEFAULTS["top_k"]
if "color_mode" not in st.session_state:
    st.session_state.color_mode = DEFAULTS["color_mode"]
if "show_noise" not in st.session_state:
    st.session_state.show_noise = DEFAULTS["show_noise"]
if "cluster_keywords_top_n" not in st.session_state:
    st.session_state.cluster_keywords_top_n = DEFAULTS["cluster_keywords_top_n"]
if "selected_faculties" not in st.session_state:
    st.session_state.selected_faculties = DEFAULTS["selected_faculties"]
if "selected_sources" not in st.session_state:
    st.session_state.selected_sources = DEFAULTS["selected_sources"]
if "selected_clusters" not in st.session_state:
    st.session_state.selected_clusters = DEFAULTS["selected_clusters"]
if "year_range" not in st.session_state:
    st.session_state.year_range = DEFAULTS["year_range"]
if "view" not in st.session_state:
    st.session_state.view = DEFAULTS["view"]
if "map_view_key" not in st.session_state:
    st.session_state.map_view_key = 0
if "reset_ui_requested" not in st.session_state:
    st.session_state.reset_ui_requested = False

if st.session_state.reset_ui_requested:
    st.session_state.point_opacity = DEFAULTS["point_opacity"]
    st.session_state.base_radius = DEFAULTS["base_radius"]
    st.session_state.base_radius_preset = DEFAULTS["base_radius_preset"]
    st.session_state.show_legend = DEFAULTS["show_legend"]
    st.session_state.match_mode = DEFAULTS["match_mode"]
    st.session_state.query = DEFAULTS["query"]
    st.session_state.top_k = DEFAULTS["top_k"]
    st.session_state.color_mode = DEFAULTS["color_mode"]
    st.session_state.show_noise = DEFAULTS["show_noise"]
    st.session_state.cluster_keywords_top_n = DEFAULTS["cluster_keywords_top_n"]
    st.session_state.selected_faculties = DEFAULTS["selected_faculties"]
    st.session_state.selected_sources = DEFAULTS["selected_sources"]
    st.session_state.selected_clusters = DEFAULTS["selected_clusters"]
    st.session_state.year_range = DEFAULTS["year_range"]
    st.session_state.view = DEFAULTS["view"]
    st.session_state.map_view_key += 1
    st.session_state.reset_ui_requested = False

BASE_RADIUS_MAP = {"Low": 2.0, "Medium": 3.0, "High": 5.0}
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path

    cwd_candidate = os.path.abspath(path)
    root_candidate = os.path.abspath(os.path.join(PROJECT_ROOT, path))

    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return root_candidate


def stop_with_missing_file_message(label: str, path: str, build_step: str):
    st.error(f"Missing required file for {label}: {path}")
    st.info("Generate the artifact and rerun the app:")
    st.code(build_step, language="bash")
    st.stop()

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
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(parquet_path)
    return load_umap(parquet_path)


def apply_filters(df: pd.DataFrame, faculties, sources, clusters, year_range):
    filtered = df.copy()

    if faculties:
        filtered = filtered[filtered["faculty"].isin(faculties)]
    if sources:
        filtered = filtered[filtered["source"].isin(sources)]
    if clusters and "cluster" in filtered.columns:
        filtered = filtered[filtered["cluster"].isin(clusters)]
    if year_range and "year" in filtered.columns:
        filtered = filtered[filtered["year"].between(year_range[0], year_range[1])]

    return filtered


def build_categorical_palette(values):
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


def normalize_year_color(year_value, year_min, year_max):
    if year_value is None or pd.isna(year_value):
        return [120, 120, 120]
    if year_min == year_max:
        return [64, 160, 255]
    ratio = (int(year_value) - year_min) / (year_max - year_min)
    ratio = max(0.0, min(1.0, ratio))
    return [int(64 + ratio * 160), int(80 + ratio * 120), int(255 - ratio * 150)]


def compute_cluster_keywords(df, top_n: int = 5):
    keywords = {}
    if "cluster" not in df.columns:
        return keywords
    text_series = df.get("title", "").fillna("") + " " + df.get("abstract", "").fillna("")
    for cluster_id, group in df.groupby("cluster"):
        tokens = (
            text_series.loc[group.index]
            .str.lower()
            .str.replace(r"[^a-z0-9áéíóúüñ ]", " ", regex=True)
            .str.split()
        )
        counts = {}
        for words in tokens:
            for word in words:
                if len(word) < 4:
                    continue
                counts[word] = counts.get(word, 0) + 1
        top_words = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        keywords[cluster_id] = ", ".join([word for word, _ in top_words])
    return keywords


def add_density_radius(df: pd.DataFrame, bins: int = 80) -> pd.DataFrame:
    density_df = df.copy()
    x_bins = pd.cut(density_df["x"], bins=bins, labels=False, include_lowest=True)
    y_bins = pd.cut(density_df["y"], bins=bins, labels=False, include_lowest=True)
    density_df["density_bin"] = list(zip(x_bins, y_bins))
    counts = density_df["density_bin"].value_counts()
    density_df["density_count"] = density_df["density_bin"].map(counts).fillna(1)
    density_df["radius"] = density_df["density_count"].clip(1, 50).pow(0.5)
    return density_df


def render_top_matches(df, matches_idx):
    if matches_idx is not None:
        st.subheader("Top matches")
        st.dataframe(df.loc[matches_idx, ["title", "year", "faculty", "source", "url"]])


def render_details_panel(df: pd.DataFrame, matches_idx):
    if df.empty:
        return
    st.subheader("Details")
    candidate_df = df
    if matches_idx is not None and not df.loc[matches_idx].empty:
        candidate_df = df.loc[matches_idx]
    candidate_df = candidate_df.head(200)
    options = candidate_df.index.tolist()
    if not options:
        return
    selected = st.selectbox("Pick a record", options, index=0)
    row = candidate_df.loc[selected]
    st.markdown(f"**Title:** {row.get('title', '')}")
    st.markdown(f"**Year:** {row.get('year', '')}")
    st.markdown(f"**Faculty:** {row.get('faculty', '')}")
    st.markdown(f"**Source:** {row.get('source', '')}")
    if "cluster" in row:
        st.markdown(f"**Cluster:** {row.get('cluster', '')}")
    url = row.get("url", "")
    if url:
        st.markdown(f"**URL:** {url}")


def on_radius_preset_change():
    preset = st.session_state.base_radius_preset
    if preset in BASE_RADIUS_MAP:
        st.session_state.base_radius = BASE_RADIUS_MAP[preset]


umap_path_input = st.sidebar.text_input("UMAP parquet", "data/processed/unam_embeddings_2d.parquet")
arrow_path_input = st.sidebar.text_input("UMAP Arrow", "data/processed/unam_embeddings_2d.arrow")
som_path_input = st.sidebar.text_input("SOM parquet", "data/processed/som_map.parquet")
umatrix_path_input = st.sidebar.text_input(
    "SOM U-Matrix parquet", "data/processed/som_umatrix.parquet"
)
index_path_input = st.sidebar.text_input("FAISS index", "data/index/index.faiss")
model_name = st.sidebar.text_input(
    "Embedding model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

umap_path = resolve_path(umap_path_input)
arrow_path = resolve_path(arrow_path_input)
som_path = resolve_path(som_path_input)
umatrix_path = resolve_path(umatrix_path_input)
index_path = resolve_path(index_path_input)

view_options = ["UMAP (Pydeck)", "SOM (U-Matrix)", "Compare"]
view = st.sidebar.radio(
    "View",
    view_options,
    index=view_options.index(st.session_state.view),
    key="view",
)

st.sidebar.subheader("Semantic search")
query = st.sidebar.text_input("Search topic", key="query")
top_k = st.sidebar.slider("Top K", min_value=5, max_value=100, value=st.session_state.top_k, step=5, key="top_k")

st.sidebar.subheader("Map settings")
point_opacity = st.sidebar.slider("Point opacity", 0.1, 1.0, st.session_state.point_opacity, 0.05, key="point_opacity")
color_mode = st.sidebar.selectbox(
    "Color by",
    ["Faculty", "Source", "Cluster", "Year"],
    index=["Faculty", "Source", "Cluster", "Year"].index(st.session_state.color_mode),
    key="color_mode",
)
show_noise = st.sidebar.checkbox("Show cluster noise (-1)", value=st.session_state.show_noise, key="show_noise")
base_radius_preset = st.sidebar.selectbox(
    "Base radius preset",
    ["Low", "Medium", "High"],
    index=["Low", "Medium", "High"].index(st.session_state.base_radius_preset),
    key="base_radius_preset",
    on_change=on_radius_preset_change,
)
base_radius = st.sidebar.slider(
    "Base radius (advanced)",
    1.0,
    8.0,
    st.session_state.base_radius,
    0.5,
    key="base_radius",
)
show_legend = st.sidebar.checkbox("Show faculty legend", value=st.session_state.show_legend, key="show_legend")
match_mode = st.sidebar.radio(
    "Match mode",
    ["Context", "Only matches"],
    index=["Context", "Only matches"].index(st.session_state.match_mode),
    key="match_mode",
    help="Context keeps background points translucent. Only matches hides the rest.",
)

if st.sidebar.button("Reset UI (Master)"):
    st.session_state.reset_ui_requested = True
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

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


try:
    umap_df = load_umap_data(arrow_path, umap_path)
except FileNotFoundError:
    stop_with_missing_file_message(
        "UMAP",
        umap_path,
        "python scripts/03_umap_cluster.py --config config/default.yaml",
    )

umap_df = umap_df.reset_index(drop=True)
umap_df["row_id"] = umap_df.index

faculty_options = sorted(umap_df["faculty"].dropna().unique().tolist())
source_options = sorted(umap_df["source"].dropna().unique().tolist())

st.sidebar.subheader("Filters")
selected_faculties = st.sidebar.multiselect("Faculty", faculty_options, default=st.session_state.selected_faculties, key="selected_faculties")
selected_sources = st.sidebar.multiselect("Source", source_options, default=st.session_state.selected_sources, key="selected_sources")

cluster_options = []
if "cluster" in umap_df.columns:
    cluster_options = sorted(umap_df["cluster"].dropna().unique().tolist())
    selected_clusters = st.sidebar.multiselect(
        "Cluster",
        cluster_options,
        default=st.session_state.selected_clusters,
        key="selected_clusters",
    )
else:
    selected_clusters = []

year_range = None
if "year" in umap_df.columns and not umap_df["year"].dropna().empty:
    year_min = int(umap_df["year"].min())
    year_max = int(umap_df["year"].max())
    default_years = (year_min, year_max)
    candidate = st.session_state.year_range
    if (
        isinstance(candidate, (list, tuple))
        and len(candidate) == 2
        and candidate[0] is not None
        and candidate[1] is not None
    ):
        default_years = (int(candidate[0]), int(candidate[1]))
    else:
        st.session_state.year_range = default_years
    year_range = st.sidebar.slider("Year range", year_min, year_max, default_years, key="year_range")

match_set = set(matches) if matches is not None else set()

faculty_palette = build_categorical_palette(faculty_options)
source_palette = build_categorical_palette(source_options)

if show_legend and color_mode == "Faculty" and faculty_options:
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

if show_legend and color_mode == "Source" and source_options:
    legend_rows = []
    for source in source_options:
        color = source_palette.get(source, [120, 120, 120])
        legend_rows.append(
            f"""
            <div style='display:flex;align-items:center;margin-bottom:4px;'>
                <div style='width:12px;height:12px;background-color:rgb({color[0]},{color[1]},{color[2]});margin-right:6px;border:1px solid #555;'></div>
                <span style='font-size:12px;'>{source}</span>
            </div>
            """
        )
    st.sidebar.markdown("".join(legend_rows), unsafe_allow_html=True)


def build_umap_pydeck(df, matches_idx, opacity_value: float, radius_scale: float, mode: str):
    plot_df = df.copy()
    alpha = int(255 * opacity_value)

    if not plot_df.empty:
        plot_df = add_density_radius(plot_df)
    else:
        plot_df["radius"] = 1.0

    plot_df["final_radius"] = plot_df["radius"] * radius_scale * 10

    if mode == "Faculty":
        plot_df["color"] = plot_df["faculty"].map(faculty_palette)
    elif mode == "Source":
        plot_df["color"] = plot_df["source"].map(source_palette)
    elif mode == "Cluster" and "cluster" in plot_df.columns:
        cluster_values = sorted(plot_df["cluster"].dropna().unique().tolist())
        cluster_palette = build_categorical_palette(cluster_values)
        plot_df["color"] = plot_df["cluster"].map(cluster_palette)
    elif mode == "Year" and "year" in plot_df.columns:
        year_min = int(plot_df["year"].min()) if not plot_df["year"].dropna().empty else 0
        year_max = int(plot_df["year"].max()) if not plot_df["year"].dropna().empty else 0
        plot_df["color"] = plot_df["year"].apply(lambda y: normalize_year_color(y, year_min, year_max))
    else:
        plot_df["color"] = [[120, 120, 120]] * len(plot_df)

    plot_df["color"] = plot_df["color"].apply(
        lambda c: c if isinstance(c, list) else [120, 120, 120]
    )
    plot_df["color"] = plot_df["color"].apply(lambda c: [c[0], c[1], c[2], alpha])

    plot_df["is_match"] = plot_df["row_id"].isin(match_set)

    if matches_idx is not None:
        if match_mode == "Context":
            plot_df.loc[plot_df["is_match"], "color"] = plot_df.loc[plot_df["is_match"], "color"].apply(
                lambda c: [255, 70, 70, 255]
            )
            plot_df.loc[~plot_df["is_match"], "color"] = plot_df.loc[~plot_df["is_match"], "color"].apply(
                lambda c: [c[0], c[1], c[2], int(255 * 0.05)]
            )
        else:
            plot_df = plot_df[plot_df["is_match"]]

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position="[x, y]",
        get_radius="final_radius",
        radius_min_pixels=2,
        radius_max_pixels=15,
        get_fill_color="color",
        coordinate_system=pdk.constants.COORDINATE_SYSTEM.CARTESIAN,
        pickable=True,
    )

    view_state = pdk.ViewState(
        target=[
            float(plot_df["x"].mean()) if not plot_df.empty else 0.0,
            float(plot_df["y"].mean()) if not plot_df.empty else 0.0,
            0,
        ],
        zoom=2,
        min_zoom=-1,
        max_zoom=12,
    )

    tooltip = {
        "text": "{title}\n{faculty}\n{year}\n{source}\nCluster: {cluster}",
    }

    return pdk.Deck(
        layers=[scatter],
        views=[pdk.View(type="OrthographicView", controller=True)],
        initial_view_state=view_state,
        map_provider=None,
        map_style=None,
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
    filtered_umap = apply_filters(umap_df, selected_faculties, selected_sources, selected_clusters, year_range)
    if "cluster" in filtered_umap.columns:
        if not show_noise:
            filtered_umap = filtered_umap[filtered_umap["cluster"] != -1]
    st.pydeck_chart(
        build_umap_pydeck(filtered_umap, matches, point_opacity, base_radius, color_mode),
        key=f"umap_chart_{st.session_state.map_view_key}",
    )
    render_top_matches(filter_matches(filtered_umap), filter_matches(filtered_umap).index)
    render_details_panel(filter_matches(filtered_umap), filter_matches(filtered_umap).index)

elif view == "SOM (U-Matrix)":
    if not os.path.exists(som_path):
        stop_with_missing_file_message(
            "SOM map",
            som_path,
            "python scripts/04_som.py --config config/default.yaml",
        )
    if not os.path.exists(umatrix_path):
        stop_with_missing_file_message(
            "SOM U-Matrix",
            umatrix_path,
            "python scripts/04_som.py --config config/default.yaml",
        )

    som_df = load_som(som_path).reset_index(drop=True)
    som_df["row_id"] = som_df.index
    som_df = apply_filters(som_df, selected_faculties, selected_sources, selected_clusters, year_range)

    umatrix_df = load_umatrix(umatrix_path)
    fig = build_umatrix_figure(umatrix_df, som_df, matches)
    st.plotly_chart(fig, use_container_width=True)

    render_top_matches(filter_matches(som_df), filter_matches(som_df).index)

else:
    left, right = st.columns(2)

    filtered_umap = apply_filters(umap_df, selected_faculties, selected_sources, selected_clusters, year_range)

    if not os.path.exists(som_path):
        stop_with_missing_file_message(
            "SOM map",
            som_path,
            "python scripts/04_som.py --config config/default.yaml",
        )
    if not os.path.exists(umatrix_path):
        stop_with_missing_file_message(
            "SOM U-Matrix",
            umatrix_path,
            "python scripts/04_som.py --config config/default.yaml",
        )

    som_df = load_som(som_path).reset_index(drop=True)
    som_df["row_id"] = som_df.index
    som_df = apply_filters(som_df, selected_faculties, selected_sources, selected_clusters, year_range)

    umatrix_df = load_umatrix(umatrix_path)

    with left:
        st.subheader("UMAP (Pydeck)")
        st.pydeck_chart(
            build_umap_pydeck(filtered_umap, matches, point_opacity, base_radius, color_mode),
            key=f"umap_chart_{st.session_state.map_view_key}",
        )

    with right:
        st.subheader("SOM U-Matrix")
        st.plotly_chart(
            build_umatrix_figure(umatrix_df, som_df, matches),
            use_container_width=True,
        )

    render_top_matches(filter_matches(filtered_umap), filter_matches(filtered_umap).index)
    render_details_panel(filter_matches(filtered_umap), filter_matches(filtered_umap).index)