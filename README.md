# Semantic Research Atlas

Interactive knowledge cartography platform for exploring 200k+ scholarly documents from UNAM. Uses WebGL rendering via **DeepScatter** for fluid visualization of UMAP clusters and a **Self-Organizing Map (SOM)** for topographic analysis.

## Features
- **Ingestion** from OpenAlex (UNAM ROR), with support for UNAM Institutional Repository and SciELO (configurable)
- **Embeddings** with multilingual models (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Clustering** via UMAP dimensionality reduction + HDBSCAN
- **SOM** view with U-Matrix heatmap and density overlay
- **TF-IDF Semantic Labels** — automatic topic extraction per cluster
- **WebGL Frontend** (Vite + React + DeepScatter) with 3 interactive views:
  - **Atlas** — Full scatter plot with 200k+ points, floating topic labels, click-to-filter
  - **SOM** — U-Matrix heatmap with cluster-colored overlay and hover inspection
  - **Compare** — Side-by-side UMAP vs SOM view

## Repository Structure
```
config/               # YAML configs (sources, models, params)
frontend/             # React/Vite DeepScatter WebGL App
  src/App.jsx         # Main app with 3 views (Atlas, SOM, Compare)
  src/index.css       # Full design system
notebooks/            # Colab notebooks (GPU embeddings)
scripts/              # CLI pipeline steps (01-06)
src/semantic_research_atlas/  # Library code (utils)
```

## Quick Start
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

### 1. Ingest (OpenAlex)
```bash
python scripts/01_ingest.py --config config/default.yaml
```

### 2. Embed
```bash
# Local CPU (slow):
python scripts/02_embed.py --config config/default.yaml

# GPU via Colab (recommended): use notebooks/embed_gpu_colab.ipynb
# then import: python scripts/02_embed.py --config config/default.yaml --import-npy embeddings.npy
```

### 3. UMAP + HDBSCAN Clustering
```bash
python scripts/03_umap_cluster.py --config config/default.yaml
```

### 4. SOM (Self-Organizing Map)
```bash
python scripts/04_som.py --config config/default.yaml
```

### 5. Export DeepScatter Tiles
```bash
python scripts/05_export_tiles.py --config config/default.yaml
```

### 6. Generate Semantic Labels (TF-IDF)
```bash
python scripts/06_generate_labels.py --config config/default.yaml
```

## Running the Frontend

You need **two terminals**:

**Terminal A — Data Server (CORS):**
```bash
python scripts/serve_cors.py --port 8000
```

**Terminal B — Vite Dev Server:**
```bash
cd frontend
npm install  # first time only
npm run dev
```

Open `http://localhost:5173` and explore:
- **Atlas tab** — WebGL scatter plot with 200k+ points
- **SOM tab** — Interactive U-Matrix heatmap
- **Compare tab** — Side-by-side UMAP vs SOM

## Output Artifacts
| File | Description |
|------|-------------|
| `data/processed/records.parquet` | Clean records from ingestion |
| `data/processed/embeddings.npy` | 384-dim sentence embeddings |
| `data/processed/unam_embeddings_2d.parquet` | UMAP 2D coordinates + cluster labels |
| `data/processed/som_map.parquet` | SOM neuron assignments |
| `data/processed/som_umatrix.parquet` | U-Matrix distances |
| `data/tiles/` | Quadfeather tiles for DeepScatter |
| `data/tiles/cluster_labels.json` | TF-IDF topic labels and metadata |
| `data/tiles/som_umatrix.json` | SOM data for frontend |

## Configuration
All settings live in `config/default.yaml`.

```yaml
ingest:
  max_records_per_source: 0  # 0 = no limit
  polite_pause_seconds: 2.0

umap:
  n_neighbors: 30
  min_dist: 0.05

clustering:
  method: hdbscan
  min_cluster_size: 30

som:
  grid_x: 30
  grid_y: 20
  train_sample_size: 50000
```

## Notes
- For 200k+ records, embeddings should be computed on **GPU** (Google Colab T4 recommended).
- UMAP + HDBSCAN are optimized for **single-threaded, low-memory** execution to avoid OOM on consumer hardware.
- SOM trains on a subsample (default 50k) then maps the full dataset.
