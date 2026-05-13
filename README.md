# Semantic Research Atlas

Semantic clustering and interactive exploration of academic production from UNAM sources plus OpenAlex and SciELO. The project follows the same end-to-end flow as **jpbascur/text-similarity-maker**, adding a **Self-Organizing Map (SOM)** view for comparison.

## Features
- **Ingestion** from: UNAM Institutional Repository, OpenAlex, SciELO (configurable)
- **Massive Data Processing** using **DuckDB** for memory-efficient chunking and deduplication.
- **Embeddings** with multilingual models (Spanish-friendly)
- **Clustering** (UMAP + HDBSCAN)
- **SOM** view (U-Matrix + neuron mapping)
- **Interactive Apps**:
  - Legacy Streamlit app (UMAP, SOM, FAISS semantic search)
  - **Modern WebGL frontend** (Vite + React + DeepScatter) for rendering millions of points smoothly.

## Repository Structure
```
app/                  # Legacy Streamlit app
config/               # YAML configs (sources, models, params)
frontend/             # React/Vite DeepScatter WebGL App
notebooks/            # Optional experiments
scripts/              # CLI pipeline steps
src/semantic_research_atlas/  # Library code
```

## Quick Start
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pipeline (Local/Colab)
1. **Ingest**
```bash
python scripts/01_ingest.py --config config/default.yaml
```
2. **Embed**
```bash
python scripts/02_embed.py --config config/default.yaml
```
3. **UMAP + Clustering (Parquet + Arrow)**
```bash
python scripts/03_umap_cluster.py --config config/default.yaml
```
4. **SOM**
```bash
python scripts/04_som.py --config config/default.yaml
```
5. **Build FAISS Index**
```bash
python scripts/05_faiss.py --config config/default.yaml
```
6. **Export DeepScatter Tiles (WebGL)**
```bash
python scripts/06_export_deepscatter.py --config config/default.yaml
```

## Apps

### 1. WebGL DeepScatter Frontend (Recommended)
This uses the tiled feather files to render millions of points using WebGL. You need two terminals:
**Terminal A (CORS Data Server):**
```bash
python scripts/serve_cors.py --port 8000
```
**Terminal B (Vite Dev Server):**
```bash
cd frontend
npm install  # Solo la primera vez
npm run dev
```

### 2. Legacy Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## Output Artifacts
- `data/processed/unam_embeddings_2d.parquet`
- `data/processed/unam_embeddings_2d.arrow`
- `data/processed/som_map.parquet`
- `data/processed/som_umatrix.parquet`
- `data/index/index.faiss`
- `data/tiles/` (Quadfeather structure for DeepScatter)

## Notes
- For 300k records, **Arrow + FAISS** is recommended.
- For SOM, train on a **subsample (e.g., 50k)** then map the full set.

## Configuration
All settings live in `config/default.yaml`.

Recommended OAI-PMH defaults for slower servers:
```yaml
ingest:
  max_records_per_source: 2000
  polite_pause_seconds: 2.0
  request_timeout_seconds: 120
  request_max_retries: 6
  retry_backoff_seconds: 3
```

---

If you want this deployed to **Hugging Face Spaces**, Streamlit works out-of-the-box.
